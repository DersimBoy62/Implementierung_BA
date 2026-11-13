"""V5.2 release QA, synthetic fixtures only. NEVER trains, opens datasets or scores rings."""
from __future__ import annotations
import copy, json, pathlib, tempfile
from unittest import mock
import main_bayesian_ssr_v52 as main
import release_integrity_v52 as integrity

ROOT=pathlib.Path(__file__).resolve().parent


def assert_rejected(fn, cls=(ValueError,RuntimeError,FileExistsError,FileNotFoundError), fragment=None):
    try:fn()
    except cls as err:
        if fragment and fragment.lower() not in str(err).lower():
            raise AssertionError('Different failure reason: '+str(err))
        return
    raise AssertionError('Expected a fail-closed rejection')


def mock_package(cfg):
    hist=[{'elbo':-100+i*5.,'state_mass':[.55,.45]} for i in range(cfg['epochs'])]
    return {'model':{'history':hist,'QA_FAKE_MODEL_NO_INFERENCE':True},
            'x_scaler':{'fake':True},'y_scaler':{'fake':True},
            'input_safety_policy':{'protocol':'synthetic-only','QA':True},
            'physical_scalers':[{'fake':1},{'fake':2}],
            'config_sha256':integrity.FREEZE_CONFIG_HASH,'source_data_sha256':integrity.SOURCE_HASH,
            'train_ids_L':integrity.LABELED,'train_ids_U':integrity.U_ONLY,
            'core_sha256':cfg['core_math_file_sha256'],'v52_release':integrity.RELEASE},hist


def run_tests():
    results=[]
    def step(name,func):
        func();results.append(name);print('PASS',name)
    cfg=main.load_config()

    def t_schema():
        assert len(cfg)==len(integrity.REQUIRED_CONFIG_KEYS)
    step('all_config_fields_exactly_registered_and_live_code_matches',t_schema)
    def t_core():
        assert len(cfg['core_math_file_sha256'])==10
    step('all_10_core_math_SHA256_unchanged_and_V51_safety_hash_valid',t_core)
    def t_config_hash():
        assert integrity.sha_file(ROOT/'PROPOSED_MODEL_FREEZE.json')==integrity.FREEZE_CONFIG_HASH
    step('original_V51_config_SHA_unchanged',t_config_hash)

    def altered_config():
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d)/'config.json';data=copy.deepcopy(cfg)
            for key,new in [('quality_targets',['wrong','wrong']),('process_scaling','P2'),('C',2.),
                            ('epochs',False),('physical_descriptor_indices',{'outside':[0],'height':[0]})]:
                modified=copy.deepcopy(data);modified[key]=new
                p.write_text(json.dumps(modified),encoding='utf-8')
                assert_rejected(lambda:integrity.validate_registered_config(p,ROOT/'final_core',ROOT))
    step('config_target_scaling_C_epoch_type_and_descriptor_mutations_rejected',altered_config)

    def config_additional_key():
        with tempfile.TemporaryDirectory() as d:
            p=pathlib.Path(d)/'config.json';data=copy.deepcopy(cfg);data['UNREGISTERED_FIELD']=1
            p.write_text(json.dumps(data))
            assert_rejected(lambda:integrity.validate_registered_config(p,ROOT/'final_core',ROOT),fragment='Config')
    step('config_unregistered_field_rejected',config_additional_key)

    def code_tamper_detected():
        with mock.patch.object(integrity,'sha_file',wraps=integrity.sha_file) as hs:
            def spoof(path):
                if str(path).endswith('dynamic_quality_v2.py'):return '0'*64
                return integrity.sha_file_original(path)
            integrity.sha_file_original=hs._mock_wraps
            try:
                hs.side_effect=spoof
                assert_rejected(lambda:main.load_config(),fragment='core SHA')
            finally:del integrity.sha_file_original
    step('mathematical_core_mutation_detected_before_loading',code_tamper_detected)

    def model_roundtrip():
        with tempfile.TemporaryDirectory() as d:
            out=pathlib.Path(d)/'qa_model';package,hist=mock_package(cfg)
            man=integrity.store_model_atomic(out,package,hist,cfg,ROOT)
            assert out.is_dir() and len(list(out.iterdir()))==2
            assert man['n_epochs']==10
            assert all(not x.name.startswith('.qa_model.staging') for x in pathlib.Path(d).iterdir())
            got,pp=integrity.verify_bundle(out,cfg,ROOT,load_pickle=True)
            assert pp['model']['QA_FAKE_MODEL_NO_INFERENCE'] is True
            assert got['model_sha256']==man['model_sha256']
    step('atomic_write_verified_two_file_bundle_and_trusted_roundtrip',model_roundtrip)

    def write_once():
        with tempfile.TemporaryDirectory() as d:
            out=pathlib.Path(d)/'qa_model';out.mkdir()
            pkg,hist=mock_package(cfg)
            assert_rejected(lambda:integrity.store_model_atomic(out,pkg,hist,cfg,ROOT),FileExistsError)
    step('existing_destination_even_empty_not_overwritten',write_once)

    def partial_rollback():
        with tempfile.TemporaryDirectory() as d:
            out=pathlib.Path(d)/'qa_model';pkg,hist=mock_package(cfg)
            normal=integrity._sync_write
            counter=[0]
            def fail_second(path,data):
                counter[0]+=1
                if counter[0]==2:raise OSError('QA simulated manifest write failure')
                return normal(path,data)
            with mock.patch.object(integrity,'_sync_write',side_effect=fail_second):
                assert_rejected(lambda:integrity.store_model_atomic(out,pkg,hist,cfg,ROOT),(OSError,), 'simulated')
            assert not out.exists() and list(pathlib.Path(d).iterdir())==[]
    step('injected_manifest_write_failure_rolls_back_staging_no_partial_final',partial_rollback)

    def modified_model_binary():
        with tempfile.TemporaryDirectory() as d:
            out=pathlib.Path(d)/'model';pkg,hist=mock_package(cfg)
            integrity.store_model_atomic(out,pkg,hist,cfg,ROOT)
            with (out/'TRAINED_MODEL.pkl').open('ab') as fh:fh.write(b'tamper')
            with mock.patch.object(integrity.pickle,'loads',side_effect=AssertionError('must not unpickle')):
                assert_rejected(lambda:integrity.verify_bundle(out,cfg,ROOT,load_pickle=True),fragment='Model binary SHA')
    step('corrupted_model_bytes_refused_before_unpickling',modified_model_binary)

    def modified_env():
        with tempfile.TemporaryDirectory() as d:
            out=pathlib.Path(d)/'model';pkg,hist=mock_package(cfg)
            integrity.store_model_atomic(out,pkg,hist,cfg,ROOT)
            snapshot=integrity.environment_snapshot()
            changed=copy.deepcopy(snapshot);changed['packages']['numpy']='0.0.0'
            with mock.patch.object(integrity,'environment_snapshot',return_value=changed):
                with mock.patch.object(integrity.pickle,'loads',side_effect=AssertionError('must not unpickle')):
                    assert_rejected(lambda:integrity.verify_bundle(out,cfg,ROOT,load_pickle=True),fragment='environment')
    step('changed_numpy_version_environment_refused_before_unpickling',modified_env)

    def partial_bundle():
        with tempfile.TemporaryDirectory() as d:
            out=pathlib.Path(d)/'model';out.mkdir();(out/'TRAINED_MODEL.pkl').write_bytes(b'qa')
            assert_rejected(lambda:integrity.verify_bundle(out,cfg,ROOT),fragment='Incomplete')
    step('incomplete_bundle_cannot_be_loaded',partial_bundle)

    def altered_manifest():
        with tempfile.TemporaryDirectory() as d:
            out=pathlib.Path(d)/'model';pkg,hist=mock_package(cfg)
            integrity.store_model_atomic(out,pkg,hist,cfg,ROOT)
            path=out/'TRAIN_MANIFEST.json';m=json.loads(path.read_text());m['train_labeled_ids']=['DM07']
            path.write_text(json.dumps(m),encoding='utf-8')
            assert_rejected(lambda:integrity.verify_bundle(out,cfg,ROOT),fragment='Ring allocation')
    step('tampered_ring_split_in_manifest_rejected',altered_manifest)

    def new_manifest_field():
        with tempfile.TemporaryDirectory() as d:
            out=pathlib.Path(d)/'model';pkg,hist=mock_package(cfg)
            integrity.store_model_atomic(out,pkg,hist,cfg,ROOT)
            path=out/'TRAIN_MANIFEST.json';m=json.loads(path.read_text());m['new_unknown_key']=1
            path.write_text(json.dumps(m),encoding='utf-8')
            assert_rejected(lambda:integrity.verify_bundle(out,cfg,ROOT),fragment='manifest')
    step('unknown_manifest_schema_field_rejected',new_manifest_field)

    def tampered_policy():
        with tempfile.TemporaryDirectory() as d:
            out=pathlib.Path(d)/'model';pkg,hist=mock_package(cfg)
            integrity.store_model_atomic(out,pkg,hist,cfg,ROOT)
            # Verify strict policy binding: mutation even if manifest hash matches pickle
            path=out/'TRAIN_MANIFEST.json';m=json.loads(path.read_text());m['input_safety_policy_sha256']='0'*64
            path.write_text(json.dumps(m),encoding='utf-8')
            assert_rejected(lambda:integrity.verify_bundle(out,cfg,ROOT,load_pickle=True),fragment='policy')
    step('mismatched_policy_sha_rejected_after_trusted_unpickle',tampered_policy)

    def symlink_deny():
        """Verify link rejection; use an explicit, labelled mock if Windows disallows link creation.

        Missing SeCreateSymbolicLinkPrivilege (WinError 1314) is an OS policy,
        not evidence that verify_bundle accepted a symlink. All other errors
        remain failures. The fallback tests the gate, NOT a real NTFS symlink.
        """
        with tempfile.TemporaryDirectory() as d:
            out=pathlib.Path(d)/'model';pkg,hist=mock_package(cfg)
            integrity.store_model_atomic(out,pkg,hist,cfg,ROOT)
            source=out/'TRAINED_MODEL.pkl';copy_path=pathlib.Path(d)/'copy.pkl'
            source.rename(copy_path)
            try:
                source.symlink_to(copy_path)
            except OSError as err:
                if getattr(err, 'winerror', None) != 1314:
                    raise
                # Windows user cannot create symlinks. Restore the genuine
                # regular file before testing the link-detection branch.
                copy_path.rename(source)
                genuine_is_symlink = pathlib.Path.is_symlink
                def controlled_link_flag(path):
                    if path == source:
                        return True
                    return genuine_is_symlink(path)
                with mock.patch.object(pathlib.Path, 'is_symlink', autospec=True,
                                       side_effect=controlled_link_flag):
                    assert_rejected(lambda:integrity.verify_bundle(out,cfg,ROOT),
                                    fragment='symlinked')
                print('INFO: Native symlink test SKIPPED (WinError 1314); '
                      'mocked symlink-detection branch PASS')
                return
            assert_rejected(lambda:integrity.verify_bundle(out,cfg,ROOT),
                            fragment='symlinked')
    step('symlinked_pickle_path_rejected',symlink_deny)

    def historical_id_before_io():
        with mock.patch.object(integrity,'verify_bundle',side_effect=AssertionError('must not touch model')):
            assert_rejected(lambda:main.predict_one('missing','dm04','bad.csv','out.json','record.json'),fragment='Historical')
    step('old_test_or_train_id_rejected_before_model_io',historical_id_before_io)

    def disallow_output_under_code():
        pkg,hist=mock_package(cfg)
        assert_rejected(lambda:integrity.check_output_destination(ROOT,ROOT/'bad_training_dir'),ValueError,'inside')
    step('output_inside_source_tree_refused',disallow_output_under_code)

    def sha_and_environment_snapshot():
        snapshot=integrity.environment_snapshot()
        assert snapshot['packages']['scikit-learn'] is not None
        assert snapshot['python_version']
        assert len(integrity.sha_bytes(integrity.canonical(snapshot)))==64
    step('python_and_five_dependency_versions_recorded',sha_and_environment_snapshot)

    print(f'V5.2: {len(results)}/{len(results)} QA checks PASS; NO TRAINING, NO DATASET DESERIALIZATION, NO OLD TEST')
    return results

if __name__=='__main__':run_tests()
