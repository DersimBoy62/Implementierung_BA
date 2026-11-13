"""V5.2 release and artifact integrity, strictly pre-freeze and development-only.

Two layers: (1) exact registered config+code SHA validation, (2) write-once
transactional model bundle with verification BEFORE unpickling. This is NOT
cryptographic authentication against someone who can replace all local files.
Never read untrusted pickles. No inference or training takes place here.
"""
from __future__ import annotations
import hashlib, json, os, pathlib, platform, shutil, sys, tempfile, pickle, struct
from datetime import datetime, timezone
from importlib import metadata

RELEASE='DM-V52-CONFIG-AND-ARTIFACT-2026-11-12-v1'
FREEZE_CONFIG_HASH='88f9da6aa51be71db7968a8222cb36e5e4838b8a8431988bd7c54c09bace51ab'
V51_SAFETY_HASH='3af23eacf61151b106c9135ef5674c455ae6c80d8877007355a975b440f48af9'
V51_WRAPPER_HASH='6d8f99e6b800d1b931916c565df698c9c2c024e7496a405b3d57a02039856286'
REQUIRED_CONFIG_KEYS=frozenset('''freeze_id freeze_status scientific_evidence dataset_sha256 labeled_ring_ids unlabeled_ring_ids never_use_for_selection_ids input_feature_set input_columns process_window process_scaling quality_targets target_scaling physical_descriptor_indices physical_descriptor_scaling model K C physical_var state_var_source epochs local_steps random_seed core_math_file_sha256 alternative_models pending_research_checks'''.split())
FEATURES=('F_rad63','F_ax64','Ist_D150','Ist_s151','Ist_h152','RWG55')
TARGETS=('ovality.ovality_radial_mm','planarity.height_deviation_mm')
LABELED=('DM04','DM05','DM06','DM09','DM12','DM16','DM17','DM18')
U_ONLY=('DM01','DM02','DM10','DM14')
LOCKED=('DM07','DM08','DM11','DM15')
SOURCE_HASH='bf29db68e7d89b14bfe0cc7b75915abd4414fc05221e2258015f80a79dcf1db5'
CORE_NAMES=('compare_dynamic_quality_cv4.py','compare_two_head_cv4.py','dm_data.py','dynamic_markov_v2.py','dynamic_quality_v2.py','full_markov_b1.py','fullcov_b1.py','quality_link_challenge.py','two_head_b1.py','vb_b1.py')
PACKAGE_KEYS=frozenset('model x_scaler y_scaler input_safety_policy physical_scalers config_sha256 source_data_sha256 train_ids_L train_ids_U core_sha256 v52_release'.split())
MANIFEST_KEYS=frozenset('''schema release freeze_id status trained_at_utc training_input_sha256 config_sha256 model_sha256 input_safety_policy_sha256 original_core_math_hashes release_file_hashes train_labeled_ids train_U_process_only_ids locked_old_test_ids_not_used final_elbo final_state_mass n_epochs environment environment_sha256 artifact_format caveats'''.split())


def canonical(obj):
    return json.dumps(obj,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(',',':')).encode('utf-8')


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h=hashlib.sha256()
    with pathlib.Path(path).open('rb') as fh:
        for blob in iter(lambda:fh.read(1024*1024),b''):h.update(blob)
    return h.hexdigest()


def _must(ok, reason):
    if not ok:raise ValueError(reason)


def validate_registered_config(config_path, core_dir, root):
    """Hash lock means EVERY key and value of the registered config is pinned.

    Additional schema/semantics checks provide clear diagnostics and catch accidental
    changes in assumptions, option layout and imported kernels.
    """
    path=pathlib.Path(config_path)
    cfg=json.loads(path.read_text(encoding='utf-8'))
    _must(isinstance(cfg,dict) and set(cfg)==REQUIRED_CONFIG_KEYS,'Config schema keys mismatch')
    _must(type(cfg['K']) is int and cfg['K']==2, 'K changed or invalid')
    _must(type(cfg['C']) is float and cfg['C']==0.2, 'C changed or invalid')
    _must(type(cfg['physical_var']) is float and cfg['physical_var']==0.38526108747593474,'Physical prior changed or invalid')
    _must(type(cfg['epochs']) is int and cfg['epochs']==10,'Epochenzahl changed')
    _must(type(cfg['local_steps']) is int and cfg['local_steps']==8,'Local VI steps changed')
    _must(type(cfg['random_seed']) is int and cfg['random_seed']==20261111,'Training seed changed')
    for key,val in [('labeled_ring_ids',LABELED),('unlabeled_ring_ids',U_ONLY),('never_use_for_selection_ids',LOCKED),('input_columns',FEATURES),('quality_targets',TARGETS)]:
        _must(type(cfg[key]) is list and tuple(cfg[key])==val,key+' mismatch')
    _must(cfg['dataset_sha256']==SOURCE_HASH,'Training dataset SHA mismatch')
    _must(cfg['freeze_status']=='PROVISIONAL_USER_APPROVAL_PENDING','Model freeze was not approved')
    _must(cfg['freeze_id']=='DM-V5-MAIN-PROPOSED-2026-11-12-v0','Unexpected model ID')
    _must(cfg['input_feature_set']=='V1_Kern_6' and cfg['process_window']=='F2','Process feature/window mismatch')
    _must(cfg['process_scaling']=='P0 pooled timepoint mean/std over labeled and U training rings only','X scaling mismatch')
    _must(cfg['target_scaling']=='train-labeled only mean/std, separately for two heads','Y scaling mismatch')
    _must(cfg['physical_descriptor_scaling']=='standard mean/std over labeled and U training rings only','Descriptor scaling mismatch')
    _must(cfg['physical_descriptor_indices']=={'outside':[0,1,3,5,6,10],'height':[0,2,4,7,11]},'Descriptor mapping mismatch')
    _must(cfg['model']=='full-covariance Bayesian Gaussian-Wishart HMM with M2 late-minus-early state occupancy and separate Bayesian NIG quality heads','Model schema mismatch')
    _must(cfg['state_var_source']=='dynamic_quality_v2.Spec.prior_variances','Prior implementation mismatch')
    _must(type(cfg['alternative_models']) is list and len(cfg['alternative_models'])==2,'Alternative register mismatch')
    _must(type(cfg['pending_research_checks']) is list and len(cfg['pending_research_checks'])==4,'Pending checks register mismatch')
    hashes=cfg['core_math_file_sha256']
    _must(isinstance(hashes,dict) and set(hashes)==set(CORE_NAMES),'10-core-module register mismatch')
    for name in CORE_NAMES:
        _must(sha_file(pathlib.Path(core_dir)/name)==hashes[name],'Mathematical core SHA mismatch: '+name)
    _must(sha_file(pathlib.Path(root)/'input_safety_v51.py')==V51_SAFETY_HASH,'V5.1 input safety SHA mismatch')
    _must(sha_file(pathlib.Path(root)/'main_bayesian_ssr_v51.py')==V51_WRAPPER_HASH,'V5.1 reference wrapper SHA mismatch')
    _must(sha_file(path)==FREEZE_CONFIG_HASH,'Registered config byte SHA mismatch (even if otherwise parseable)')
    return cfg


def environment_snapshot():
    packages={}
    for name in ('numpy','scipy','scikit-learn','numba','threadpoolctl'):
        try:packages[name]=metadata.version(name)
        except metadata.PackageNotFoundError:packages[name]=None
    return {'python_version':platform.python_version(),'python_implementation':platform.python_implementation(),
            'operating_system':platform.system(),'machine':platform.machine(),
            'byteorder':sys.byteorder,'word_bits':struct.calcsize('P')*8,
            'packages':packages}


def release_file_hashes(root):
    root=pathlib.Path(root)
    return {name:sha_file(root/name) for name in ('main_bayesian_ssr_v52.py','release_integrity_v52.py','input_safety_v51.py','main_bayesian_ssr_v51.py')}


def check_output_destination(root,out):
    root=pathlib.Path(root).resolve()
    out=pathlib.Path(out).absolute()
    if out.is_symlink() or out.exists():raise FileExistsError('Output path must not exist: write-once atomic release')
    if out==root or root in out.parents:raise ValueError('Output must not be created inside a source directory')
    if not out.parent.is_dir():raise FileNotFoundError('Output parent directory must already exist')
    if out.parent.is_symlink():raise ValueError('Symlinked destination parent is not supported')
    return out


def _sync_write(path,data):
    with pathlib.Path(path).open('xb') as fh:
        fh.write(data);fh.flush();os.fsync(fh.fileno())


def _sync_parent(path):
    """Directory fsync where OS permits, otherwise atomic rename remains basic guarantee."""
    if os.name!='posix':return
    flags=os.O_RDONLY | getattr(os,'O_DIRECTORY',0)
    fd=os.open(str(path),flags)
    try:os.fsync(fd)
    finally:os.close(fd)


def _verify_manifest(manifest,cfg,root,env,strict_environment=True):
    _must(isinstance(manifest,dict) and set(manifest)==MANIFEST_KEYS,'Unknown or incomplete V5.2 manifest')
    _must(manifest['schema']=='DM-V52-MODEL-MANIFEST-v1' and manifest['release']==RELEASE,'Unknown release')
    _must(manifest['freeze_id']==cfg['freeze_id'] and manifest['status']=='PROVISIONAL_DEVELOPMENT_ONLY','Wrong research status')
    _must(manifest['training_input_sha256']==SOURCE_HASH and manifest['config_sha256']==FREEZE_CONFIG_HASH,'Model/dataset config provenance mismatch')
    _must(tuple(manifest['train_labeled_ids'])==LABELED and tuple(manifest['train_U_process_only_ids'])==U_ONLY and tuple(manifest['locked_old_test_ids_not_used'])==LOCKED,'Ring allocation mismatch')
    _must(manifest['original_core_math_hashes']==cfg['core_math_file_sha256'],'Math hash register changed')
    _must(manifest['release_file_hashes']==release_file_hashes(root),'Release code has changed since training')
    _must(manifest['artifact_format']=='trusted-local-pickle-v1','Unexpected model format')
    _must(type(manifest['n_epochs']) is int and manifest['n_epochs']==10,'Epoch manifest changed')
    _must(type(manifest['final_elbo']) in (float,int) and __import__('math').isfinite(manifest['final_elbo']),'Nonfinite ELBO in manifest')
    _must(type(manifest['model_sha256']) is str and len(manifest['model_sha256'])==64,'Invalid model hash')
    _must(type(manifest['input_safety_policy_sha256']) is str and len(manifest['input_safety_policy_sha256'])==64,'Invalid policy hash')
    _must(manifest['environment_sha256']==sha_bytes(canonical(manifest['environment'])),'Environment hash mismatch')
    if strict_environment:_must(manifest['environment']==env,'Runtime environment changed: exact release environment required')
    return True


def store_model_atomic(out,package,history,cfg,root):
    """Atomically commit one complete model+manifest directory (not checkpoints).

    A hard crash may leave a hidden staging directory, but the final destination
    is never populated with an incomplete pair in ordinary single-writer operation.
    """
    root=pathlib.Path(root).resolve();dest=check_output_destination(root,out)
    env=environment_snapshot()
    _must(set(package)==PACKAGE_KEYS,'Model package schema mismatch')
    _must(package['config_sha256']==FREEZE_CONFIG_HASH and package['source_data_sha256']==SOURCE_HASH,'Package provenance mismatch')
    _must(tuple(package['train_ids_L'])==LABELED and tuple(package['train_ids_U'])==U_ONLY,'Package split mismatch')
    _must(package['core_sha256']==cfg['core_math_file_sha256'] and package['v52_release']==RELEASE,'Package core/release mismatch')
    _must(type(history) is list and len(history)==cfg['epochs'],'Training history missing epochs')
    import math
    last=history[-1]
    _must(math.isfinite(float(last['elbo'])),'ELBO not finite')
    _must(isinstance(last['state_mass'],list) and len(last['state_mass'])==cfg['K'],'Invalid state_mass')
    raw=pickle.dumps(package,protocol=pickle.HIGHEST_PROTOCOL)
    policy_sha=sha_bytes(canonical(package['input_safety_policy']))
    manifest={'schema':'DM-V52-MODEL-MANIFEST-v1','release':RELEASE,'freeze_id':cfg['freeze_id'],
              'status':'PROVISIONAL_DEVELOPMENT_ONLY',
              'trained_at_utc':datetime.now(timezone.utc).isoformat(),
              'training_input_sha256':SOURCE_HASH,'config_sha256':FREEZE_CONFIG_HASH,
              'model_sha256':sha_bytes(raw),'input_safety_policy_sha256':policy_sha,
              'original_core_math_hashes':cfg['core_math_file_sha256'],
              'release_file_hashes':release_file_hashes(root),
              'train_labeled_ids':list(LABELED),'train_U_process_only_ids':list(U_ONLY),
              'locked_old_test_ids_not_used':list(LOCKED),
              'final_elbo':float(last['elbo']),'final_state_mass':[float(v) for v in last['state_mass']],
              'n_epochs':len(history),'environment':env,'environment_sha256':sha_bytes(canonical(env)),
              'artifact_format':'trusted-local-pickle-v1',
              'caveats':['No independent validation; outer four previously opened and excluded',
                         'Package SHA proves internal consistency only, not independent authenticity',
                         'Environment versions recorded; full bitwise determinism is not guaranteed']}
    _verify_manifest(manifest,cfg,root,env)
    tmp=None
    try:
        tmp=pathlib.Path(tempfile.mkdtemp(prefix='.'+dest.name+'.staging-',dir=str(dest.parent)))
        _sync_write(tmp/'TRAINED_MODEL.pkl',raw)
        _sync_write(tmp/'TRAIN_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False).encode('utf-8')+b'\n')
        _sync_parent(tmp)
        # Full verification of the temporary bundle before atomic commit.
        verify_bundle(tmp,cfg,root,strict_environment=True,load_pickle=False)
        if dest.exists() or dest.is_symlink():raise FileExistsError('Output appeared during staging; refusing to overwrite')
        os.replace(str(tmp),str(dest))
        tmp=None
        _sync_parent(dest.parent)
        return manifest
    finally:
        if tmp is not None and tmp.exists():shutil.rmtree(tmp)


def verify_bundle(directory,cfg,root,strict_environment=True,load_pickle=False):
    """Read only a model trusted to be generated by this project.

    Verify the exact bytes which will be unpickled, to avoid a hash-then-reopen
    window.  A manifest next to a pickle is NOT an external authenticity proof.
    """
    d=pathlib.Path(directory)
    _must(d.is_dir() and not d.is_symlink(),'Model directory missing/symlink')
    model_path=d/'TRAINED_MODEL.pkl';manifest_path=d/'TRAIN_MANIFEST.json'
    for path in (model_path,manifest_path):
        _must(path.is_file() and not path.is_symlink(),'Incomplete/symlinked model bundle')
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    _verify_manifest(manifest,cfg,root,environment_snapshot(),strict_environment)
    raw=model_path.read_bytes()
    _must(sha_bytes(raw)==manifest['model_sha256'],'Model binary SHA does not match manifest')
    if not load_pickle:return manifest
    # ONLY for a locally generated and trusted bundle, never for downloads or unknown archives.
    package=pickle.loads(raw)
    _must(isinstance(package,dict) and set(package)==PACKAGE_KEYS,'Unrecognized package layout')
    _must(package['v52_release']==RELEASE and package['config_sha256']==FREEZE_CONFIG_HASH,'Wrong model release/config')
    _must(package['source_data_sha256']==SOURCE_HASH and tuple(package['train_ids_L'])==LABELED and tuple(package['train_ids_U'])==U_ONLY,'Model data split mismatch')
    _must(package['core_sha256']==cfg['core_math_file_sha256'],'Model kernel hashes mismatch')
    _must(sha_bytes(canonical(package['input_safety_policy']))==manifest['input_safety_policy_sha256'],'Stored safety policy mismatch')
    _must(type(package['physical_scalers']) is list and len(package['physical_scalers'])==2,'Two physical scalers required')
    return manifest,package
