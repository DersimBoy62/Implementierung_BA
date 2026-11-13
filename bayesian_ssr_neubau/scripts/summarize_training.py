"""Post-training audit; reads no quality labels and never supplies training state."""
from pathlib import Path
import hashlib
import json
import numpy as np

root = Path(__file__).resolve().parents[2]
project = root/'bayesian_ssr_neubau'
bundle = project/'outputs/baseline_8L4U_001'
model = json.loads((bundle/'model.json').read_text(encoding='utf-8'))
history,events = model['history'],model['update_history']
reference_path = root/'results/v60_math_m2/V60_INDUCED_PRIOR_M2_30E_01.json'
reference = json.loads(reference_path.read_text(encoding='utf-8'))
new_curve = np.array([h['elbo'] for h in history])
old_curve = np.array([h['elbo'] for h in reference['history']])
process11 = next(e for e in events if e['epoch']==11 and e['stage']=='process')
components11 = {name:process11['components_after'][name]-process11['components_before'][name]
                for name in ('emission_NW','transition_Dirichlet','initial_Dirichlet')}
min_gain = min(e['delta'] for e in events)
report = dict(status='fresh 30-epoch training complete; no quality scoring',
    dataset_sha256=model['dataset_sha256'],model_sha256=hashlib.sha256((bundle/'model.json').read_bytes()).hexdigest(),
    config=model['config'],n_epochs=len(history),n_block_updates=len(events),
    first_epoch_J=history[0]['elbo'],last_epoch_J=history[-1]['elbo'],
    min_block_delta=min_gain,monotonic_atol=model['config']['monotonic_atol'],
    n_violations=sum(e['delta'] < -model['config']['monotonic_atol'] for e in events),
    final_weighted_N=history[-1]['state_N'],final_transition_mean=history[-1]['dirichlet_transition_mean'],
    epoch11_total_increase=history[10]['elbo']-history[9]['elbo'],
    epoch11_process_increase=process11['delta'],epoch11_process_components=components11,
    n_local_optimizers=sum(e['stage']=='local' and 'nit' in e['optimizer'] for e in events),
    n_local_accepted=sum(e['stage']=='local' and e['optimizer'].get('accepted',False) for e in events),
    optimizer_max_final_grad_inf=max(e['optimizer'].get('grad_inf',0) for e in events if e['stage']=='local'),
    optimizer_non_success=sum(e['stage']=='local' and 'optimizer_success' in e['optimizer'] and not e['optimizer']['optimizer_success'] for e in events),
    post_training_reference=dict(path=reference_path.relative_to(root).as_posix(),
        sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        use='read-only historical training-curve comparison after new fit; never input to training',
        max_abs_curve_difference=float(np.max(np.abs(new_curve-old_curve))),
        final_difference=float(new_curve[-1]-old_curve[-1])),
    limitation='Train objective and numerical parity are not independent quality/generalization evidence')
dest = project/'docs/TRAINING_30E_AUDIT.json'
with dest.open('x',encoding='utf-8') as handle:
    json.dump(report,handle,ensure_ascii=False,indent=2)
print(json.dumps(report,ensure_ascii=False,indent=2))
