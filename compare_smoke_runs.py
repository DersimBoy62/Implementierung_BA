"""Vergleicht OFF und TILT_JOINT aus vorhandenen smoke_report.json-Dateien.

Keine Modellrechnung. Kein Zugriff auf zurückgehaltene Testsamples.
"""
from pathlib import Path
import json

DEFAULT_OFF=Path('results/smoke_v1_f2_k3_off_c1/smoke_report.json')
DEFAULT_TILT=Path('results/smoke_v1_f2_k3_tilt_joint_c1/smoke_report.json')


def main():
    import argparse
    pa=argparse.ArgumentParser()
    pa.add_argument('--off',type=Path,default=DEFAULT_OFF)
    pa.add_argument('--joint',type=Path,default=DEFAULT_TILT)
    pa.add_argument('--full',type=Path,default=None,help='Optionale FULL_JOINT-Berichtsdatei')
    a=pa.parse_args()
    for p in [a.off,a.joint]:
        if not p.is_file():
            raise SystemExit(f'Fehlt: {p}. Pfad mit --off oder --joint angeben.')
    off=json.loads(a.off.read_text(encoding='utf-8'))
    joint=json.loads(a.joint.read_text(encoding='utf-8'))
    for fld in ['split','channel_names','xfit_ids','yfit_ids','source_sha256']:
        if off[fld]!=joint[fld]:
            raise SystemExit(f'Nicht vergleichbare Versuche: {fld} unterscheidet sich.')
    for fld in ['features','window','K','C','epochs','seed']:
        if off['config'][fld]!=joint['config'][fld]:
            raise SystemExit(f'Nicht vergleichbare Hyperparameter: {fld} unterscheidet sich.')
    if off['config']['mode']!='OFF' or joint['config']['mode']!='TILT_JOINT':
        print('WARNUNG: Vergleich ist nicht OFF vs. TILT_JOINT')
    print('Epochenvergleich mit voller gespeicherter Genauigkeit')
    print(f"{'Ep':>3s} {'OFF':>20s} {'JOINT':>20s} {'JOINT - OFF':>20s} {'max Δ Belegung':>19s}")
    for ro,rj in zip(off['history'],joint['history']):
        dif=float(rj['elbo'])-float(ro['elbo'])
        diff_state=max(abs(x-y) for x,y in zip(ro['states_effective'],rj['states_effective']))
        print(f"{ro['epoch']:3d} {ro['elbo']:20.9f} {rj['elbo']:20.9f} {dif:20.9f} {diff_state:19.11f}")
    if a.full is not None:
        if not a.full.is_file():
            raise SystemExit(f'Fehlt: {a.full}')
        full=json.loads(a.full.read_text(encoding='utf-8'))
        for fld in ['split','channel_names','xfit_ids','yfit_ids','source_sha256']:
            if off[fld]!=full[fld]:
                raise SystemExit(f'Nicht vergleichbare FULL-Daten: {fld}')
        for fld in ['features','window','K','C','epochs','seed']:
            if off['config'][fld]!=full['config'][fld]:
                raise SystemExit(f'Nicht vergleichbare FULL-Hyperparameter: {fld}')
        print('\nFULL_JOINT gegen OFF:')
        for ro,rf in zip(off['history'],full['history']):
            diff=float(rf['elbo'])-float(ro['elbo'])
            mg=rf.get('full_local_gains',{})
            print(f"Epoche {ro['epoch']}: Δ ELBO={diff:+.9f}, Summe akzeptierter lokaler Gewinne={sum(mg.values()):.8f}")
    print('Interpretation: Differenzen sind technische Trainingsdiagnostik, keine Qualitätsmetriken.')

if __name__=='__main__':main()
