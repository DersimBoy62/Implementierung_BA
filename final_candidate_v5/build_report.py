from __future__ import annotations
import json, pathlib, hashlib, csv, math, statistics
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Cm,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

root=pathlib.Path(__file__).resolve().parent
paths=[root/'sources'/'C_DOSE_SUMMARY_canonical.json',root/'sources'/'C_DOSE_SUMMARY_20261114.json',root/'sources'/'C_DOSE_SUMMARY_20261115.json']
data=[json.loads(p.read_text(encoding='utf-8')) for p in paths]
pairings=[d['results']['13']['pairing'] for d in data]
assert pairings==['canonical','20261114','20261115']
assert all(d['protocol']==data[0]['protocol'] for d in data)
assert all(set(d['results'])=={'13','22'} for d in data)
assert all(d['results'][k]['trial_configuration']['K']==kk for d in data for k,kk in [('13',3),('22',2)])
assert all(d['results'][k]['trial_configuration']['preprocessing']=='P0' for d in data for k in ('13','22'))
assert all(d['results'][k]['trial_configuration']['variant']=='M2' for d in data for k in ('13','22'))

selected=[('13',0.),('13',.1),('13',.2),('13',2.),('22',0.),('22',.05),('22',.1),('22',.2),('22',.35),('22',2.)]
def row(k,c,idx):
 return next(r for r in data[idx]['results'][k]['dose_curve'] if abs(r['C']-c)<1e-8)
def ff(x, n=4):return f'{x:.{n}f}'.replace('.',',')

summary={ 'study':'DM-V5-C-DOSE-THREE-PAIRS-EXPLORATORY',
  'source_original_json_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
  'source_protocol_sha256':{pair:data[i]['source_protocol_sha256'] for i,pair in enumerate(pairings)},
  'pairing_ids':pairings,
  'warning':'The same 8 labeled development rings recur in all three pairings; averages over pairings are descriptive, not new independent evaluations.',
  'configurations':{},'registered_decision':'K2 P0 M2 C0.2 physical_prior_var=0.38526108747593474 provisional; alternatives K2 C0.1 and K3 C2.'}
allrows=[]
for k in ['13','22']:
  series=[]
  for template in data[0]['results'][k]['dose_curve']:
    c=template['C'];rr=[row(k,c,i) for i in range(3)]
    rmse=np.array([r['rmse_mm'] for r in rr]); mae=np.array([r['mae_mm'] for r in rr]); delta=np.array([r['delta_mse_vs_shared_C0_mm2'] for r in rr]);wins=np.array([r['better_rings_abs_vs_shared_C0'] for r in rr]);loo=np.array([r['loo_sign_stable_vs_shared_C0'] for r in rr]);
    rec={'C':c,'rmse_mean_over_pairings_mm':rmse.mean(0).tolist(),'rmse_rms_over_pairings_mm':np.sqrt((rmse**2).mean(0)).tolist(),
         'rmse_by_pairing_mm':dict(zip(pairings,rmse.tolist())), 'mae_mean_over_pairings_mm':mae.mean(0).tolist(),
         'beneficial_split_counts_vs_shared_C0':(delta < -1e-12).sum(0).tolist(),
         'better_abs_error_count_by_pairing':dict(zip(pairings,wins.tolist())),
         'loo_stable_count_by_pairing':loo.sum(0).tolist(),
         'range_rmse_over_pairings_mm':(rmse.max(0)-rmse.min(0)).tolist()}
    series.append(rec)
    allrows.append({'model':'K'+str(data[0]['results'][k]['trial_configuration']['K']),'C':c,'outer_mean_rmse_mm':rmse[:,0].mean(),'height_mean_rmse_mm':rmse[:,1].mean(),
                    'outer_split_wins':rec['beneficial_split_counts_vs_shared_C0'][0],'height_split_wins':rec['beneficial_split_counts_vs_shared_C0'][1],
                    'outer_loo_stable_count':rec['loo_stable_count_by_pairing'][0],'height_loo_stable_count':rec['loo_stable_count_by_pairing'][1]})
  summary['configurations'][k]={'original_optuna':data[0]['results'][k]['trial_configuration'],'dose_curve':series}
(root/'C_DOSE_VERIFIED_THREE_SPLITS_SUMMARY.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
with (root/'C_DOSE_VERIFIED_COMPARISON.csv').open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(allrows[0]));writer.writeheader();writer.writerows(allrows)

# Figures: each model and target, all three pairings + arithmetic mean
for k in ['13','22']:
  d=summary['configurations'][k]['dose_curve']
  xx=np.arange(len(d));cs=[r['C'] for r in d]
  for j,target in enumerate(['outer','height']):
    fig,ax=plt.subplots(figsize=(8.7,3.45))
    for pair in pairings:
      yy=[r['rmse_by_pairing_mm'][pair][j] for r in d]
      ax.plot(xx,yy,marker='o',linewidth=1.25,alpha=.7,label=('Original' if pair=='canonical' else pair))
    ax.plot(xx,[r['rmse_mean_over_pairings_mm'][j] for r in d],marker='D',linewidth=2.4,label='Mittel der 3 Aufteilungen')
    ax.set_xticks(xx,[f'{c:g}' if abs(c-round(c))>1e-8 else str(int(round(c))) for c in cs],rotation=0,fontsize=8)
    ax.set_xlabel('Gewicht C der ungelabelten Prozess-ELBO')
    ax.set_ylabel('RMSE [mm]')
    ax.set_title(f'K={data[0]["results"][k]["trial_configuration"]["K"]} · '+('Außenunrundheit' if j==0 else 'Höhenschlag'))
    ax.grid(alpha=.22);ax.legend(fontsize=8,ncol=2,loc='best');fig.tight_layout()
    fig.savefig(root/'figures'/f'C_dose_K{k}_{target}.png',dpi=170)
    plt.close(fig)

# Pareto of all 20 configs by mean over three reused pairings
coords=np.array([[x['outer_mean_rmse_mm'],x['height_mean_rmse_mm']] for x in allrows])
pareto=[]
for i,p in enumerate(coords):
    if not any((q <= p+1e-12).all() and (q < p-1e-12).any() for j,q in enumerate(coords) if j!=i):pareto.append(i)
summary['pareto_mean_rmse_configs']=[{key:allrows[i][key] for key in ('model','C','outer_mean_rmse_mm','height_mean_rmse_mm')} for i in pareto]
(root/'C_DOSE_VERIFIED_THREE_SPLITS_SUMMARY.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
fig,ax=plt.subplots(figsize=(7.4,4.3))
for label,mark in [('K2','o'),('K3','s')]:
    idx=[i for i,r in enumerate(allrows) if r['model']==label]
    ax.scatter(coords[idx,0],coords[idx,1],s=36,marker=mark,label=label)
for i in pareto:
    r=allrows[i];ax.scatter(*coords[i],s=115,facecolors='none',edgecolors='black',linewidth=1.5)
    ax.annotate(f"{r['model']} / C={r['C']:g}",tuple(coords[i]),xytext=(6,6),textcoords='offset points',fontsize=8)
ax.set_xlabel('Mittlerer Außen-RMSE [mm]');ax.set_ylabel('Mittlerer Höhen-RMSE [mm]');ax.grid(alpha=.22);ax.legend();fig.tight_layout();fig.savefig(root/'figures'/'Pareto_C_dose.png',dpi=170);plt.close(fig)

# Document theme copied from handover: Times New Roman, narrow typography, non-obtrusive blue-gray
D=Document(); sec=D.sections[0];sec.top_margin=Cm(2.4);sec.bottom_margin=Cm(2.3);sec.left_margin=Cm(2.5);sec.right_margin=Cm(2.4)
style=D.styles['Normal'];style.font.name='Times New Roman';style.font.size=Pt(10.5);style.paragraph_format.space_after=Pt(5)
for n,size in [('Title',19),('Heading 1',13.5),('Heading 2',11.5),('Heading 3',10.5)]:
 st=D.styles[n];st.font.name='Times New Roman';st.font.size=Pt(size);st.font.bold=True;st.font.color.rgb=RGBColor(28,65,94)

D.add_heading('BAYESIAN BACHELOR – V5',0)
p=D.add_paragraph('Vollständige C-Dosis-Nachprüfung, Hauptmodellentscheidung und reproduzierbarer Pilotcode');p.style='Subtitle'
D.add_paragraph('Forschungsstatus 12. November 2026  |  Fortsetzung zum Master-Handover V3/V4  |  Entwicklungsdaten: 8 L + 4 U')
p=D.add_paragraph();r=p.add_run('WICHTIG: VORLÄUFIGER MODEL-FREEZE, KEINE UNABHÄNGIGE LEISTUNGSBESTÄTIGUNG');r.bold=True;r.font.color.rgb=RGBColor(151,69,25)

def para(s,bold=False):
 p=D.add_paragraph();r=p.add_run(s);r.bold=bold;return p

def table(headers,rows,widths=None):
 t=D.add_table(rows=1,cols=len(headers));t.style='Table Grid';t.alignment=WD_TABLE_ALIGNMENT.CENTER
 for j,x in enumerate(headers):t.rows[0].cells[j].text=str(x)
 for row in rows:
  cells=t.add_row().cells
  for j,x in enumerate(row):cells[j].text=str(x)
 for c in t.rows[0].cells:
  c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
  tcPr=c._tc.get_or_add_tcPr();sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'E6EDF4');tcPr.append(sh)
  for p in c.paragraphs:
   for run in p.runs:run.bold=True
 for row in t.rows:
  for c in row.cells:
   for p in c.paragraphs:
    for run in p.runs:run.font.size=Pt(8.5)
 D.add_paragraph().paragraph_format.space_after=Pt(1)
 return t

def picture(path,caption):
 D.add_picture(str(path),width=Cm(15.7));p=D.add_paragraph(caption);p.alignment=WD_ALIGN_PARAGRAPH.CENTER
 for r in p.runs:r.font.size=Pt(8.5);r.italic=True

D.add_heading('1. Herkunft, Vollständigkeit und Wiederholbarkeit',1)
para('Drei unveränderte, durch den Nutzer hochgeladene C_DOSE_SUMMARY.json wurden unmittelbar ausgelesen: canonical, 20261114 und 20261115. Jede Datei enthält die beiden registrierten Modellkandidaten Trial 13 und Trial 22, zehn C-Dosisstufen je Modell, jeweils Außen- und Höhenschlag-RMSE sowie MAE, gepaarte MSE-Unterschiede gegenüber Shared C=0, die Anzahl verbesserter Ringe und Leave-one-ring-out-Vorzeichenindikatoren.')
para('Die drei Dateien haben das gemeinsame Protokoll DM-V3-U-WEIGHT-DOSE-2026-11-12-v1, jedoch unterschiedliche Pairing-/Quellprotokoll-Prüfsummen. Das ist als drei getrennte Versuchsausführungen zu behandeln. Zusammengefasst sind dies 60 Modell-C-Pairing-Ergebnisse (2 Modelle × 10 Stufen × 3 Aufteilungen).')
table(['Aufteilung','Protokoll-SHA256 (Anfang)','Datei-SHA256 (Anfang)'],[(pair,summary['source_protocol_sha256'][pair][:16],summary['source_original_json_sha256'][paths[i].name][:16]) for i,pair in enumerate(pairings)])
para('Sämtliche drei Aufteilungen verwenden dieselben acht gelabelten Entwicklungsringe und vier nur mit Prozesswerten genutzten U-Ringe. Dadurch entstehen keine 24 unabhängigen Versuchsringe, keine unabhängige Testgüte und keine statistische Signifikanz aus drei Split-Wiederholungen. Die vier früher geöffneten äußeren Testringe DM07, DM08, DM11 und DM15 bleiben für alle neuen Entscheidungen ausgeschlossen.')

D.add_heading('2. Fixierte Modelle und Messgrößen',1)
table(['Merkmal','Trial 13','Trial 22'],[
('K (HMM-Zustände)','3','2'),('Loggerkanäle und Fenster','V1_Kern_6, F2','V1_Kern_6, F2'),('Prozessskalierung','P0','P0'),('Qualitätskopplung','M2','M2'),('Priorvarianz der physikalischen Koeffizienten',ff(.473291536158372,6),ff(.38526108747593474,6)),('Optuna-Anker-C',ff(.25053356993048204,6),ff(.36845006495018195,6)),('Weitere Trainingseinstellungen','10 Epochen, 8 lokale VI-Schritte','10 Epochen, 8 lokale VI-Schritte')])
para('C multipliziert den Prozess-ELBO-Beitrag ungelabelter Ringe und ist keine Prozentangabe oder Anzahl verwendeter U-Ringe. Im Modell Shared C=0 werden U-Sequenzen weiterhin für Prozessskalierung und K-Means-Initialisierung berücksichtigt; die reine U-ELBO-Last ist 0. Strict Supervised enthält keinerlei U-Daten im Training.')

D.add_heading('3. Ergebnisse der drei C-Dosis-Kurven',1)
para('Die folgende Tabelle gibt für vorab ausgewählte Vergleichspunkte den arithmetischen Mittelwert der drei aufteilungsbezogenen RMSE wieder. Diese Mittelung ist rein deskriptiv; dieselben acht Ringe werden in jeder Aufteilung erneut prognostiziert.')
rows=[]
for k,c in selected:
 r=next(z for z in summary['configurations'][k]['dose_curve'] if abs(z['C']-c)<1e-8)
 rows.append((('K3' if k=='13' else 'K2'),ff(c,4),ff(r['rmse_mean_over_pairings_mm'][0]),ff(r['rmse_mean_over_pairings_mm'][1]),
              f"{r['beneficial_split_counts_vs_shared_C0'][0]}/3",f"{r['beneficial_split_counts_vs_shared_C0'][1]}/3"))
table(['Modell','C','Außen RMSE [mm]','Höhe RMSE [mm]','Außen besser vs C0','Höhe besser vs C0'],rows)
for k in ['13','22']:
 D.add_heading(f'3.{1 if k=="13" else 2} Verlauf bei K={3 if k=="13" else 2}',2)
 picture(root/'figures'/f'C_dose_K{k}_outer.png',f'Abbildung: Außenunrundheits-RMSE, K={3 if k=="13" else 2}; alle drei Aufteilungen und deren Mittelwert.')
 picture(root/'figures'/f'C_dose_K{k}_height.png',f'Abbildung: Höhenschlag-RMSE, K={3 if k=="13" else 2}; alle drei Aufteilungen und deren Mittelwert.')

D.add_heading('4. Ringweise Stabilitätsindikatoren – Aussagegrenzen',1)
para('Die Quelldateien liefern die ANZAHL der Ringe mit geringerem Absolutfehler gegenüber Shared C=0 je Dosis, aber nicht die Ring-IDs oder sämtliche individuellen Vorhersagen. Deshalb kann aus diesen drei JSON-Dateien keine neue Tabelle mit den acht vollständigen ringweisen Fehlerkurven pro C rekonstruiert werden. Für eine solche Aufarbeitung wären die einzelnen C-DOSE-TRIAL-Berichte mit per_ring-Feldern nötig.')
for k,c in [('22',.1),('22',.2),('13',2.)]:
 r=next(z for z in summary['configurations'][k]['dose_curve'] if abs(z['C']-c)<1e-8)
 rows=[]
 for pair in pairings:
  vals=r['rmse_by_pairing_mm'][pair];b=r['better_abs_error_count_by_pairing'][pair]
  rr=row(k,c,pairings.index(pair));lo=rr['loo_sign_stable_vs_shared_C0']
  rows.append((pair,ff(vals[0]),ff(vals[1]),f'{b[0]}/8',f'{b[1]}/8',f'{"ja" if lo[0] else "nein"}/{"ja" if lo[1] else "nein"}'))
 D.add_heading(f'K={3 if k=="13" else 2}, C={ff(c,3)}',2)
 table(['Pairing','Außen RMSE','Höhe RMSE','Außen bessere Ringe','Höhe bessere Ringe','LOO-Richtung A/H'],rows)
para('Die Leave-one-ring-out-Vorzeichenprüfung untersucht nur, ob der Vorteil gegenüber Shared C=0 beim Weglassen jeweils eines Rings dasselbe Vorzeichen behält; sie schätzt keine unabhängige Generalisierung und ersetzt keine Konfidenzintervalle auf neuen Ringen.')

D.add_heading('5. Pareto-Front und Zielkonflikt',1)
picture(root/'figures'/'Pareto_C_dose.png','Abbildung: Nicht-dominierte Kandidaten anhand des mittleren RMSE über drei erneut verwendete Entwicklungssplits; schwächerer RMSE ist besser.')
para('Über die beiden Modellstrukturen und alle zehn registrierten C-Stufen ergeben sich auf Basis des mittleren RMSE folgende deskriptive Pareto-Punkte: '+ '; '.join(f"{p['model']} / C={ff(p['C'],3)} (Außen {ff(p['outer_mean_rmse_mm'])}; Höhe {ff(p['height_mean_rmse_mm'])})" for p in summary['pareto_mean_rmse_configs'])+'.')
para('K2/C0,2 ist nicht auf dieser reinen Mittelwert-Pareto-Front: K2/C0,1 schneidet im Mittel bei beiden Zielgrößen minimal besser ab. Die Wahl von C=0,2 kann daher nur als bewusst robustheitsorientierte, deskriptive Kompromissentscheidung gelten: Gegenüber C=0 sind die beiden Ziel-RMSE in allen drei Aufteilungen niedriger. Beim K2/C0,1 verschlechtert sich der Höhen-RMSE auf Pairing 20261114 leicht.')

D.add_heading('6. Verknüpfung mit der verschachtelten Optuna-CV',1)
para('Die separat bereits abgeschlossene 4×3-Nested-CV auf denselben acht Entwicklungsringen ergab für den innerlich ausgewählten SSL-Workflow Außen-/Höhen-RMSE von 0,5453/3,2100 mm; für separat optimiertes Strict Supervised 0,6541/3,1756 mm; für Shared C=0 mit SSL-fixierten Parametern 0,5503/3,2633 mm. Diese Metriken dürfen methodisch nicht als unabhängige Bestätigung der nachträglichen C-Dosiswahl interpretiert werden.')
table(['Außenfold','K SSL','Gewähltes C SSL','Gewählte phys. Priorvarianz'],[('1','3','0,05150','0,20545'),('2','3','0,09427','0,24270'),('3','3','0,11063','0,20019'),('4','2','0,05090','0,46140')])
para('Alle vier innerlich ausgewählten C-Werte liegen zwischen rund 0,05 und 0,11 und stehen im Spannungsverhältnis zum deskriptiven K3/C2-Minimum. Das spricht für eine explizite Diskussion von Trainingsumfang, Modellkopplung, Hyperparameterinteraktionen und Entwicklungs-Selection-Bias statt für ein allgemeingültiges optimales U-Gewicht.')

D.add_heading('7. Vorläufiger Hauptmodellpfad: Model Freeze V0',1)
para('EMPFEHLUNG – PROVISORISCH, NOCH NICHT UNABHÄNGIG BESTÄTIGT: K=2, Prozessvorverarbeitung P0, zeitliche Qualitätskopplung M2, sechs Loggerkanäle V1_Kern_6 im zusammenhängenden F2-Walzfenster, zwei getrennte Bayesian-NIG-Qualitätsköpfe und volle Gaussian-Wishart-HMM-Emissionskovarianz. Gewählter U-Faktor C=0,2, physikalische Priorvarianz 0,3852610875 (ursprüngliche Optuna-Trial-22-Priorwahl). Weiterhin 10 Epochen, acht lokale VI-Schritte und Seed 20261111.')
para('Rationale: K2 ist sparsamer als K3; die drei geteilten Entwicklungsaufteilungen zeigen bei C0,2 eine RMSE-Verbesserung gegenüber C0 für beide Zielgrößen, während die gemittelten Fehler nur leicht über K2/C0,1 liegen. Der vollständig optimierte Nested-CV-Workflow bevorzugte kleine positive C-Werte, was gegen eine unkritische Fixierung auf den oberen Suchrand C2 spricht. Die Priorvarianz ist dabei nicht isoliert als optimal bestätigt; ihre Festlegung stammt aus der korrelierten Optuna-Trial-22-Studie.')
para('WICHTIGE ALTERNATIVEN: (i) K2/P0/M2/C0,1 mit besserem mittleren Fehler, (ii) K3/P0/M2/C2 mit bestem mittleren Außen-RMSE, (iii) die ursprüngliche eingefrorene Forschungsbaseline K3/C1 und (iv) strikt überwachtes Training. Diese Varianten bleiben für Kapitel 4 und 5 vollständig erhalten.')

D.add_heading('8. Exakte Eingangsgrößen und Skalierungsprotokoll',1)
table(['Modellkomponente','Vorläufig festgelegte Spezifikation'],[
('Prozesskanäle','F_rad63, F_ax64, Ist_D150, Ist_s151, Ist_h152, RWG55'),
('Fenster','F2: phase804 ∈ {1,2}, LaserOK=1, MessEin=1; zeitlich zusammenhängend'),
('Prozess-P0','Pooled Mean/Standardabweichung je Kanal über alle zulässigen TRAIN-L- und TRAIN-U-Zeitpunkte'),
('Qualitätstransformation','Mittelwert und Standardabweichung ausschließlich der gelabelten TRAIN-Ringe'),
('Physikalische Merkmale Außen','Indizes 0,1,3,5,6,10 des 12-dimensionalen originalen Deskriptorvektors'),
('Physikalische Merkmale Höhe','Indizes 0,2,4,7,11 des originalen Deskriptorvektors'),
('Merkmals-Skalierung','StandardScaler pro Qualitätskopf, Fit nur auf TRAIN-L + TRAIN-U'),
('Dynamik','M2: Markovzustandsbelegung + Differenz Spät-/Frühbelegung, analytische erste/zweite Momente'),
('HMM','2 Zustände, Bayesian Full Gaussian-Wishart, keine pseudo-labels'),
('Priors / Lernen','physische NIG-Priorvarianz = 0,3852610875; sonst ursprüngliche mathematische V3-Module unverändert'),
('Vorhersagezeitpunkt','nach vollständigem F2-Prozessverlauf; keine bereits validierte Online-Prozessregelung')])
para('Vor endgültiger wissenschaftlicher Freigabe ist die Kanalbedeutung von RWG55 aus der ursprünglichen Logger-Dokumentation zu überprüfen. Der historische Deskriptorname avg_radial_speed/std_radial_speed wurde im Forschungscode für Statistikwerte des sechsten Kanals verwendet; allein der Name beweist nicht, dass RWG55 tatsächlich eine Radialgeschwindigkeit ist. Bei unklarer Semantik die Größen als Mittelwert/Standardabweichung RWG55 beschreiben, ohne ihre numerische Berechnung zu ändern.')

D.add_heading('9. Trennung von Forschung, Hauptmodell und Testgüte',1)
para('Alle bisherigen Modellvarianten und Berichte werden archiviert. Der neue, separate Modellwrapper nutzt die originalen mathematischen V3-Kerne bytegleich; eine versionierte PROPOSED_MODEL_FREEZE.json dokumentiert Parameter, erlaubte Ring-IDs und SHA-Prüfsummen. Ein kompletter TRAIN-Fit mit allen acht L und vier U-Ringen wurde technisch getestet; weder dieser Fit noch die technische Entwicklungsring-Replay-Prognose sind neue Leistungsnachweise. Der alte äußere Testringsatz DM07/08/11/15 darf nicht für erneute Modellwahl genutzt werden.')
para('Zur tatsächlichen Generalisierungsbestätigung sind neue, unabhängig erhobene DM-Ringe erforderlich. Erst danach kann eine zurückhaltende Aussage über den Einsatz als Soft-Sensor in der Prozessoptimierung folgen; ein Prognosemodell allein weist noch keine reale Verbesserung des Walzprozesses nach.')

D.add_heading('10. Verwendung in der Bachelorarbeit',1)
para('Kapitel 4: Daten- und Foldabgrenzung, Definition des U-Gewichts als ELBO-Faktor, C-Dosis-Raster und drei Aufteilungen, Tabelle aller 20 K/C-Varianten, vier vollständige Kurvengrafiken, Pareto-Auswertung, Nested-CV mit acht ringweisen Prognosen, Baselines und Sicherheits-/Mathematiktests. Ergebnisse der geteilten acht Entwicklungsringe stets als explorativ kennzeichnen.')
para('Kapitel 5: Unterschiedlicher Nutzen von U für Außenunrundheit und Höhenschlag, kleine zusätzliche Wirkung der U-ELBO gegenüber geteilter Vorverarbeitung, Abhängigkeit von C und Prior, Trade-off zwischen deskriptivem Minimum und Robustheit, Einfluss einzelner Ringe, Selektionsverzerrung durch wiederverwendete Daten, Grenzen der Unsicherheitskalibrierung und die Differenz zwischen Qualitätsprognose und nachgewiesener Prozessoptimierung.')

D.add_heading('11. Quellen- und Prüfstatus',1)
para('Vollständig maschinell geprüft: die drei aktuellen JSON-Dateien (je 532 File-Zeilen in der bereitgestellten Darstellung), die vorher bereitgestellten Nested-CV-CSV-/JSON-Dateien (als separate V4-Auswertung), die ursprünglichen Python-Quellmodule aus dem vorhandenen lokalen Forschungsarchiv, der unveränderte Daten-SHA und die registrierten Ringgruppen.')
para('Methodik-Hintergrund aus der vorherigen, bereits erstellten Forschungsnotiz: Varma & Simon (2006), Bias in error estimation when using cross-validation for model selection, BMC Bioinformatics 7:91; Cawley & Talbot (2010), On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation, JMLR 11, 2079–2107. Keine erneute originale Seitenprüfung dieser Paper in diesem Arbeitsschritt.')
para('Modellliteratur Shao et al. (2022), Mowbray et al. (2022) und Projekt-Herleitungen A–D sind im historischen Quellenbestand dokumentiert; hier keine neuen textlichen Originalquellenextraktionen und keine erfundenen Seitenzahlen. Eigene Entscheidungen (K2, M2, C0,2, Priorvarianz, 4×3-Nested-CV) sind nicht als Literaturvorgabe auszugeben.')
para('Für eine vollständige C-Dosis-Einzelringdiagnostik fehlen in den drei SUMMARY-Dateien die individuellen Ringprognosen aller C-Stufen. Detaillierte Einzelringanalysen sind bislang für die separate Nested-CV dokumentiert; C-Dosis-Per-Ring-Aussagen bleiben auf die berichteten Zählungen und LOO-Indikatoren begrenzt.')

D.add_heading('12. Reproduzierbarer neuer Code – Freigabestatus',1)
para('Der separate Ordner enthält die vorläufige Konfiguration, ein Trainings-/Vorhersageprogramm und Original-Kernmodule mit verifizierbaren Hashes. Acht synthetische Vertragstests bestanden; vollständiger Fit auf Entwicklung L8+U4 und numerisch erfolgreiche Prozessvorhersage aus einem technischen Replay wurden geprüft. Bei absichtlich fremden/unplausiblen Sensoreingaben konnte der historische Forward-Backward-Kern numerisch nicht normieren. Der Wrapper bricht deshalb kontrolliert ab, statt in solchen Fällen eine Qualitätsprognose zu behaupten. Die Grenzen und die fehlende externe Bestätigung müssen vor einer offiziellen Codefreigabe behoben oder dokumentiert sein.')

# header/footer
header=sec.header.paragraphs[0];header.text='BAYESIAN BACHELOR | Forschungsdokumentation | 12.11.2026';header.style=D.styles['Normal']
for rr in header.runs:rr.font.size=Pt(8);rr.font.color.rgb=RGBColor(85,98,109)
footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
r=footer.add_run('ENTWURF – explorative Entwicklung; kein neuer unabhängiger Test  |  ');r.font.size=Pt(8)
fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');footer._p.append(fld)

out=root/'FORSCHUNGSBERICHT_C_DOSIS_MODELFREEZE_V0_KAP4_5_2026-11-12.docx'
D.save(out)
print('created',out,'paragraphs',len(D.paragraphs),'tables',len(D.tables),'pareto',summary['pareto_mean_rmse_configs'])
