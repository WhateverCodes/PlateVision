from pathlib import Path
import json,shutil
p=Path('outputs/PlateVision1'); trials=[]
for name,freeze,data in [('character_verified_300_trial',False,'train.json'),('character_verified_300_head_trial',True,'verified_and_synthetic.json')]:
 f=p/'models'/name/'evaluation.json';r=json.loads(f.read_text());r['training_configuration']={'base':'models/character_conservative_v1/last.pt','data':'data/character_verified_300/'+data,'freeze_features':freeze,'cpu_threads':2,'seed':42,'learning_rate':1e-5,'batch_size':24,'maximum_training_seconds':90};f.write_text(json.dumps(r,indent=2));trials.append(r)
report={'reviews_saved':300,'new_batch_saved':200,'usable_verified_characters':299,'excluded_characters':1,'deployed_model':'models/character_conservative_v1/last.pt','promotion':False,'reason':'Neither training trial improved exact plate accuracy; retain current model.','baseline':{k:v for k,v in trials[0]['baseline'].items() if k!='results'},'trials':[{ 'configuration':r['training_configuration'],'elapsed_seconds':r['elapsed_seconds'],'candidates':[{k:v for k,v in c.items() if k!='results'} for c in r['candidates']]} for r in trials],'diagnostic':{'wrong_plates':32,'wrong_character_count':20,'note':'Length mismatch suggests segmentation/count errors; it does not alone identify all causes.'},'evaluation_scope':'45 repeatedly used development crops, not independent final-test accuracy.'}
out=p/'outputs/evaluation/cnn_300_review_report.json';out.write_text(json.dumps(report,indent=2))
doc='''# CNN trial after 300 character checks — 3 October 2026

All 200 new reviews were saved. The full review collection now has 300 checks: 299 usable verified character crops and one previously excluded crop. Two labels were corrected in the new batch. Originals, reviews, and earlier datasets are preserved.

Two CPU-limited trials used the new snapshot. The first updated the full CNN with verified-character weighting, weak-label downweighting and synthetic replay. Its three stages scored 11, 12 and 12 exact plates out of 45. The second kept feature extraction fixed and used only verified characters plus synthetic replay; its stages scored 11, 11 and 12. Both were worse than the current 13/45. No candidate was deployed. No OCR weights changed.

Current own-CNN result remains 13/45 exact (28.9%), 8 correct accepted, 1 wrong accepted and 36 rejected. The same 45 development crops were used for selection; these are not independent test results.

Twenty of the 32 wrong predictions have a different character count from their reference. This suggests improving plate-to-character separation is a higher-priority next experiment than more epochs on these crops. It does not prove that classification is otherwise correct. Do not request more manual character reviews solely because these trials failed.

You have completed this review batch. No additional reviews or repeated training clicks are required. The app still uses models/character_conservative_v1/last.pt. Detailed measurements are in outputs/evaluation/cnn_300_review_report.json. Training snapshots are kept locally in data/character_verified_300; experimental checkpoints remain in separate trial folders.
'''
(p/'docs/CNN_300_REVIEW_RESULTS.md').write_text(doc,encoding='utf8')
for t in Path('outputs').glob('PlateVision*'):
 if t==p or not (t/'app.py').exists():continue
 for name in ['docs/CNN_300_REVIEW_RESULTS.md','outputs/evaluation/cnn_300_review_report.json','training/refine_verified_cnn.py']:
  (t/name).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p/name,t/name)
print('Saved trial report; deployed checkpoints unchanged.')
