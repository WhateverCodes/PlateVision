import json
from pathlib import Path
p=Path('outputs/PlateVision1');names=['cnn_segmentation_expanded','cnn_segmentation_decoding','cnn_deskew','cnn_plate_deskew','cnn_edge_padding','cnn_stroke_closing']
results=[]
for name in names:
 r=json.loads((p/'outputs/evaluation'/f'{name}.json').read_text());results.append({'experiment':name,**{k:v for k,v in r.items() if k!='results'}})
report={'baseline':{'exact':13,'accepted_correct':8,'accepted_wrong':1,'character_errors':131},'images':45,'target_exact':18,'experiments':results,'promoted':False,'reason':'Best exact result 14/45 below target, with more wrongly accepted readings than baseline. Restored production segmenter and classifier.','scope':'Repeated development tuning only, not independent generalization accuracy. Ground-truth labels used for evaluation only.'}
(p/'outputs/evaluation/cnn_segmentation_trials.json').write_text(json.dumps(report,indent=2))
text='''# Character-separation experiments — 3 October 2026

Target: 18/45 exact plate readings (40%) on the existing development set. Result: not achieved.

| Experiment | Exact / 45 | Correct accepted | Wrong accepted |
|---|---:|---:|---:|
| Existing app | 13 | 8 | 1 |
| More threshold candidates | 14 | 10 | 2 |
| Thresholds + CNN alternatives | 14 | 10 | 2 |
| Character stroke straightening | 14 | 10 | 4 |
| Whole-plate tilt correction | 14 | 10 | 2 |
| Preserve edge-touching components | 12 | 9 | 2 |
| Reconnect broken vertical strokes | 14 | 10 | 4 |

Previously saved CNN checkpoints were also evaluated with revised crops; none exceeded 14/45. No pretrained OCR was substituted into the own-CNN route. No evaluation labels were used for training or hard-coded corrections. These are repeatedly tuned development results, not independent final-test results.

No experimental pipeline was deployed because the small exact-match gain came with more false acceptances. The working app's original character segmenter and classifier were restored. CNN weights and all 300 character reviews remain intact. Existing checkpoints and review data were not overwritten. Experimental scripts and outputs remain available locally for analysis.

The next structural step is to measure character recognition separately on trustworthy, independently cropped glyphs, then address merged/split components using actual segmentation annotations. More thresholds and epochs alone did not reach the target. No claim that 40% or a particular number of additional reviews will guarantee improvement.
'''
(p/'docs/CNN_SEGMENTATION_TRIALS.md').write_text(text,encoding='utf8');print(json.dumps(report,indent=2))
