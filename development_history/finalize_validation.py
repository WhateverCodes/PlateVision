from pathlib import Path
import json,yaml
root=Path.cwd();p=root/'outputs/PlateVision1/data/ocr_finetune_audit_v1'
rows=json.loads((p/'validation_candidates.json').read_text());train=json.loads((p/'train.json').read_text())
assert not ({r['id'] for r in rows}&{r['id'] for r in train})
assert not ({r['text'] for r in rows}&{r['text'] for r in train})
(p/'validation.txt').write_text(''.join(f"validation_images/{r['id']}.png\t{r['text']}\n" for r in rows),encoding='utf8')
config=yaml.safe_load((root/'work/ocr_cpu_trial_DRAFT.yaml').read_text());config['Eval']['dataset']['label_file_list']=[str(p/'validation.txt')]
(root/'work/ocr_cpu_trial.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
card=json.loads((p/'audit.json').read_text());card['validation_single_line_candidates']=len(rows);card['validation_layout_excluded']=19;card['validation_note']='26 single-line crops visually screened for completeness. A development subset only; continue to report all 45 crops for end-to-end comparisons.'
(p/'audit.json').write_text(json.dumps(card,indent=2));print('26 validation lines prepared; no ID or label overlaps with 76 training candidates.')
