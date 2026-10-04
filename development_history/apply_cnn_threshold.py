from pathlib import Path
p=Path('outputs/PlateVision1')
f=p/'config.py';s=f.read_text();assert 'character_threshold: float = 0.75' in s;f.write_text(s.replace('character_threshold: float = 0.75','character_threshold: float = 0.70'))
f=p/'src/comparison_ui.py';s=f.read_text();s=s.replace("st.caption('Own CNN uses a minimum character score of 0.75; whole-line OCR uses a mean token score of 0.90.","st.caption(f'Own CNN uses a minimum character score of {DEFAULTS.character_threshold:.2f}; whole-line OCR uses a mean token score of {DEFAULTS.ocr_threshold:.2f}.");f.write_text(s)
f=p/'training/evaluate_comparison.py';s=f.read_text();s=s.replace('own CNN minimum-glyph gate 0.75','own CNN minimum-glyph gate {DEFAULTS.character_threshold}');f.write_text(s)
