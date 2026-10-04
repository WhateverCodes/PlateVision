from pathlib import Path
p=Path('outputs/PlateVision1')
f=p/'app.py';s=f.read_text(encoding='utf8');a=s.index('    with st.popover("Processing settings"');b=s.index("st.markdown('<div class=\"pv-line\"></div>'",a)
s=s[:a]+'''    st.caption("PaddleOCR · automatic plate detection")
# Main workspace deliberately exposes only the retained practical pipeline.
vehicle_enabled=True
show_steps=True
cfg=replace(DEFAULTS,reader_mode="whole_line",localization_mode="pretrained",
            rectify=False,character_threshold=DEFAULTS.ocr_threshold)
'''+s[b:]
a=s.index('with st.expander("Training guide');b=s.index("st.markdown('<div class=\"pv-footer",a);s=s[:a]+s[b:];s=s.replace('Open the training guide below.','See the setup instructions in README.md.');f.write_text(s,encoding='utf8')
f=p/'src/comparison.py';s=f.read_text(encoding='utf8');s=s.replace("('learned' if learned_own else 'classical')","('learned' if learned_own else 'pretrained')")
s=s.replace("'My model':Pipeline(own),","'My model':Pipeline(own,detector=None if learned_own and not cropped else detector),")
s=s.replace("detections=pipe(image.copy())","detections=sorted(pipe(image.copy()),key=lambda d:d.get('confidence',0.),reverse=True)")
f.write_text(s,encoding='utf8')
f=p/'src/comparison_ui.py';s=f.read_text(encoding='utf8');s=s.replace("    learned=st.checkbox('Use my trained plate detector',value=False,disabled=cropped)","    learned=False")
s=s.replace("st.caption('My model uses '+('your trained detector' if learned else 'your OpenCV plate-finding code')+' and your own character CNN. The pretrained route uses YOLOv9 and PaddleOCR whole-line recognition.')","st.caption('Both readers use the same external pretrained YOLOv9 plate detector. My model reads its crops with your custom CNN; the other reader uses adapted PaddleOCR. Candidates are listed by reading confidence, highest first, including rejected readings.')")
s=s.replace("'Model score':round(d['confidence'],3)","'Reading confidence':round(d['confidence'],3),\n                     'Plate detection score':round(d.get('score',0.),3)")
f.write_text(s,encoding='utf8')

