"""Side-by-side image comparison without mixing model outputs or training data."""
import hashlib
from pathlib import Path
import cv2
import pandas as pd
import streamlit as st
from config import DEFAULTS,ROOT
from .demo_media import example_path
from .preprocessing import read_image
from .validator import normalize


@st.cache_resource(show_spinner=False,max_entries=3)
def load_models(cropped,learned,stamp):
    from .runtime import configure
    from .comparison import build_pipelines
    configure(2)
    return build_pipelines(DEFAULTS,cropped,learned)


def render():
    st.title('My model vs pretrained')
    st.caption('Same image. Separate predictions. Models run one after the other with two CPU threads.')
    media=st.file_uploader('Image to compare',type=['jpg','jpeg','png','webp'],key='compare_upload')
    example=example_path(ROOT)
    use_example=st.checkbox('Use comparison example',disabled=media is not None or example is None)
    cropped=st.radio('What does the image contain?',['A vehicle or several vehicles','One cropped plate'],horizontal=True)== 'One cropped plate'
    learned=False
    if not cropped:
        st.caption('Both readers use the same external pretrained YOLOv9 plate detector. My model reads its crops with your custom CNN; the other reader uses adapted PaddleOCR. Candidates are listed by reading confidence, highest first, including rejected readings.')
    else:
        st.caption('Both readers receive the entire crop without plate detection or perspective changes. This compares reading only.')
    expected_text=st.text_input('Correct plate text, if known (optional)',help='For several plates, separate registrations with commas. This is used only to check results, never to guide predictions.')
    expected={normalize(t) for t in expected_text.split(',') if normalize(t)}
    content=media.getvalue() if media else (example.read_bytes() if use_example and example else None)
    if media and media.size>DEFAULTS.max_upload_mb*1024*1024:
        st.error('Image exceeds the 100 MB limit.');return
    key=(hashlib.sha256(content).hexdigest() if content else '',cropped,learned)
    if st.session_state.get('comparison_key')!=key:
        st.session_state.pop('comparison_result',None)
        st.session_state.comparison_key=key
    if use_example and not media:st.caption('Development example already used during testing; not an independent accuracy test.')
    if st.button('Compare both models',type='primary',disabled=content is None):
        try:
            image=read_image(content,DEFAULTS.max_image_pixels)
            paths=[DEFAULTS.adapted_character_path,DEFAULTS.whole_line_folder/'english.onnx',DEFAULTS.whole_line_folder/'text_detector.onnx',DEFAULTS.pretrained_folder/'yolo-v9-t-384-license-plates-end2end.onnx']
            stamp=tuple(p.stat().st_mtime_ns if p.exists() else 0 for p in paths)
            with st.spinner('Running both models locally…'):
                models=load_models(cropped,learned,stamp)
                from .comparison import compare
                st.session_state.comparison_result=compare(image,models)
        except (ValueError,OSError,RuntimeError,ImportError) as exc:
            st.error(str(exc));return
    results=st.session_state.get('comparison_result')
    if not results:
        if content:
            try:st.image(read_image(content),channels='BGR',use_container_width=True)
            except ValueError as exc:st.error(str(exc))
        st.info('Choose an image, then click Compare both models. Existing video recognition is available in the Recognition workspace.')
        return
    image=read_image(content);export=[]
    from .comparison import reference_counts
    for col,(name,result) in zip(st.columns(2),results.items()):
        with col:
            st.subheader(name)
            st.caption(f"Processing: {result['seconds']:.2f} seconds · loading excluded")
            canvas=image.copy();rows=[]
            for i,d in enumerate(result['detections'],1):
                x1,y1,x2,y2=d['box'];color=(80,255,190) if d['recognized'] else (100,180,255)
                cv2.rectangle(canvas,(x1,y1),(x2,y2),color,2)
                cv2.putText(canvas,str(i),(x1,max(18,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.7,color,2)
                row={'Candidate':i,'Raw reading':d['raw_text'] or 'No reading',
                     'Status':'Accepted (not verified)' if d['recognized'] else 'Rejected / uncertain',
                     'Reading confidence':round(d['confidence'],3),
                     'Plate detection score':round(d.get('score',0.),3)}
                if expected:row['Matches reference']='Yes' if d['raw_text'] in expected else 'No'
                rows.append(row);export.append({'Model':name,**row})
            st.image(canvas,channels='BGR',use_container_width=True)
            if rows:st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
            else:st.warning('No plate candidates detected.')
            if expected:
                counts=reference_counts(result['detections'],expected)
                st.write(f"Accepted matches: {counts['correct']}/{len(expected)} · missed: {counts['missed']} · incorrect/extra: {counts['wrong']}")
            with st.expander('How this result was produced'):
                st.write(result['detector']);st.write(result['reader'])
                for d in result['detections']:
                    st.image(d['crop'],channels='BGR',caption='Plate crop',use_container_width=True)
                    if name=='My model':
                        st.image(d['grayscale'],caption='Grayscale',use_container_width=True)
                        st.image(d['binary'],caption='Black and white character mask',use_container_width=True)
                        if d['characters']:st.image([c.image for c in d['characters']],width=40)
                    else:st.caption('This pretrained reader uses a colour crop directly; it does not use our character-cutting stage.')
    st.caption(f'Own CNN uses a minimum character score of {DEFAULTS.character_threshold:.2f}; whole-line OCR uses a mean token score of {DEFAULTS.ocr_threshold:.2f}. These scores are not directly comparable accuracy percentages. Agreement between models does not prove correctness. Neither route is guaranteed to read every Indian plate. The pretrained route reads text lines without cutting individual characters.')
    if export:
        st.download_button('Download comparison CSV',pd.DataFrame(export).to_csv(index=False).encode('utf-8'),'platevision_comparison.csv','text/csv')
