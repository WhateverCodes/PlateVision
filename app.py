"""PLATEVISION: local Indian license-plate learning workbench."""
from dataclasses import replace
import hashlib
import logging
from pathlib import Path
import tempfile
import pandas as pd
import streamlit as st
from config import DEFAULTS,ROOT
from src.demo_media import example_path,LocalMedia
from src.ui_style import STYLE
from src.timestamp_manager import Registry,timestamp

st.set_page_config(page_title="PLATEVISION | Recognition lab",page_icon="◈",layout="wide")
st.markdown(STYLE,unsafe_allow_html=True)
log=logging.getLogger("platevision")

workspace=st.radio("Workspace",["Recognition","Compare models"],horizontal=True)
if workspace=="Compare models":
    from src.comparison_ui import render
    render()
    st.stop()

def reset_media():
    st.session_state.registry=Registry()
    st.session_state.result_image=None
    st.session_state.detections=[]
    st.session_state.is_video=False
    st.session_state.processed=False
    st.session_state.pop("event_filter",None)
    st.session_state.pop("provenance",None)

def choose_event(event_id):
    st.session_state.event_filter=event_id

if "registry" not in st.session_state:
    reset_media()

head,tools=st.columns([3,1])
with head:
    st.markdown('<div class="pv-eyebrow">COMPUTER VISION / INDIAN PLATES</div>'
                '<div class="pv-brand">PLATE<span>VISION</span></div>'
                '<div class="pv-sub">AI-based license plate recognition. Every stage, in focus.</div>',unsafe_allow_html=True)
with tools:
    st.markdown('<span class="pv-badge">● LOCAL SESSION</span><span class="pv-badge">CPU ECO MODE</span>',unsafe_allow_html=True)
    st.caption("PaddleOCR · automatic plate detection")
# Main workspace deliberately exposes only the retained practical pipeline.
vehicle_enabled=True
show_steps=True
cfg=replace(DEFAULTS,reader_mode="whole_line",localization_mode="pretrained",
            rectify=False,character_threshold=DEFAULTS.ocr_threshold)
st.markdown('<div class="pv-line"></div>',unsafe_allow_html=True)
left,right=st.columns([1.6,1],gap="large")
with left:
    st.markdown('<div class="pv-panel-title">01 / MEDIA WORKSPACE</div>',unsafe_allow_html=True)
    st.markdown('<div class="pv-hint">Upload an image or a short video to begin a fresh recognition session.</div>',unsafe_allow_html=True)
    with st.container(border=True):
        media_slot=st.empty()
        uploaded=st.file_uploader("Upload Media",type=["jpg","jpeg","png","webp","mp4","avi","mov","mkv"],
            help="Up to 100 MB. Videos: maximum 30 seconds. Processing starts only when you press Analyze.")
        example=example_path(ROOT)
        if example is not None:
            use_example=st.checkbox("Use local development example",value=False,disabled=uploaded is not None)
            if use_example and uploaded is None:
                uploaded=LocalMedia(example)
                st.caption("Development example, not an independent accuracy test. Recognition runs on the image when you press Analyze.")
        if uploaded:
            token=hashlib.sha256(uploaded.getbuffer()).hexdigest()
            if st.session_state.get("media_token")!=token:
                reset_media()
                st.session_state.media_token=token
            suffix=Path(uploaded.name).suffix.lower()
            is_video=suffix in {".mp4",".avi",".mov",".mkv"}
            st.session_state.is_video=is_video
        else:
            if st.session_state.get("media_token"):
                reset_media()
                st.session_state.pop("media_token",None)
            is_video=False
        a,b=st.columns([1,1])
        analyze=a.button("Analyze media  →",type="primary",disabled=uploaded is None,use_container_width=True)
        clear=b.button("Clear results",use_container_width=True)
        if clear:
            reset_media()
            st.session_state.is_video=is_video
    progress_slot=st.empty()
    events_slot=st.container()

if analyze and uploaded:
    try:
        if uploaded.size>cfg.max_upload_mb*1024*1024:
            raise ValueError("This file exceeds the 100 MB upload limit.")
        from src.runtime import configure
        configure(cfg.cpu_threads)
        from src.pipeline import Pipeline
        from src.vehicle_detector import VehicleDetector
        from src.preprocessing import read_image
        from src.visualization import annotate,jpeg
        from src.video_processor import process_video
        with st.spinner("Loading local models…"):
            vehicles=VehicleDetector(cfg.vehicle_path,cfg.vehicle_threshold) if vehicle_enabled else None
            pipe=Pipeline(cfg,vehicle_detector=vehicles)
        if is_video and cfg.localization_mode=="crop":
            raise ValueError("Already-cropped plate mode supports still images only.")
        if is_video:
            progress=progress_slot.progress(0.,text="Validating and processing video…")
            with tempfile.TemporaryDirectory(prefix="platevision_") as temp:
                path=Path(temp)/("input"+suffix)
                path.write_bytes(uploaded.getvalue())
                registry,info=process_video(path,pipe,cfg,lambda v:progress.progress(v,text=f"Processing video · {v:.0%}"))
            st.session_state.registry=registry
            st.session_state.detections=[]
            st.session_state.result_image=None
            progress_slot.empty()
        else:
            image=read_image(uploaded.getvalue(),cfg.max_image_pixels)
            with st.spinner("Detecting plates and reading characters…"):
                detections=pipe(image)
            registry=Registry(cfg.fuzzy_threshold)
            registry.observe(detections)
            st.session_state.registry=registry
            st.session_state.result_image=jpeg(annotate(image,detections))
            st.session_state.detections=detections
        st.session_state.provenance={"Plate detector":pipe.detector.provenance,"Character model":pipe.classifier.provenance}
        st.session_state.processed=True
        st.session_state.pop("event_filter",None)
    except Exception as exc:
        progress_slot.empty()
        if isinstance(exc,(ValueError,OSError,RuntimeError)):
            st.error(str(exc))
        else:
            log.exception("Media analysis failed")
            st.error("Analysis could not finish. Check the terminal log and the troubleshooting guide.")

registry=st.session_state.registry
event_id=-1
with right:
    st.markdown('<div class="pv-panel-title">02 / RECOGNITION RESULTS</div>',unsafe_allow_html=True)
    st.markdown('<div class="pv-hint">Recognized plates stay in this session only.</div>',unsafe_allow_html=True)
    with st.container(border=True):
        c1,c2=st.columns(2)
        c1.metric("UNIQUE PLATES",len(registry.entries))
        c2.metric("DETECTION EVENTS",len(registry.events) if is_video else "—")
        if is_video and registry.events:
            options=[-1]+[ev.id for ev in registry.events]
            def event_label(i):
                if i==-1:
                    return "All timestamps"
                ev=registry.events[i]
                return f"{timestamp(ev.seconds)} · event {ev.id+1}"
            event_id=st.selectbox("Timestamp filter",options,format_func=event_label,key="event_filter")
        rows=registry.rows(is_video,None if event_id==-1 else event_id)
        columns=["Sr No.","Vehicle Type","License Plate"]+(["Timestamp"] if is_video else [])
        if rows:
            st.dataframe(pd.DataFrame(rows,columns=columns),height=340,hide_index=True,use_container_width=True)
            st.download_button("Export CSV",pd.DataFrame(rows,columns=columns).to_csv(index=False).encode("utf-8"),
                               "platevision_results.csv","text/csv",use_container_width=True)
        else:
            st.markdown('<div class="pv-result-empty"><b>Your results appear here</b><br>'
                        'Upload media, then select Analyze.<br>Each unique plate receives a session serial.</div>',unsafe_allow_html=True)
        st.caption("Box labels show serial numbers only. Plate text stays in this table.")
    required=[("Whole-plate reader",cfg.reader_path)] if cfg.reader_mode=="whole_plate" else [("Character CNN",cfg.character_path)]
    if cfg.reader_mode=="whole_line":required=[("Whole-line reader",cfg.whole_line_folder/"english.onnx"),("Text-line detector",cfg.whole_line_folder/"text_detector.onnx")]
    if cfg.localization_mode=="pretrained":required.append(("Pretrained plate detector",cfg.pretrained_folder/"yolo-v9-t-384-license-plates-end2end.onnx"))
    if cfg.localization_mode=="learned":required.append(("Plate detector",cfg.detector_path))
    if vehicle_enabled:required.append(("Vehicle detector",cfg.vehicle_path))
    missing=[name for name,path in required if not path.exists()]
    if missing:
        st.info("Setup needed: "+", ".join(missing)+". See the setup instructions in README.md.")
    elif st.session_state.get("provenance"):
        with st.expander("Model training provenance"):
            for name,value in st.session_state.provenance.items():
                st.write(f"**{name}:** {value}")
            st.caption("Confidence is a model score, not a calibrated probability of correctness.")

with media_slot.container():
    if is_video and uploaded:
        if registry.events and event_id!=-1:
            ev=registry.events[event_id]
            st.image(ev.image,use_container_width=True)
            st.caption(f"Event {ev.id+1} · {ev.seconds:.3f} s · frame {ev.frame_number}. Saved event frame.")
        else:
            st.video(uploaded.getvalue())
            st.caption("If your browser cannot play this codec, convert to MP4/H.264. Event frames remain available after analysis.")
    elif st.session_state.result_image:
        st.image(st.session_state.result_image,use_container_width=True)
    elif uploaded:
        try:
            from src.preprocessing import read_image
            st.image(read_image(uploaded.getvalue(),cfg.max_image_pixels),channels="BGR",use_container_width=True)
        except Exception:
            st.warning("Image preview unavailable. Check that the file is a valid image.")
    else:
        st.markdown('<div class="pv-empty"><div class="pv-target">[ ··· ]</div>'
                    '<h3>A new perspective on recognition.</h3><p>Your media. Your models. Every step visible.</p></div>',
                    unsafe_allow_html=True)

with events_slot:
    if is_video and registry.events:
        st.markdown('<div class="pv-panel-title">DETECTION EVENTS</div>',unsafe_allow_html=True)
        for start in range(0,len(registry.events),5):
            cols=st.columns(5)
            for col,ev in zip(cols,registry.events[start:start+5]):
                col.button(timestamp(ev.seconds),key=f"event_{ev.id}",help=f"Event {ev.id+1}, frame {ev.frame_number}",
                           use_container_width=True,on_click=choose_event,args=(ev.id,))
    if st.session_state.processed and not registry.entries:
        if not is_video and st.session_state.detections:
            st.warning("Plate candidates were found, but no reading passed the confidence and registration-format checks. See the uncertain readings below.")
        else:
            st.warning("No accepted plate readings. Inspect the recognition pipeline to distinguish missing plates from rejected readings.")
    if st.session_state.processed and not is_video and st.session_state.detections:
        uncertain=[d for d in st.session_state.detections if not d['recognized']]
        if uncertain:
            st.write("Uncertain readings — check against the photograph")
            st.dataframe([{"Candidate":i+1,"Unverified text":d.get('raw_text') or 'No text read',
                           "OCR score":f"{d['confidence']:.1%}","Status":"Not accepted"}
                          for i,d in enumerate(st.session_state.detections) if not d['recognized']],
                         hide_index=True,use_container_width=True)
            st.caption("These readings are not added to recognized results or exports. An OCR score is not measured accuracy.")
    if show_steps and st.session_state.detections and not is_video:
        with st.expander("Inspect the recognition pipeline"):
            for i,d in enumerate(st.session_state.detections):
                st.write(f"Plate candidate {i+1} · {d['text']} · {d.get('confidence_method','model score')} {d['confidence']:.1%}")
                cols=st.columns(3)
                cols[0].image(d["crop"],channels="BGR",caption="Plate crop",use_container_width=True)
                cols[1].image(d["grayscale"],caption="Grayscale",use_container_width=True)
                cols[2].image(d["binary"],caption="Thresholded / segmented",use_container_width=True)
                if d.get("reader_mode") in {"whole_plate","whole_line"}:
                    st.caption("This reader reads complete text lines or a complete plate. The black-and-white image is for inspection, not character cutting.")
                if d["characters"]:
                    st.image([c.image for c in d["characters"]],width=48)
                st.caption(f"Raw character output: {d['raw_text'] or 'No characters segmented'}")

st.markdown('<div class="pv-footer">PLATEVISION / LEARNING LAB &nbsp; · &nbsp; LOCAL INFERENCE &nbsp; · &nbsp; '
            'NO DATABASE &nbsp; · &nbsp; BUILT TO UNDERSTAND</div>',unsafe_allow_html=True)
