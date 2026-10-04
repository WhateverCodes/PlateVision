"""Review registration text for the whole-plate reader. No model inference."""
import json,re
from pathlib import Path
import streamlit as st
from PIL import Image
from config import ROOT
from src.dataset_review import save_review,load_reviews
from src.ui_style import STYLE
st.set_page_config(page_title="PLATEVISION | Check plate text",layout="wide")
st.markdown(STYLE,unsafe_allow_html=True)
st.title("PLATEVISION - Check plate text")
st.caption("Read the plate and check the supplied registration. No character boxes or training here.")
folder=ROOT/"data/ocr_text_review"
location_file=ROOT/"ocr_review_location.txt"
if location_file.exists():
    folder=Path(location_file.read_text(encoding="utf-8").strip())
if not (folder/"catalog.json").exists():
    st.error("The review dataset could not be found. Check the folder in ocr_review_location.txt.");st.stop()
st.caption("Reviews are saved in: " + str(folder))
catalog=json.loads((folder/"catalog.json").read_text(encoding="utf-8"));reviews=load_reviews(folder)
a,b,c=st.columns(3)
a.metric("Reviews saved",len(reviews));b.metric("Flagged images remaining",sum(bool(r["flags"]) and r["status"]=="pending_review" and r["id"] not in reviews for r in catalog));c.metric("Source placeholders set aside",sum(r["status"]!="pending_review" for r in catalog))
if st.session_state.pop("saved_notice",False):st.success("Saved. Your next image is ready.")
queue_path=folder/"accuracy_review_queue.json"
accuracy_ids=json.loads(queue_path.read_text()) if queue_path.exists() else []
view=st.sidebar.selectbox("Show",["Accuracy check - not reviewed","Flagged - not reviewed","All - not reviewed","Saved reviews","Source placeholders"])
choices=[r for r in catalog if (view=="Flagged - not reviewed" and r["flags"] and r["status"]=="pending_review" and r["id"] not in reviews) or (view=="All - not reviewed" and r["status"]=="pending_review" and r["id"] not in reviews) or (view=="Saved reviews" and r["id"] in reviews) or (view=="Source placeholders" and r["status"]!="pending_review")]
if view=="Accuracy check - not reviewed":
    pending={r["id"]:r for r in catalog if r["id"] not in reviews and r["status"]=="pending_review"}
    choices=[pending[i] for i in accuracy_ids if i in pending]
    st.info("Check 30 plates from this queue. These images were kept out of training. Your labels will help measure recognition accuracy. Mark unreadable images instead of guessing.")
if not choices:st.info("No images in this view. You can select another view on the left.");st.stop()
by_id={r["id"]:r for r in choices}
identity=st.sidebar.selectbox("Image",list(by_id),format_func=lambda i:by_id[i]["original_file"])
source=by_id[identity];r=reviews.get(identity,source)
st.sidebar.info("Start with 30 images. Each save stays on disk; you can stop whenever you need.")
st.image(Image.open(folder/r["image"]),width=min(900,max(450,r["width"])))
st.caption("Source: "+source["original_file"]+" | Supplied label: "+source["source_text"])
for flag in source["flags"]:st.warning(flag)
st.write("Enter only the registration, without spaces, IND or slogans. For two rows, read the top row first. Do not guess a hidden or blurred character.")
with st.form("text_"+identity):
    text=st.text_input("Registration",value=r.get("text",""))
    unreadable=st.checkbox("Cannot read reliably / unsupported script",value=r.get("status","").startswith("excluded"))
    submit=st.form_submit_button("Save and next",type="primary")
if submit:
    cleaned=re.sub(r"[\s-]","",text.upper())
    if not unreadable and not re.fullmatch("[A-Z0-9]+",cleaned):st.error("Enter the actual letters and digits, or mark the image unreadable.")
    else:
        result={**source,"text":"" if unreadable else cleaned,"status":"excluded_reviewed" if unreadable else "reviewed","review_scope":"Human checked registration text only; capture grouping remains provisional."}
        save_review(folder,result)
        st.session_state["saved_notice"]=True
        st.rerun()
st.caption("Flags only prioritize checks; an unflagged label is not automatically verified. Existing character reviews and models are unchanged.")
