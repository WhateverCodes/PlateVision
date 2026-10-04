"""Local review UI for the imported Kaggle collection. Starts no training."""
from pathlib import Path
import json
import cv2
import pandas as pd
import streamlit as st
from config import ROOT
from src.preprocessing import read_image
from src.dataset_review import validate_review,save_review,load_reviews,source_manifest
from src.ui_style import STYLE

st.set_page_config(page_title="PLATEVISION | Dataset review",page_icon="◈",layout="wide")
st.markdown(STYLE,unsafe_allow_html=True)
cv2.setNumThreads(1)
folder=ROOT/"data/kaggle_indian_review"
st.title("PLATEVISION · Dataset review")
st.caption("Check supplied plate boxes and enter the true registration. No model training runs here.")
if not (folder/"catalog.json").exists():
    st.info("Import the dataset first. See docs/KAGGLE_IMPORT.md.")
    st.stop()
catalog=json.loads((folder/"catalog.json").read_text(encoding="utf-8"))
audit=json.loads((folder/"audit.json").read_text(encoding="utf-8"))
reviews=load_reviews(folder)
ready=[r for r in catalog if r["status"]=="pending_review"]
a,b,c=st.columns(3)
a.metric("Images with usable boxes",len(ready))
b.metric("Reviewed",sum(r.get("status")=="reviewed" for r in reviews.values()))
c.metric("Exact duplicate copies removed",audit["duplicate_copies_removed"])
st.info("Start with 10–20 images. Plate close-ups are useful for recognition; a detector also needs whole-vehicle scenes.")
with st.sidebar:
    view=st.selectbox("Show",["Not yet reviewed","Reviewed","All with usable boxes","Quarantined"])
    kind_filter=st.selectbox("Suggested image type",["All","plate_closeup","scene_or_collage"])
    st.caption("Suggested types are based on box size, not a verified vehicle classifier.")
    choices=catalog if view=="Quarantined" else ready
    choices=[r for r in choices if
             (view!="Not yet reviewed" or r["id"] not in reviews) and
             (view!="Reviewed" or r["id"] in reviews) and
             (view!="Quarantined" or r["status"]=="quarantine") and
             (kind_filter=="All" or r["suggested_kind"]==kind_filter)]
    if not choices:
        st.info("No images match this filter.")
        st.stop()
    by_id={r["id"]:r for r in choices}
    identity=st.selectbox("Image",list(by_id),format_func=lambda i:by_id[i]["original_files"][0])
    st.caption("You do not need to review every image in one session. Saved reviews stay on disk.")
    st.write("**Typing the registration:**")
    st.write("Leave out small country markings such as IND. For two-line plates, read the top line first.")
    st.write("Never guess unreadable characters. Mark the whole plate unreadable instead.")
r=reviews.get(identity,by_id[identity])
image=read_image(folder/r["image"])
marked=image.copy()
for i,box in enumerate(r["boxes"]):
    x1,y1,x2,y2=map(round,box)
    cv2.rectangle(marked,(x1,y1),(x2,y2),(80,255,100),2)
    cv2.putText(marked,str(i+1),(x1,max(18,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,0.65,(80,255,100),2)
left,right=st.columns([1.45,1],gap="large")
with left:
    st.image(marked,channels="BGR",caption="Supplied boxes. Numbers match the rows on the right.",use_container_width=True)
    st.caption("Original files: "+", ".join(r["original_files"]))
    st.caption(f"Image dimensions: {r['width']} × {r['height']} pixels.")
    if r.get("reason"): st.warning(r["reason"])
    for i,box in enumerate(r["boxes"]):
        x1,y1,x2,y2=map(round,box)
        st.image(image[y1:y2,x1:x2],channels="BGR",caption=f"Plate {i+1}",width=min(550,max(120,x2-x1)))
with right:
    initial=[]
    for i,box in enumerate(r["boxes"]):
        initial.append({"x1":box[0],"y1":box[1],"x2":box[2],"y2":box[3],
                        "text":r.get("texts",[""]*len(r["boxes"]))[i],
                        "unreadable":r.get("unreadable",[False]*len(r["boxes"]))[i]})
    with st.form("review_"+identity):
        st.write("**1. Enter each plate number**")
        st.caption("If the green boxes are correct, leave their coordinates alone. Add/remove rows only to correct missing or incorrect boxes.")
        table=st.data_editor(pd.DataFrame(initial,columns=["x1","y1","x2","y2","text","unreadable"]),
                hide_index=True,num_rows="dynamic",use_container_width=True,
                column_config={"text":st.column_config.TextColumn("Registration"),
                               "unreadable":st.column_config.CheckboxColumn("Unreadable")})
        types=["Full vehicle scene","Plate close-up","Collage / article graphic","Exclude"]
        default=r.get("image_kind","Plate close-up" if r["suggested_kind"]=="plate_closeup" else "Collage / article graphic")
        kind=st.selectbox("2. What kind of image is this?",types,index=types.index(default))
        group=st.text_input("3. Related-image group",value=r["group"])
        st.caption("Use the same group for photos of the same vehicle/capture. The suggested visual group may miss related images.")
        confirmed=st.checkbox("I checked the boxes, all visible plates, transcription, image type and related-image group.")
        submit=st.form_submit_button("Save review",type="primary")
    if submit:
        try:
            result=validate_review(r,table.fillna({"text":"","unreadable":False}).to_dict("records"),group,kind,confirmed)
            save_review(folder,result)
            st.success("Saved. Select another image to continue.")
        except (ValueError,OSError) as exc:
            st.error(str(exc))
with st.expander("Export reviewed annotations"):
    fresh=load_reviews(folder)
    payload=source_manifest(folder,fresh)
    st.write(f"{len(payload['records'])} reviewed image records available.")
    st.download_button("Download reviewed annotations",json.dumps(payload,indent=2),
                       "reviewed_source.json","application/json",disabled=not payload["records"])
    st.caption("These are plate boxes and text. Character boxes must still be reviewed before training the character CNN. No train/test split is created by this screen.")
