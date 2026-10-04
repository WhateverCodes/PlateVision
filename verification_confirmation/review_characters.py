"""Review real character crops without loading models or training."""
import json
import cv2
import streamlit as st
from config import ROOT
from src.preprocessing import read_image
from src.character_segmenter import segment
from src.dataset_review import load_reviews
from src.character_review import fingerprint,crop_bounds,save_characters
from src.ui_style import STYLE

st.set_page_config(page_title="PLATEVISION | Character review",layout="wide")
st.markdown(STYLE,unsafe_allow_html=True)
cv2.setNumThreads(1)
folder=ROOT/"data/kaggle_indian_review"
st.title("PLATEVISION - Character review")
st.caption("Check one letter or digit at a time. This screen never starts training.")
plates=[]
for r in load_reviews(folder).values():
    if r.get("status")!="reviewed": continue
    for i,text in enumerate(r["texts"]):
        if text: plates.append((r,i))
if not plates:
    st.info("Save a readable plate in REVIEW_DATASET.cmd first.")
    st.stop()
# Simple horizontal plates first; changing this order never changes saved identities.
plates.sort(key=lambda v:(v[0]["boxes"][v[1]][3]-v[0]["boxes"][v[1]][1])/(v[0]["boxes"][v[1]][2]-v[0]["boxes"][v[1]][0]))
selected=st.sidebar.selectbox("Plate",range(len(plates)),format_func=lambda n: f'{plates[n][0]["original_files"][0]} / plate {plates[n][1]+1} / {plates[n][0]["texts"][plates[n][1]]}')
r,p=plates[selected]; text=r["texts"][p]; token=fingerprint(r,p)
x0,y0,xe,ye=crop_bounds(r,p)
crop=read_image(folder/r["image"])[y0:ye,x0:xe]
h,w=crop.shape[:2]
path=folder/"character_reviews"/(r["id"]+f"_{p}.json")
saved=json.loads(path.read_text()) if path.exists() else None
if saved and saved.get("source_fingerprint")!=token:
    st.warning("The plate review changed. Check these characters again.")
    saved=None
statekey="chars_"+token
if statekey not in st.session_state:
    entries=[None]*len(text)
    if saved:
        entries=[{"label":c["label"],"box":[c["box"][0]-x0,c["box"][1]-y0,c["box"][2]-x0,c["box"][3]-y0],"polarity":c["polarity"],"checked":True} for c in saved["characters"]]
    st.session_state[statekey]=entries
entries=st.session_state[statekey]
st.write("Registration: **"+text+"**")
st.image(crop,channels="BGR",width=min(900,max(400,w)))
st.info("Select a character, adjust its box to contain only that whole letter or digit, then confirm. Ignore IND, logos and screws. For two rows, follow the registration from the top row first.")
index=st.selectbox("Character to check",range(len(text)),format_func=lambda i:f"{i+1}: {text[i]}",key="index_"+token)
chars,mask=segment(crop)
old=entries[index]
if old:
    initial=old["box"]
elif len(chars)==len(text):
    a,b,c,d=chars[index].box
    initial=[round(a*w/mask.shape[1]),round(b*h/mask.shape[0]),round(c*w/mask.shape[1]),round(d*h/mask.shape[0])]
else:
    initial=[0,0,w,h]
    st.warning("Automatic separation needs correction on this plate. Use the sliders to isolate the selected character, or choose an easier plate for now.")
initial=[max(0,min(w-1,initial[0])),max(0,min(h-1,initial[1])),max(1,min(w,initial[2])),max(1,min(h,initial[3]))]
key=token+"_"+str(index)
left,right=st.columns([2,1])
with left:
    xx=st.slider("Left and right edges",0,w,(initial[0],initial[2]),key="x_"+key)
    yy=st.slider("Top and bottom edges",0,h,(initial[1],initial[3]),key="y_"+key)
    shown=crop.copy()
    cv2.rectangle(shown,(xx[0],yy[0]),(xx[1]-1,yy[1]-1),(70,255,120),1)
    st.image(shown,channels="BGR",width=min(900,max(400,w)))
with right:
    valid=xx[0]<xx[1] and yy[0]<yy[1]
    if valid:
        st.image(crop[yy[0]:yy[1],xx[0]:xx[1]],channels="BGR",caption="This crop must contain only: "+text[index],width=160)
    colour=st.radio("Character colour",["Dark letters on a light plate","Light letters on a dark plate"],index=1 if old and old["polarity"]=="light" else 0,key="colour_"+key)
    if st.button("Confirm this character",type="primary",disabled=not valid):
        entries[index]={"label":text[index],"box":[xx[0],yy[0],xx[1],yy[1]],"polarity":"dark" if colour.startswith("Dark") else "light","checked":True}
        st.success("Confirmed. Choose the next character above.")
# Slider edits must invalidate an earlier confirmation immediately.
current=entries[index]
if current and (current["box"]!=[xx[0],yy[0],xx[1],yy[1]] or current["polarity"]!=("dark" if colour.startswith("Dark") else "light")):
    entries[index]=None
count=sum(e is not None for e in entries)
st.write(f"{count} of {len(text)} characters confirmed for this plate.")
st.caption("Confirmations stay while this page session is open. Save the completed plate below before closing the window.")
if st.button("Save completed plate",disabled=count!=len(text)):
    try:
        save_characters(folder,r,p,entries)
        st.success("Saved to disk. You can close this window or choose another plate.")
    except (ValueError,OSError) as exc: st.error(str(exc))
if saved: st.caption("A completed review is saved for this plate. Save again to keep any new changes.")
st.sidebar.write("Start with 2-3 clear plates. Stop if a character is hidden or cannot be isolated; select another plate.")
