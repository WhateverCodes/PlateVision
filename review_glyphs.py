"""Verify automatically aligned training glyphs without drawing boxes."""
import json,re
from pathlib import Path
import streamlit as st
from PIL import Image
from config import ROOT,ALPHABET
from src.ui_style import STYLE

st.set_page_config(page_title='PLATEVISION | Check characters',layout='wide')
st.markdown(STYLE,unsafe_allow_html=True)
st.title('Check individual characters')
st.caption('Training images only. No model training starts on this page.')
location=ROOT/'glyph_review_location.txt'
folder=Path(location.read_text().strip()) if location.exists() else ROOT/'data/glyph_review'
# Downloaded project copies omit training data. Reuse a single existing local
# queue rather than requiring users to copy or recreate their reviews.
if not (folder/'queue.json').exists():
    candidates=[p.parent for p in ROOT.parent.glob('PlateVision*/data/glyph_review/queue.json')]
    if len(candidates)==1:
        folder=candidates[0]
    elif len(candidates)>1:
        st.error('Several character datasets were found. Set glyph_review_location.txt to the dataset you want to review.')
        st.stop()
if not (folder/'queue.json').exists():
    st.info('The character review queue has not been prepared yet.');st.stop()
queue=json.loads((folder/'queue.json').read_text())
saved={p.stem:json.loads(p.read_text()) for p in (folder/'reviews').glob('*.json')}
st.metric('Characters checked',len(saved))
batch_path=folder/'active_batch.json'
batch=json.loads(batch_path.read_text()) if batch_path.exists() else None
if batch:
    batch_ids=batch['ids'];done=sum(i in saved for i in batch_ids)
    st.metric('This batch',f'{done} / {len(batch_ids)}')
    st.progress(done/max(1,len(batch_ids)))
    by_id={r['id']:r for r in queue}
    queue=[by_id[i] for i in batch_ids if i in by_id]
    if done==len(batch_ids):st.success('All 200 checks are saved. You can stop and tell me the batch is complete.')
st.info('Review the character crop on the right. Choose the actual visible letter or digit. If it cuts off strokes, merges characters, is blank, or is unclear, mark it unusable. Do not guess from the registration above. Save and next saves immediately; you can close and resume anytime.')
view=st.sidebar.selectbox('Show',['Not checked','Saved checks'])
choices=[r for r in queue if (r['id'] in saved)==(view=='Saved checks')]
if not choices:st.success('No more characters in this view.');st.stop()
lookup={r['id']:r for r in choices}
identity=st.sidebar.selectbox('Character',list(lookup),format_func=lambda i:lookup[i]['plate_text']+f" / position {lookup[i]['index']+1} / {i[:6]}")
r=lookup[identity];previous=saved.get(identity)
if st.session_state.pop('glyph_saved',False):st.success('Saved. The next character is ready.')
left,right=st.columns([3,1])
with left:
    st.image(Image.open(r['plate_image']),width=600)
    st.write('Reviewed registration: **'+r['plate_text']+'**')
    st.caption('Proposed position: '+str(r['index']+1)+'. Automatic alignment may be wrong; judge the actual crop shown on the right.')
with right:
    st.image(Image.open(r['image']),width=192,caption='Character crop used for training')
with st.form('glyph_'+identity):
    label=st.selectbox('Actual character',list(ALPHABET),index=ALPHABET.index(previous['label'] if previous and previous['label'] else r['suggested']),key='label_'+identity)
    unusable=st.checkbox('Unusable crop / cannot read reliably',value=bool(previous and previous['unusable']),key='exclude_'+identity)
    submit=st.form_submit_button('Save and next',type='primary')
if submit:
    assert re.fullmatch(r'[0-9a-f]{20}_\d+',identity)
    result={**r,'label':'' if unusable else label,'unusable':unusable,'status':'human_verified_glyph'}
    path=folder/'reviews'/(identity+'.json');path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(result,indent=2));temp.replace(path)
    st.session_state.glyph_saved=True;st.rerun()
st.caption('Reviews save immediately to '+str(folder))
