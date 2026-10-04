"""Prepare whole-line and weak character training pairs from training-only plates."""
import json,hashlib
from pathlib import Path
import cv2
from config import DEFAULTS
from src.runtime import configure
from src.whole_line_reader import WholeLineReader
from src.character_classifier import Classifier
from src.character_segmenter import segmentation_candidates
from src.format_decoder import supported

def main():
 configure(2);root=Path('data/combined_training_2026_10_03');out=root/'reader';(out/'lines').mkdir(exist_ok=True);(root/'characters/weak').mkdir(exist_ok=True)
 reader=WholeLineReader(DEFAULTS.whole_line_folder);cnn=Classifier(DEFAULTS.adapted_character_path);cnn.format_aware=False
 source=json.loads((out/'plate_train.json').read_text());lines=[];weak=[];review=[];seen=set()
 for index,r in enumerate(source):
  im=cv2.imread(r['image']);text=r['text'];crops=reader.line_crops(im)
  # Use complete single lines; handle stacked plates only when line lengths and concatenated visual reading agree.
  segments=[]
  if len(crops)==1:segments=[(crops[0]['crop'],text)]
  elif not crops and im.shape[1]/im.shape[0]>=2.2:segments=[(im,text)]
  elif len(crops)==2:
   read=[reader.reader.read(c['crop'])[0] for c in crops];clean=[''.join(c for c in t.upper() if c.isascii() and c.isalnum()) for t in read]
   if ''.join(clean)==text:segments=[(c['crop'],t) for c,t in zip(crops,clean)]
  if not segments:review.append({'id':r['id'],'text':text,'reason':'multiline_layout_not_reliably_aligned'})
  for j,(crop,label) in enumerate(segments):
   if not label:continue
   digest=hashlib.sha256(crop.tobytes()+label.encode()).hexdigest()
   if digest in seen:continue
   seen.add(digest);dest=out/'lines'/f'{digest[:20]}.png';cv2.imwrite(str(dest),crop);lines.append(dict(id=digest[:20],image=str(dest.resolve()),text=label,source_plate=r['id'],source_registration=text,label_origin=r.get('label_origin','source_label')))
  candidates=[]
  for _,chars,mask in segmentation_candidates(im):
   if len(chars)!=len(text):continue
   prediction,scores=cnn(chars);agreement=sum(a==b for a,b in zip(prediction,text))/len(text)
   if agreement>=.7:candidates.append((agreement,sum(scores)/len(scores),chars))
  if candidates:
   _,_,chars=max(candidates,key=lambda v:v[:2])
   for j,(char,label) in enumerate(zip(chars,text)):
    dest=root/'characters/weak'/f'{r["id"]}_{j}.png';cv2.imwrite(str(dest),char.image);weak.append(dict(image=str(dest.resolve()),label=label,origin='combined_weak_alignment',source_plate=r['id']))
  if index%100==0:print('Prepared plates',index+1,'/',len(source),'lines',len(lines),'weak glyphs',len(weak),flush=True)
 val=json.loads(Path('data/ocr_finetune_audit_v1/validation_candidates.json').read_text());held={r['text'] for r in val};assert all(r['source_registration'] not in held for r in lines)
 (out/'train.json').write_text(json.dumps(lines,indent=2));(out/'validation_candidates.json').write_text(json.dumps(val,indent=2));(out/'layout_review.json').write_text(json.dumps(review,indent=2));chars=json.loads((root/'characters/train.json').read_text());(root/'characters/train_with_real_plates.json').write_text(json.dumps(chars+weak,indent=2));(out/'audit.json').write_text(json.dumps({'input_training_plates':len(source),'training_lines':len(lines),'weak_glyphs':len(weak),'layout_quarantined':len(review),'validation_lines':len(val),'note':'Automatic count/agreement alignment creates weak labels, not human glyph verification. Only declared training images used.'},indent=2));print('Prepared',len(lines),'lines and',len(weak),'weak glyphs',flush=True)
if __name__=='__main__':main()
