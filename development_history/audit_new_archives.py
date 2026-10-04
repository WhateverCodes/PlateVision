import io,json,hashlib,re
from pathlib import Path
from zipfile import ZipFile
from collections import Counter,defaultdict
import xml.etree.ElementTree as ET
from PIL import Image,ImageDraw
out=Path("outputs/PlateVision/outputs/dataset_review/new_archives");out.mkdir(parents=True,exist_ok=True)
report={}
with ZipFile("C:/Users/Grace/Downloads/archive (3).zip") as z:
 images={Path(n).name:n for n in z.namelist() if n.lower().endswith(".jpg")}
 rows=[];issues=[];attrs=Counter();tiles=[]
 for n in z.namelist():
  if not n.endswith(".xml"):continue
  node=ET.fromstring(z.read(n));name=node.findtext("filename")
  if name not in images:issues.append([name,"missing image"]);continue
  im=Image.open(io.BytesIO(z.read(images[name])));im.load();w,h=im.size
  boxes=[]
  if (int(node.findtext("size/width")),int(node.findtext("size/height")))!=(w,h):issues.append([name,"size mismatch"])
  if im.getexif().get(274,1)!=1:issues.append([name,"EXIF rotation",im.getexif().get(274)])
  for obj in node.findall("object"):
   box=[float(obj.findtext("bndbox/"+k)) for k in ("xmin","ymin","xmax","ymax")];boxes.append(box)
   if not (0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h):issues.append([name,"invalid box"])
   attrs.update(a.findtext("name") for a in obj.findall("attributes/attribute"))
  rows.append({"image":images[name],"boxes":boxes})
  if len(tiles)<8:
   im=im.convert("RGB");draw=ImageDraw.Draw(im)
   for box in boxes:draw.rectangle(box,outline="lime",width=8)
   im.thumbnail((300,270));tile=Image.new("RGB",(320,300),"white");tile.paste(im,(0,25));ImageDraw.Draw(tile).text((4,4),name,fill="black");tiles.append(tile)
 sheet=Image.new("RGB",(1280,600),"#dddddd")
 for i,t in enumerate(tiles):sheet.paste(t,((i%4)*320,(i//4)*300))
 sheet.save(out/"detector_samples.jpg")
 report["archive3"]={"images":len(images),"annotations":len(rows),"boxes":sum(len(r["boxes"]) for r in rows),"attributes":dict(attrs),"issues":issues}
with ZipFile("C:/Users/Grace/Downloads/archive (4).zip") as z:
 rows=[];issues=[];hashes=defaultdict(list);texts=defaultdict(set);tiles=[];counts=Counter()
 for split in ("train","valid","test"):
  for line in z.read(split+"_labels.txt").decode("utf-8-sig").splitlines():
   if not line.strip():continue
   parts=line.split("\t")
   if len(parts)!=2:issues.append([split,"malformed label",line]);continue
   name,text=parts;name="images/"+split+"/"+name
   if name not in z.namelist():issues.append([name,"missing image"]);continue
   im=Image.open(io.BytesIO(z.read(name))).convert("RGB");im.load()
   digest=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
   row={"image":name,"text":text,"split":split,"sha256":digest,"size":im.size};rows.append(row);hashes[digest].append(row);texts[text].add(split);counts[split]+=1
   if not re.fullmatch("[A-Z0-9]+",text):issues.append([name,"non alphanumeric text",text])
   if len(tiles)<24:
    im.thumbnail((300,100));tile=Image.new("RGB",(320,135),"white");tile.paste(im,(5,25));ImageDraw.Draw(tile).text((5,5),text+" / "+Path(name).name,fill="black");tiles.append(tile)
 sheet=Image.new("RGB",(1280,810),"#dddddd")
 for i,t in enumerate(tiles):sheet.paste(t,((i%4)*320,(i//4)*135))
 sheet.save(out/"ocr_samples.jpg")
 report["archive4"]={"images":len(rows),"splits":dict(counts),"unique_pixel_images":len(hashes),"duplicate_copies":sum(len(v)-1 for v in hashes.values()),"exact_image_groups_across_splits":sum(len({r["split"] for r in v})>1 for v in hashes.values()),"conflicting_label_image_groups":sum(len({r["text"] for r in v})>1 for v in hashes.values()),"unique_texts":len(texts),"texts_across_splits":sum(len(v)>1 for v in texts.values()),"issues":issues,"character_counts":dict(sorted(Counter(''.join(r['text'] for r in rows)).items()))}
 (out/"ocr_inventory.json").write_text(json.dumps(rows,indent=2))
(out/"audit.json").write_text(json.dumps(report,indent=2));print(json.dumps(report))
