exec(open('work/eval_user_photos.py',encoding='utf-8-sig').read().split('report={}')[0])
for num in (3,8,10,9):
 im=cv2.imread(str(folder/f'dataset{num}.jpg'))
 for d in pipe.detector(im):
  x,y,r,b=d['box'];crop=im[y:b,x:r]
  for name,c in [('original',crop),('pad',cv2.copyMakeBorder(crop,8,8,8,8,cv2.BORDER_REPLICATE)),('up',cv2.resize(crop,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC))]:
   text,s=pipe.classifier.read(c);print(num,name,text,sum(s)/len(s) if s else 0,flush=True)
