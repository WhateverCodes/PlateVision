import urllib.request,json,zipfile,io
from pathlib import Path
rev=json.loads(Path('work/paddle_source_revision.json').read_text())['sha']
url=f'https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/{rev}/configs/rec/PP-OCRv5/multi_language/en_PP-OCRv5_mobile_rec.yaml'
data=urllib.request.urlopen(url,timeout=30).read();Path('work/en_ocr_train.yaml').write_bytes(data);print(data.decode())
doc=urllib.request.urlopen(f'https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/{rev}/docs/version3.x/module_usage/text_recognition.en.md',timeout=30).read().decode()
for line in doc.splitlines():
 if 'en_PP-OCRv5_mobile_rec' in line:print(line)
