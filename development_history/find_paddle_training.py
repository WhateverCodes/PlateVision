import urllib.request,json
from pathlib import Path
url='https://api.github.com/repos/PaddlePaddle/PaddleOCR/git/trees/main?recursive=1'
r=json.load(urllib.request.urlopen(url,timeout=30))
rows=[x['path'] for x in r['tree'] if 'en_PP-OCRv5' in x['path']]
print('\n'.join(rows))
Path('work/paddle_source_revision.json').write_text(json.dumps({'sha':r['sha'],'paths':rows},indent=2))
