from pathlib import Path
p=Path('work/paddle_train_env/Lib/site-packages/paddle/dataset/common.py')
s=p.read_text();s=s.replace("DATA_HOME = os.path.join(HOME, '.cache', 'paddle', 'dataset')", "DATA_HOME = os.environ.get('PLATEVISION_PADDLE_DATA_CACHE', os.path.join(HOME, '.cache', 'paddle', 'dataset'))")
p.write_text(s)
