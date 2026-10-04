from pathlib import Path
p=Path('work/prepare_combined_training.py');s=p.read_text().replace('out.mkdir(exist_ok=False)','out.mkdir(exist_ok=True)').replace("dict(**r,image=str(q),dataset='existing_indian')","{**r,'image':str(q),'dataset':'existing_indian'}");p.write_text(s)
