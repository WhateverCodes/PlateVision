from pathlib import Path
p=Path('work/prepare_archive10.py');s=p.read_text().replace("archive_member=n[:-4]+'.jpg'","archive_member=n.rsplit('/',1)[0]+'/images/'+Path(n).stem+'.jpg'");p.write_text(s)
