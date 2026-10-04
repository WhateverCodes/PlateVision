from pathlib import Path
p=Path('outputs/PlateVision1/src')
f=p/'character_classifier.py';s=f.read_text();s=s.replace('confidence,indices = probs.max(1)','''from .format_decoder import decode
        alternative = decode([{c:float(score) for c,score in zip(self.alphabet,row)} for row in probs.tolist()]) if getattr(self,'format_aware',True) else None
        if alternative is not None:
            return alternative
        confidence,indices = probs.max(1)''');f.write_text(s)
f=p/'paddle_candidate.py';s=f.read_text();s=s.replace("beams=[('',[],0.)];last=0", "beams=[('',[],0.)];last=0;self.last_options=[]");s=s.replace('if token and token!=last:\n                options=[]','''if token and token!=last:
                emitted=self.alphabet[token].upper()
                if len(emitted)==1 and emitted in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
                    row={}
                    for index,letter in enumerate(self.alphabet):
                        letter=letter.upper()
                        if len(letter)==1 and letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789':
                            row[letter]=max(row.get(letter,0.),float(distribution[index]))
                    self.last_options.append(row)
                elif emitted not in ' .-\\t\\r\\n':
                    self.last_options.append({})
                options=[]''');f.write_text(s)
f=p/'whole_line_reader.py';s=f.read_text();s=s.replace("beams=[('',[],0.)]", "beams=[('',[],0.)];token_options=[]")
s=s.replace("if alternatives and alternatives[0][0]=='IND':continue", "if alternatives and alternatives[0][0]=='IND':continue\n            token_options.extend(getattr(self.reader,'last_options',[]))")
s=s.replace("return selected[0],selected[1]",'''if getattr(self,'format_aware',True):
            from .format_decoder import supported,decode
            if not supported(selected[0]):
                supported_beams=[v for v in beams if supported(v[0])]
                repair=decode(token_options)
                if repair is not None:
                    reading,confidence=repair
                    supported_beams.append((reading,confidence,sum(float(np.log(max(s,1e-8))) for s in confidence)))
                if supported_beams:
                    selected=max(supported_beams,key=lambda v:v[2])
        return selected[0],selected[1]''');f.write_text(s)
