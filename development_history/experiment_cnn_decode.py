from pathlib import Path
p=Path('outputs/PlateVision1/src/character_classifier.py');s=p.read_text();s=s.replace('        return "".join(self.alphabet[i] for i in indices.tolist()),confidence.tolist()', '''        text="".join(self.alphabet[i] for i in indices.tolist())
        return text,confidence.tolist()

    @torch.inference_mode()
    def read_alternatives(self,chars):
        from .validator import plausible
        text,scores=self(chars)
        if not chars or plausible(text):return text,scores
        x=torch.from_numpy(np.stack([c.image for c in chars])).float()[:,None]/255
        probabilities=self.model(x).softmax(1).numpy()
        beams=[('',[],0.)]
        for distribution in probabilities:
            options=[(self.alphabet[i],float(distribution[i])) for i in distribution.argsort()[-3:][::-1]
                     if distribution[i]>=.02 and distribution[i]>=distribution.max()*.05]
            beams=sorted([(t+c,ss+[score],q+float(np.log(score))) for t,ss,q in beams for c,score in options],key=lambda v:v[2],reverse=True)[:24]
        valid=[v for v in beams if plausible(v[0])]
        return (valid[0][0],valid[0][1]) if valid else (text,scores)''')
s=s.replace('text,scores=self(chars)\n            quality=', 'text,scores=self.read_alternatives(chars)\n            quality=')
p.write_text(s)
