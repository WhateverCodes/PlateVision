from pathlib import Path
p=Path('outputs/PlateVision1/training');s=(p/'adapt_paddle_projection.py').read_text();s=s.replace("parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()","parser.add_argument('--out',type=Path,required=True);parser.add_argument('--data',type=Path,required=True);parser.add_argument('--source',type=Path,required=True);args=parser.parse_args()")
s=s.replace("data=p/'data/ocr_finetune_audit_v1'","data=args.data").replace("source=p/'models/paddle_candidate/english.onnx'","source=args.source").replace('>120:', '>600:')
(p/'adapt_combined_paddle.py').write_text(s)
