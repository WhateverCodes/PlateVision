"""Presentation only. All app logic and controls remain in Python."""
STYLE = """
<style>

.stApp {background:radial-gradient(ellipse at 15% 0%,#102421 0%,#07090d 42%);}
.block-container {max-width:1480px;padding:2.3rem 2.5rem 2rem;}
header[data-testid="stHeader"] {background:transparent;}
#MainMenu,footer,.stAppDeployButton {visibility:hidden;}
h1,h2,h3 {letter-spacing:-.04em;}
.pv-brand {font-size:43px;font-weight:850;letter-spacing:-2px;line-height:1.1;color:#f8fbff;}
.pv-brand span {color:#5cffe0;}
.pv-eyebrow {font-size:10px;font-weight:700;letter-spacing:3px;color:#8b9ba8;margin-bottom:14px;}
.pv-sub {font-size:14px;color:#93a0b2;margin:13px 0 30px;}
.pv-badge {display:inline-block;padding:8px 12px;border:1px solid #29443e;border-radius:30px;
 color:#7effdd;font-size:11px;letter-spacing:1px;margin:10px 4px;}
.pv-line {height:1px;background:linear-gradient(90deg,#5cffe070,#26303c,transparent);margin:0 0 26px;}
.pv-panel-title {font-size:12px;color:#f0f4fa;letter-spacing:2px;font-weight:700;margin-bottom:8px;}
.pv-hint {font-size:12px;color:#8592a4;line-height:1.6;}
.pv-empty {min-height:300px;border:1px solid #23323b;border-radius:14px;
 background:linear-gradient(145deg,#101b22,#0b1017);display:flex;flex-direction:column;align-items:center;
 justify-content:center;position:relative;overflow:hidden;}
.pv-empty:before {content:'';position:absolute;inset:0;background-image:linear-gradient(#5cffe008 1px,transparent 1px),
 linear-gradient(90deg,#5cffe008 1px,transparent 1px);background-size:32px 32px;}
.pv-target {width:92px;height:58px;border:2px solid #5cffe0;border-radius:8px;box-shadow:0 0 32px #5cffe014;
 display:flex;align-items:center;justify-content:center;font-size:17px;letter-spacing:4px;color:#5cffe0;z-index:1;}
.pv-empty h3 {font-size:20px;letter-spacing:-.4px;margin:23px 0 4px;z-index:1;}
.pv-empty p {color:#788a9e;font-size:12px;z-index:1;}
.pv-result-empty {padding:62px 16px;text-align:center;border:1px dashed #26323c;border-radius:12px;
 color:#8291a6;font-size:13px;line-height:1.8;min-height:250px;}
.pv-result-empty b {color:#d9e3ef;font-size:16px;}
.pv-footer {font-size:11px;letter-spacing:.7px;color:#647286;padding-top:25px;border-top:1px solid #202832;margin-top:25px;}
[data-testid="stVerticalBlockBorderWrapper"] {border-radius:15px;}
[data-testid="stMetricValue"] {font-size:28px;color:#f5fcff;}
[data-testid="stMetricLabel"] {font-size:11px;letter-spacing:1px;color:#8e9cac;}
.stButton>button {border-radius:9px;border:1px solid #30433f;font-weight:600;}
.stButton>button[kind="primary"] {background:#5cffe0;color:#07120f;border:0;}
[data-testid="stFileUploader"] {background:#0c121a;border-radius:10px;}
[data-testid="stFileUploaderDropzone"] {padding:12px;}
[data-testid="stFileUploaderDropzoneInstructions"]>div>span {font-size:12px;}
[data-testid="stDataFrame"] {border:1px solid #26323d;border-radius:10px;}
</style>
"""
