from pathlib import Path
from streamlit.testing.v1 import AppTest

def test_app_initial_state_and_settings():
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/"app.py"),default_timeout=30).run()
    assert not app.exception
    assert app.metric[0].value=="0"
    assert any("Setup needed" in x.value for x in app.info)
    assert next(b for b in app.button if b.label.startswith("Analyze")).disabled
    app.slider[0].set_value(0.6).run()
    assert not app.exception
