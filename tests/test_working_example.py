from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from config import ROOT,DEFAULTS
from src.demo_media import example_path

@pytest.mark.skipif(example_path(ROOT) is None or not DEFAULTS.adapted_character_path.exists(),reason="Local development assets not included in source-only distribution")
def test_real_image_app_end_to_end():
    app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=60).run()
    next(c for c in app.checkbox if c.label=="Use local development example").check().run()
    next(b for b in app.button if b.label.startswith("Analyze media")).click().run(timeout=60)
    assert not app.exception and not app.error
    assert app.metric[0].value=="1"
    table=app.dataframe[0].value
    assert "KL07BX7197" in table["License Plate"].tolist()
