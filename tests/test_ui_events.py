from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from src.timestamp_manager import Registry

def test_event_filter_and_button_share_actual_event():
    fake=BytesIO(b"synthetic-video-test-placeholder")
    fake.name="clip.mp4"
    fake.size=len(fake.getvalue())
    app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/"app.py"),default_timeout=30)
    with patch("streamlit.file_uploader",return_value=fake):
        app.run()
        registry=Registry()
        registry.observe([{"text":"MH12AB1234","recognized":True,"confidence":0.9}],5,100,20)
        registry.observe([{"text":"KA01XY5678","recognized":True,"confidence":0.9}],12,240,20)
        # Generate an actual decodable event image, not a model prediction.
        from PIL import Image
        image=BytesIO()
        Image.new("RGB",(10,10),"black").save(image,format="JPEG")
        for event in registry.events:
            event.image=image.getvalue()
        app.session_state.registry=registry
        app.run()
        assert not app.exception
        app.selectbox[0].select(1).run()
        assert not app.exception
        assert app.dataframe[0].value["License Plate"].tolist()==["KA01XY5678"]
        next(b for b in app.button if b.key=="event_0").click().run()
        assert not app.exception
        assert app.dataframe[0].value["License Plate"].tolist()==["MH12AB1234"]
