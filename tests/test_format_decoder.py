from src.format_decoder import decode, supported


def options(text, index=None, alternate=None, probability=.2):
    rows=[{c:.9} for c in text]
    if index is not None:
        rows[index][alternate]=probability
    return rows


def test_repair_digit_keeps_actual_probability():
    result=decode(options('MHI2AB3456',2,'1'))
    assert result[0]=='MH12AB3456' and result[1][2]==.2


def test_state_prefix_and_series_ambiguity():
    assert decode(options('KH12AB3456',0,'M'))[0]=='MH12AB3456'
    assert decode(options('MH128C3456',4,'B'))[0]=='MH12BC3456'


def test_supported_reading_is_not_rewritten():
    assert decode(options('DL3SCW6192',3,'5')) is None
    assert decode(options('MH12AB3456',9,'8')) is None


def test_missing_evidence_and_unsupported_formats_are_not_invented():
    assert decode(options('MHI2AB3456')) is None
    assert decode(options('MHI2AB3456',2,'1',.001)) is None
    assert decode(options('123ARMY')) is None


def test_exception_layouts():
    for value in ('HP885801','UK106679','DL3SCW6192','DL35CW6192','22BH1234AA','OR02AB1234','TG09AB1234'):
        assert supported(value)
    assert not supported('XX12AB3456')
