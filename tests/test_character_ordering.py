import numpy as np
from src.character_segmenter import Character, order_characters


def letter(x, y):
    return Character((x, y, x+12, y+30), np.zeros((32,32), dtype=np.uint8))


def test_sloping_single_line_stays_left_to_right():
    for slope in [-0.3, 0.3]:
        chars = [letter(x, int(80+slope*x)) for x in range(0,240,24)]
        ordered = order_characters(chars[::-1])
        assert [c.box for c in ordered] == [c.box for c in chars]


def test_two_sloping_lines_stay_top_then_bottom():
    top = [letter(x, int(30-0.15*x)) for x in range(30,150,24)]
    bottom = [letter(x, int(85-0.15*x)) for x in range(0,192,24)]
    ordered = order_characters((top+bottom)[::-1])
    assert [c.box for c in ordered] == [c.box for c in top+bottom]


def test_empty_input():
    assert order_characters([]) == []
