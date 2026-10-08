import pytest

from prepare_data import clean_box, voc_to_yolo


def test_voc_to_yolo_matches_hand_calculation():
    # the Day 2 example: image 400x300, box (100, 60, 140, 120)
    assert voc_to_yolo(100, 60, 140, 120, 400, 300) == pytest.approx((0.3, 0.3, 0.1, 0.2))


def test_clean_box_clips_to_image():
    assert clean_box(-5, 10, 50, 60, 100, 100) == (0, 10, 50, 60)


def test_clean_box_drops_tiny_box():
    assert clean_box(10, 10, 11, 50, 100, 100) is None


def test_clean_box_keeps_valid_box():
    assert clean_box(10, 10, 40, 40, 100, 100) == (10, 10, 40, 40)