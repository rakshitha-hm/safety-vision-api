import pytest

import database
from alerts import count_violations, evaluate_rules


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "alerts_test.db")
    database.init_db()
    with database.db() as c:
        yield c


def det(cls):
    return {"class": cls, "confidence": 0.9, "box": {"x1": 0, "y1": 0, "x2": 10, "y2": 10}}


def send(conn, classes, location="ward_test"):
    detections = [det(c) for c in classes]
    n = count_violations(detections)
    pid = database.save_prediction(conn, location, "x.png", detections, n, 10.0)
    return evaluate_rules(conn, location, pid, n)


def rules(alerts):
    return {a["rule"] for a in alerts}


def test_count_violations():
    dets = [det("with_mask"), det("without_mask"), det("mask_weared_incorrect")]
    assert count_violations(dets) == 2



def test_multiple_violations_alert(conn):
    assert "multiple_violations" in rules(send(conn, ["without_mask", "without_mask"]))


def test_single_violation_does_not_fire(conn):
    assert "multiple_violations" not in rules(send(conn, ["without_mask", "with_mask"]))


def test_cooldown_blocks_repeat(conn):
    send(conn, ["without_mask", "without_mask"])
    second = send(conn, ["without_mask", "without_mask"])
    assert "multiple_violations" not in rules(second)


def test_rate_rule_needs_minimum_faces(conn):
    # 1 face, 100% violation rate, but far too few faces to trust
    assert "high_violation_rate" not in rules(send(conn, ["without_mask"]))


def test_locations_are_independent(conn):
    send(conn, ["without_mask", "without_mask"], location="ward_a")
    assert "multiple_violations" in rules(send(conn, ["without_mask", "without_mask"], location="ward_b"))