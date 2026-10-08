from database import recent_alert_exists, save_alert, window_stats

VIOLATION_CLASSES = {"without_mask", "mask_weared_incorrect"}

MIN_VIOLATIONS_PER_IMAGE = 2
WINDOW_MINUTES = 60
MIN_FACES_IN_WINDOW = 10
MAX_VIOLATION_RATE = 0.20
COOLDOWN_MINUTES = 10


def count_violations(detections):
    return sum(1 for d in detections if d["class"] in VIOLATION_CLASSES)


def evaluate_rules(conn, location, prediction_id, num_violations):
    candidates = []

    if num_violations >= MIN_VIOLATIONS_PER_IMAGE:
        candidates.append((
            "multiple_violations",
            f"{num_violations} people without a proper mask in one image at {location}",
        ))

    stats = window_stats(conn, location, WINDOW_MINUTES)
    if stats["faces"] >= MIN_FACES_IN_WINDOW:
        rate = stats["violations"] / stats["faces"]
        if rate > MAX_VIOLATION_RATE:
            candidates.append((
                "high_violation_rate",
                f"Violation rate {rate:.0%} in the last {WINDOW_MINUTES} min at {location} "
                f"({stats['violations']} of {stats['faces']} faces)",
            ))

    fired = []
    for rule, message in candidates:
        if recent_alert_exists(conn, location, rule, COOLDOWN_MINUTES):
            continue
        save_alert(conn, location, rule, message, prediction_id)
        fired.append({"rule": rule, "message": message})
    return fired