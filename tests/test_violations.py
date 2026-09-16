from src.tracker import TrackState
from src.violations import Candidate, promote


def _track(plate=None, plate_conf=0.0, n_boxes=3):
    st = TrackState(track_id=1, cls_name="car")
    st.plate = plate
    st.plate_conf = plate_conf
    for i in range(n_boxes):
        st.boxes.append((float(i), None))
    return st


def _candidate(confidence=0.9):
    return Candidate("R1_overspeed", track_id=1, cls_name="car",
                      confidence=confidence, detail={}, ts=0.0)


def test_promoted_when_every_gate_passes():
    st = _track(plate="ABC-1234", plate_conf=0.90, n_boxes=3)
    cands = [_candidate(confidence=0.85)]

    promote(cands, st, min_conf=0.80)

    assert cands[0].enforceable is True


def test_not_promoted_without_a_plate():
    st = _track(plate=None, plate_conf=0.0, n_boxes=3)
    cands = [_candidate(confidence=0.95)]

    promote(cands, st, min_conf=0.80)

    assert cands[0].enforceable is False


def test_not_promoted_below_confidence_threshold():
    st = _track(plate="ABC-1234", plate_conf=0.90, n_boxes=3)
    cands = [_candidate(confidence=0.79)]

    promote(cands, st, min_conf=0.80)

    assert cands[0].enforceable is False


def test_not_promoted_below_plate_confidence_threshold():
    st = _track(plate="ABC-1234", plate_conf=0.84, n_boxes=3)
    cands = [_candidate(confidence=0.90)]

    promote(cands, st, min_conf=0.80)

    assert cands[0].enforceable is False


def test_not_promoted_with_fewer_than_three_evidence_frames():
    st = _track(plate="ABC-1234", plate_conf=0.90, n_boxes=2)
    cands = [_candidate(confidence=0.90)]

    promote(cands, st, min_conf=0.80)

    assert cands[0].enforceable is False


def test_a_single_rule_firing_never_writes_enforceable_directly():
    # CLAUDE.md constraint 3: a rule produces a candidate, never an
    # enforceable event by itself - enforceable starts False until promote()
    c = _candidate()
    assert c.enforceable is False
