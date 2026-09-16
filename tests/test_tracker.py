from src.tracker import TrackState


def test_speed_kmh_matches_known_constant_velocity():
    # 15 m/s = 54 km/h, sampled at 15 fps
    true_kmh = 54.0
    speed_mps = true_kmh / 3.6
    fps = 15
    dt = 1.0 / fps

    st = TrackState(track_id=1, cls_name="car")
    t, x = 0.0, 0.0
    for _ in range(20):
        st.metric_hist.append((t, x, 0.0))
        t += dt
        x += speed_mps * dt

    got = st.speed_kmh(window=1.0)
    assert got is not None
    assert abs(got - true_kmh) / true_kmh < 0.05  # CLAUDE.md: within 5% of ground truth


def test_speed_kmh_none_with_fewer_than_four_points():
    st = TrackState(track_id=2, cls_name="car")
    st.metric_hist.append((0.0, 0.0, 0.0))
    st.metric_hist.append((0.1, 1.0, 0.0))
    st.metric_hist.append((0.2, 2.0, 0.0))
    assert st.speed_kmh() is None


def test_speed_kmh_robust_to_single_frame_jitter():
    # a noisy sample shouldn't swing the least-squares estimate wildly,
    # unlike a naive (pos[-1] - pos[-2]) / dt two-point difference would
    true_kmh = 36.0
    speed_mps = true_kmh / 3.6
    dt = 1.0 / 15

    st = TrackState(track_id=3, cls_name="car")
    t, x = 0.0, 0.0
    for i in range(10):
        jitter = 0.3 if i == 8 else 0.0  # one noisy detection near the end
        st.metric_hist.append((t, x + jitter, 0.0))
        t += dt
        x += speed_mps * dt

    got = st.speed_kmh(window=1.0)
    assert got is not None
    assert abs(got - true_kmh) / true_kmh < 0.25
