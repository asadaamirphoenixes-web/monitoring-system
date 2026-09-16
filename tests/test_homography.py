import numpy as np

from src.homography import GroundPlane


def test_center_point_maps_to_known_real_world_center():
    # a 100x100 px square patch representing a 10x10 m square of road
    source_polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
    target_size_m = [10.0, 10.0]
    plane = GroundPlane(source_polygon, target_size_m)

    center_m = plane.to_metric(np.array([[50.0, 50.0]]))[0]

    assert abs(center_m[0] - 5.0) < 0.05
    assert abs(center_m[1] - 5.0) < 0.05


def test_corners_map_to_known_real_world_corners():
    source_polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]
    target_size_m = [20.0, 40.0]
    plane = GroundPlane(source_polygon, target_size_m)

    corners_px = np.array(source_polygon, dtype=np.float32)
    corners_m = plane.to_metric(corners_px)
    expected = np.array([[0, 0], [20, 0], [20, 40], [0, 40]], dtype=np.float32)

    assert np.allclose(corners_m, expected, atol=0.05)


def test_area_m2_matches_target_size():
    plane = GroundPlane([[0, 0], [100, 0], [100, 100], [0, 100]], [12.0, 45.0])
    assert plane.area_m2 == 12.0 * 45.0
