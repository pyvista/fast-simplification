import numpy as np
import pytest

import fast_simplification

try:
    import pyvista as pv

    has_vtk = True
except ModuleNotFoundError:
    has_vtk = False

skip_no_vtk = pytest.mark.skipif(not has_vtk, reason="Requires VTK")


@pytest.fixture
def mesh():
    return pv.Sphere()


def test_collapses_trivial():
    # arrays from:
    # mesh = pv.Plane(i_resolution=2, j_resolution=2).triangulate()
    points = [
        [0.5, -0.5, 0.0],
        [0.0, -0.5, 0.0],
        [-0.5, -0.5, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [-0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.0],
        [-0.5, 0.5, 0.0],
    ]

    faces = [
        [0, 1, 3],
        [4, 3, 1],
        [1, 2, 4],
        [5, 4, 2],
        [3, 4, 6],
        [7, 6, 4],
        [4, 5, 7],
        [8, 7, 5],
    ]

    with pytest.raises(ValueError, match="You must specify"):
        fast_simplification.simplify(points, faces)

    points_out, faces_out, collapses = fast_simplification.simplify(
        points, faces, 0.5, return_collapses=True
    )
    n_points_before_simplification = len(points)
    n_points_after_simplification = len(points_out)
    n_collapses = len(collapses)
    assert n_points_after_simplification == n_points_before_simplification - n_collapses


@skip_no_vtk
def test_collapses_sphere(mesh):
    triangles = mesh.faces.reshape(-1, 4)[:, 1:]
    reduction = 0.5
    points, faces, collapses = fast_simplification.simplify(
        mesh.points, triangles, reduction, return_collapses=True
    )

    n_points_before_simplification = mesh.points.shape[0]
    n_points_after_simplification = points.shape[0]
    n_collapses = collapses.shape[0]
    assert n_points_after_simplification == n_points_before_simplification - n_collapses


test_collapses_trivial()
