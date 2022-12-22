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


def test_simplify_trivial():
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

    points_out, faces_out = fast_simplification.simplify(points, faces, 0.5)
    assert points_out.shape[0] == 5
    assert faces_out.shape[0] == 4


@skip_no_vtk
def test_simplify_none(mesh):
    triangles = mesh.faces.reshape(-1, 4)[:, 1:]

    reduction = 0
    points, faces = fast_simplification.simplify(mesh.points, triangles, reduction)
    assert np.allclose(triangles, faces)
    assert np.allclose(mesh.points, points)


@skip_no_vtk
def test_simplify(mesh):
    triangles = mesh.faces.reshape(-1, 4)[:, 1:]
    reduction = 0.5
    points, faces = fast_simplification.simplify(mesh.points, triangles, reduction)
    assert triangles.shape[0] * reduction == faces.shape[0]


@skip_no_vtk
def test_simplify_agg(mesh):
    triangles = mesh.faces.reshape(-1, 4)[:, 1:]

    reduction = 0.5
    points, faces = fast_simplification.simplify(
        mesh.points,
        triangles,
        reduction,
        agg=0,
    )
    assert triangles.shape[0] == faces.shape[0]

    reduction = 0.5
    points, faces = fast_simplification.simplify(
        mesh.points,
        triangles,
        reduction,
        agg=1,
    )
    # somewhere between the requested reduction and the original number of triangles
    assert triangles.shape[0] * reduction < faces.shape[0] < triangles.shape[0]


@skip_no_vtk
def test_simplify_mesh(mesh):
    reduction = 0.5
    mesh_out = fast_simplification.simplify_mesh(mesh, reduction)
    assert mesh_out.n_faces == mesh.n_faces * reduction
