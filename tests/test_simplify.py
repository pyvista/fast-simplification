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

    # Test with return_collapses=True
    # We check that the number of points after simplification is equal to the number of
    # points before simplification minus the number of collapses
    points_out, faces_out, collapses = fast_simplification.simplify(
        points, faces, 0.5, return_collapses=True
    )
    n_points_before_simplification = len(points)
    n_points_after_simplification = len(points_out)
    n_collapses = len(collapses)
    assert n_points_after_simplification == n_points_before_simplification - n_collapses


@skip_no_vtk
def test_simplify_none(mesh):
    triangles = mesh._connectivity_array.reshape(-1, 3)

    reduction = 0
    points, faces = fast_simplification.simplify(mesh.points, triangles, reduction)
    assert np.allclose(triangles, faces)
    assert np.allclose(mesh.points, points)


@skip_no_vtk
def test_simplify(mesh):
    triangles = mesh._connectivity_array.reshape(-1, 3)
    reduction = 0.5
    points, faces, collapses = fast_simplification.simplify(
        mesh.points, triangles, reduction, return_collapses=True
    )
    assert triangles.shape[0] * reduction == faces.shape[0]
    # We check that the number of points after simplification is equal to the number of
    # points before simplification minus the number of collapses
    n_points_before_simplification = mesh.points.shape[0]
    n_points_after_simplification = points.shape[0]
    n_collapses = collapses.shape[0]
    assert n_points_after_simplification == n_points_before_simplification - n_collapses

    assert points.dtype == np.float64


@skip_no_vtk
def test_simplify_lossless(mesh):
    triangles = mesh._connectivity_array.reshape(-1, 3)
    reduction = 0.5
    points, faces = fast_simplification.simplify(mesh.points, triangles, reduction, lossless=True)
    assert np.allclose(mesh.points, points)
    assert np.allclose(triangles, faces)


@skip_no_vtk
def test_simplify_agg(mesh):
    triangles = mesh._connectivity_array.reshape(-1, 3)

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
    assert mesh_out.n_cells == mesh.n_cells * reduction


@skip_no_vtk
def test_simplify_mesh_fixed_size_storage(mesh):
    reduction = 0.5
    mesh_out = fast_simplification.simplify_mesh(mesh, reduction)

    # decimated output is uniformly triangular
    assert mesh_out.n_cells == mesh.n_cells * reduction
    assert mesh_out.is_all_triangles
    assert mesh_out.n_verts == 0
    assert mesh_out.n_lines == 0

    # geometry is correct: reconstruct the triangle connectivity and confirm
    # it matches the padded-faces path
    points, faces = fast_simplification.simplify(mesh.points, mesh.regular_faces, reduction)
    assert np.allclose(mesh_out.points, points)
    assert np.array_equal(mesh_out.regular_faces, faces)

    # on VTK >= 9.6.2 the polys use fixed-size storage (no explicit offsets)
    if pv.vtk_version_info >= (9, 6, 2):
        assert mesh_out.GetPolys().IsStorageFixedSize()


def _n_boundary_points(mesh):
    """Return the number of open-boundary (perimeter) points of a mesh."""
    edges = mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    return edges.n_points


@skip_no_vtk
def test_preserve_border():
    # A triangulated flat plane has an open boundary (its perimeter). All of
    # its points are coplanar, so an aggressive decimation happily collapses
    # boundary points unless they are explicitly protected.
    mesh = pv.Plane(i_resolution=20, j_resolution=20).triangulate()
    n_border_in = _n_boundary_points(mesh)
    assert n_border_in == 80

    # Standard path: without protection the border is eroded, with protection
    # it is retained exactly.
    out_free = fast_simplification.simplify_mesh(mesh, target_reduction=0.9, preserve_border=False)
    out_kept = fast_simplification.simplify_mesh(mesh, target_reduction=0.9, preserve_border=True)
    assert _n_boundary_points(out_free) < n_border_in
    assert _n_boundary_points(out_kept) == n_border_in

    # Lossless path (exposed through the array API): same contract.
    triangles = mesh._connectivity_array.reshape(-1, 3)
    p_free, f_free = fast_simplification.simplify(
        mesh.points, triangles, lossless=True, preserve_border=False
    )
    p_kept, f_kept = fast_simplification.simplify(
        mesh.points, triangles, lossless=True, preserve_border=True
    )

    def _as_polydata(points, faces):
        cells = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces.astype(np.int64)])
        return pv.PolyData(points, cells)

    assert _n_boundary_points(_as_polydata(p_free, f_free)) < n_border_in
    assert _n_boundary_points(_as_polydata(p_kept, f_kept)) == n_border_in
