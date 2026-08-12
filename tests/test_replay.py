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

    (
        replay_points,
        replay_faces,
        indice_mapping,
    ) = fast_simplification.replay_simplification(points, faces, collapses)
    assert np.allclose(points_out, replay_points)
    assert np.allclose(faces_out, replay_faces)


@skip_no_vtk
def test_collapses_sphere(mesh):
    points = mesh.points
    faces = mesh.faces.reshape(-1, 4)[:, 1:]
    reduction = 0.5

    points_out, faces_out, collapses = fast_simplification.simplify(
        points, faces, reduction, return_collapses=True
    )

    (
        replay_points,
        replay_faces,
        indice_mapping,
    ) = fast_simplification.replay_simplification(points, faces, collapses)
    assert np.allclose(points_out, replay_points)
    assert np.allclose(faces_out, replay_faces)


try:
    from pyvista import examples

    @pytest.fixture
    def louis():
        return examples.download_louis_louvre()

    @pytest.fixture
    def human():
        return examples.download_human()

    has_examples = True
except:
    has_examples = False
skip_no_examples = pytest.mark.skipif(not has_examples, reason="Requires pyvista.examples")


@skip_no_examples
@skip_no_vtk
def test_collapses_louis(louis):
    points = louis.points
    faces = louis.faces.reshape(-1, 4)[:, 1:]
    reduction = 0.9

    points_out, faces_out, collapses = fast_simplification.simplify(
        points, faces, reduction, return_collapses=True
    )

    (
        replay_points,
        replay_faces,
        indice_mapping,
    ) = fast_simplification.replay_simplification(points, faces, collapses)
    assert np.allclose(points_out, replay_points)
    assert np.allclose(faces_out, replay_faces)


@skip_no_examples
@skip_no_vtk
def test_human(human):
    points = human.points
    faces = human.faces.reshape(-1, 4)[:, 1:]
    reduction = 0.9

    points_out, faces_out, collapses = fast_simplification.simplify(
        points, faces, reduction, return_collapses=True
    )

    (
        replay_points,
        replay_faces,
        indice_mapping,
    ) = fast_simplification.replay_simplification(points, faces, collapses)
    assert np.allclose(points_out, replay_points)
    assert np.allclose(faces_out, replay_faces)


def _triangulated_grid(n):
    """An ``n x n`` triangulated planar grid (no VTK needed)."""
    xs, ys = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    points = np.c_[xs.ravel(), ys.ravel(), np.zeros(n * n)].astype(np.float64)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b = a + 1
            c = a + n
            d = c + 1
            faces += [[a, b, c], [b, d, c]]
    return points, np.array(faces, dtype=np.int32)


def test_replay_unreferenced_vertices_end():
    # Issue #60: vertices not referenced by any triangle are dropped by the
    # decimation core but were retained by the index bookkeeping, producing an
    # ``indice_mapping`` that pointed past the end of the decimated points and
    # raised ``IndexError`` in ``_map_isolated_points``. Here the unreferenced
    # vertices are appended at the END of the array.
    points, faces = _triangulated_grid(12)
    n_ref = len(points)
    isolated = np.array([[9.0, 9.0, 9.0], [8.0, 8.0, 8.0], [7.0, 7.0, 7.0]])
    points_iso = np.vstack([points, isolated])

    _, _, coll = fast_simplification.simplify(
        points_iso, faces, target_reduction=0.6, return_collapses=True
    )
    # must not raise
    dp, dt, vmap = fast_simplification.replay_simplification(
        points_iso.astype(np.float32), faces, coll
    )
    # every decimated triangle index is in range
    assert dt.max() < dp.shape[0]
    assert vmap.shape[0] == points_iso.shape[0]
    # the unreferenced vertices are not part of the decimated mesh
    assert np.all(vmap[n_ref:] == -1)
    # referenced vertices map inside the decimated point set
    ref_map = vmap[:n_ref]
    assert ref_map.max() < dp.shape[0]
    assert ref_map.min() >= 0


def test_replay_unreferenced_vertices_middle_matches_clean():
    # An unreferenced vertex inserted in the MIDDLE of the index range used to
    # silently shift the decimated index of every following vertex. The
    # decimated mesh and the referenced-vertex mapping must be identical to the
    # same mesh without the isolated vertices.
    points, faces = _triangulated_grid(12)

    # clean reference result
    _, _, coll = fast_simplification.simplify(
        points, faces, target_reduction=0.6, return_collapses=True
    )
    dp0, dt0, vm0 = fast_simplification.replay_simplification(
        points.astype(np.float32), faces, coll
    )

    # insert isolated vertices in the middle; shift the affected face indices
    at = 50
    isolated = np.array([[5.0, 5.0, 5.0], [6.0, 6.0, 6.0], [7.0, 7.0, 7.0]])
    points_iso = np.insert(points, at, isolated, axis=0)
    faces_iso = faces.copy()
    faces_iso[faces_iso >= at] += len(isolated)

    _, _, coll_iso = fast_simplification.simplify(
        points_iso, faces_iso, target_reduction=0.6, return_collapses=True
    )
    dp1, dt1, vm1 = fast_simplification.replay_simplification(
        points_iso.astype(np.float32), faces_iso, coll_iso
    )

    # decimated mesh is unchanged by the presence of the isolated vertices
    assert dp0.shape == dp1.shape and np.allclose(dp0, dp1)
    assert np.array_equal(dt0, dt1)

    # isolated vertices map to -1
    assert np.all(vm1[at : at + len(isolated)] == -1)

    # referenced vertices keep the same decimated index (shift-adjusted)
    def shifted(i):
        return i + (len(isolated) if i >= at else 0)

    for v in np.unique(faces):
        assert vm0[v] == vm1[shifted(v)]
