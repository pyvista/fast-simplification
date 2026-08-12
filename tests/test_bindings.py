"""Binding-surface tests for the compiled ``_simplify`` / ``_replay`` modules.

These exercise the nanobind layer directly: dtype dispatch in the loaders,
the dtype/shape/ownership contract of the returned arrays, the
non-triangle rejection path, argument validation, and replay round-trips.
Everything here is deterministic and offline (procedural PyVista meshes
only) so it never depends on network mesh downloads.
"""

import numpy as np
import pytest

import fast_simplification
from fast_simplification import _simplify

try:
    import pyvista as pv

    has_vtk = True
except ModuleNotFoundError:
    has_vtk = False

skip_no_vtk = pytest.mark.skipif(not has_vtk, reason="Requires VTK")


# A small, exactly-known planar mesh: a 2x2 triangulated plane (8 triangles,
# 9 points).  Using explicit arrays keeps these tests independent of any
# VTK/PyVista geometry.
PLANE_POINTS = np.array(
    [
        [0.5, -0.5, 0.0],
        [0.0, -0.5, 0.0],
        [-0.5, -0.5, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [-0.5, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.0],
        [-0.5, 0.5, 0.0],
    ],
    dtype=np.float64,
)

PLANE_FACES = np.array(
    [
        [0, 1, 3],
        [4, 3, 1],
        [1, 2, 4],
        [5, 4, 2],
        [3, 4, 6],
        [7, 6, 4],
        [4, 5, 7],
        [8, 7, 5],
    ],
    dtype=np.int32,
)


@pytest.fixture
def sphere():
    return pv.Sphere()


# ---------------------------------------------------------------------------
# Loader dtype dispatch
# ---------------------------------------------------------------------------


def test_simplify_int32_faces():
    points, faces = fast_simplification.simplify(
        PLANE_POINTS, PLANE_FACES.astype(np.int32), target_reduction=0.5
    )
    assert faces.shape == (4, 3)
    assert points.shape[1] == 3


def test_simplify_int64_faces_matches_int32():
    # The wrapper dispatches to ``load_int64`` for int64 faces and to
    # ``load_int32`` otherwise. Both loaders must feed the identical mesh to
    # the core, so the results must agree bit-for-bit.
    p32, f32 = fast_simplification.simplify(
        PLANE_POINTS, PLANE_FACES.astype(np.int32), target_reduction=0.5
    )
    p64, f64 = fast_simplification.simplify(
        PLANE_POINTS, PLANE_FACES.astype(np.int64), target_reduction=0.5
    )
    assert np.array_equal(f32, f64)
    assert np.allclose(p32, p64)
    # ``simplify`` always returns int32 faces regardless of the input face
    # dtype (it reads back ``return_faces_int32_no_padding``), matching the
    # return type annotation.
    assert f32.dtype == np.int32
    assert f64.dtype == np.int32


def test_simplify_list_faces():
    # A python-list of faces takes the ``astype(np.int32)`` fallback branch.
    points, faces = fast_simplification.simplify(
        PLANE_POINTS.tolist(), PLANE_FACES.tolist(), target_reduction=0.5
    )
    assert faces.shape == (4, 3)


def test_simplify_float32_points():
    # Points are cast to float64 internally; a float32 input must work and
    # give the same result as an explicit float64 input.
    p32, f32 = fast_simplification.simplify(
        PLANE_POINTS.astype(np.float32), PLANE_FACES, target_reduction=0.5
    )
    p64, f64 = fast_simplification.simplify(PLANE_POINTS, PLANE_FACES, target_reduction=0.5)
    assert np.array_equal(f32, f64)
    assert np.allclose(p32, p64)


def test_noncontiguous_input():
    # nanobind's ``c_contig`` constraint is stricter than a plain pointer
    # hand-off; the ``@ascontiguous`` decorator plus the wrapper's own
    # ascontiguousarray calls must absorb non-contiguous input. Build
    # non-contiguous views by slicing a padded buffer.
    padded_pts = np.zeros((PLANE_POINTS.shape[0], 4), dtype=np.float64)
    padded_pts[:, :3] = PLANE_POINTS
    pts_view = padded_pts[:, :3]
    assert not pts_view.flags["C_CONTIGUOUS"]

    padded_faces = np.zeros((PLANE_FACES.shape[0], 4), dtype=np.int32)
    padded_faces[:, :3] = PLANE_FACES
    faces_view = padded_faces[:, :3]
    assert not faces_view.flags["C_CONTIGUOUS"]

    points, faces = fast_simplification.simplify(pts_view, faces_view, target_reduction=0.5)
    assert faces.shape == (4, 3)


# ---------------------------------------------------------------------------
# Returned-array contract: dtype, shape, ownership
# ---------------------------------------------------------------------------


def test_return_dtypes_and_shapes():
    points, faces, collapses = fast_simplification.simplify(
        PLANE_POINTS, PLANE_FACES, target_reduction=0.5, return_collapses=True
    )
    assert points.dtype == np.float64
    assert points.shape == (5, 3)
    assert faces.dtype == np.int32
    assert faces.shape == (4, 3)
    assert collapses.dtype == np.int32
    assert collapses.ndim == 2 and collapses.shape[1] == 2
    # 9 input points, 5 output points -> 4 collapses
    assert collapses.shape[0] == PLANE_POINTS.shape[0] - points.shape[0]


def test_returned_arrays_are_owned_and_writable():
    # The nanobind buffers are malloc-backed and kept alive by a capsule
    # ``base`` (so numpy reports OWNDATA=False, but the memory is owned and
    # freed by the capsule). They must be writable and survive mutation.
    points, faces = fast_simplification.simplify(PLANE_POINTS, PLANE_FACES, target_reduction=0.5)
    assert points.base is not None  # capsule keeps the malloc buffer alive
    assert faces.base is not None
    assert points.flags["WRITEABLE"]
    assert faces.flags["WRITEABLE"]
    points += 1.0  # must not segfault; buffer is genuinely owned
    assert np.isfinite(points).all()


def test_no_aliasing_between_calls():
    # Each call allocates fresh output buffers; results from an earlier call
    # must not change when a later call runs (or when an output is mutated).
    p1, f1 = fast_simplification.simplify(PLANE_POINTS, PLANE_FACES, target_reduction=0.5)
    p1_copy = p1.copy()
    p2, f2 = fast_simplification.simplify(PLANE_POINTS, PLANE_FACES, target_reduction=0.25)
    assert np.array_equal(p1, p1_copy)
    p2 += 100.0
    assert np.array_equal(p1, p1_copy)


def test_return_faces_no_padding_is_flat_connectivity():
    # ``return_faces_int32_no_padding`` is the path simplify.py uses on
    # VTK >= 9.6.2: a flat int32 connectivity buffer, length n_tri*3, no
    # leading count column. It must match the reshaped (m, 3) faces.
    _, faces = fast_simplification.simplify(PLANE_POINTS, PLANE_FACES, target_reduction=0.5)
    unpadded = _simplify.return_faces_int32_no_padding()
    assert unpadded.dtype == np.int32
    assert unpadded.size == faces.size
    assert np.array_equal(unpadded.reshape(-1, 3), faces)


def test_return_faces_int32_is_vtk_padded():
    # ``return_faces_int32`` (used by simplify.py on a 32-bit-vtkIdType build,
    # VTK < 9.6.2) returns VTK-padded connectivity [3, i, j, k] per triangle in
    # a flat int32 buffer of length n_tri*4. A stride bug in the C++ core used
    # to write 4 values while advancing by 3, clobbering the next triangle's
    # leading count and leaving a garbage tail; guard against a regression.
    fast_simplification.simplify(PLANE_POINTS, PLANE_FACES, target_reduction=0.5)
    padded = _simplify.return_faces_int32()
    assert padded.dtype == np.int32
    assert padded.size % 4 == 0
    quads = padded.reshape(-1, 4)
    assert np.all(quads[:, 0] == 3)  # every leading count is 3, no garbage tail
    unpadded = _simplify.return_faces_int32_no_padding()
    assert np.array_equal(quads[:, 1:].ravel(), unpadded)
    # and it agrees with the (correctly strided) int64 accessor
    assert np.array_equal(padded.astype(np.int64), _simplify.return_faces_int64())


def test_return_faces_int64_is_vtk_padded():
    # ``return_faces_int64`` is the path simplify.py uses on the standard
    # 64-bit-vtkIdType build: VTK-padded connectivity [3, i, j, k] per
    # triangle. Its stripped payload must equal the unpadded connectivity.
    fast_simplification.simplify(PLANE_POINTS, PLANE_FACES, target_reduction=0.5)
    padded = _simplify.return_faces_int64()
    assert padded.dtype == np.int64
    assert padded.size % 4 == 0
    quads = padded.reshape(-1, 4)
    assert np.all(quads[:, 0] == 3)
    unpadded = _simplify.return_faces_int32_no_padding()
    assert np.array_equal(quads[:, 1:].ravel().astype(np.int32), unpadded)


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_check_args_both_specified():
    with pytest.raises(ValueError, match="but not both"):
        fast_simplification.simplify(
            PLANE_POINTS, PLANE_FACES, target_reduction=0.5, target_count=2
        )


def test_check_args_neither_specified():
    with pytest.raises(ValueError, match="You must specify"):
        fast_simplification.simplify(PLANE_POINTS, PLANE_FACES)


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_check_args_reduction_out_of_range(bad):
    with pytest.raises(ValueError, match="between 0 and 1"):
        fast_simplification.simplify(PLANE_POINTS, PLANE_FACES, target_reduction=bad)


def test_check_args_target_count_too_large():
    with pytest.raises(ValueError, match="less than the number of faces"):
        fast_simplification.simplify(
            PLANE_POINTS, PLANE_FACES, target_count=PLANE_FACES.shape[0] + 1
        )


def test_bad_points_shape():
    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        fast_simplification.simplify(PLANE_POINTS[:, :2], PLANE_FACES, target_reduction=0.5)


def test_bad_faces_shape():
    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        fast_simplification.simplify(PLANE_POINTS, PLANE_FACES[:, :2], target_reduction=0.5)


# ---------------------------------------------------------------------------
# target_count path and boundary values
# ---------------------------------------------------------------------------


def test_target_count_path():
    _, faces = fast_simplification.simplify(PLANE_POINTS, PLANE_FACES, target_count=4)
    assert faces.shape[0] == 4


def test_target_count_full_is_noop():
    # Requesting the full face count removes nothing.
    points, faces = fast_simplification.simplify(
        PLANE_POINTS, PLANE_FACES, target_count=PLANE_FACES.shape[0]
    )
    assert faces.shape[0] == PLANE_FACES.shape[0]


def test_verbose_runs(capsys):
    # verbose=True must not crash and simply enables core logging.
    points, faces = fast_simplification.simplify(
        PLANE_POINTS, PLANE_FACES, target_reduction=0.5, verbose=True
    )
    assert faces.shape == (4, 3)


def test_agg_zero_preserves_more():
    # A low aggressiveness may fail to reach the target reduction, keeping
    # more faces than an aggressive pass.
    _, faces_agg0 = fast_simplification.simplify(
        PLANE_POINTS, PLANE_FACES, target_reduction=0.5, agg=0
    )
    _, faces_agg7 = fast_simplification.simplify(
        PLANE_POINTS, PLANE_FACES, target_reduction=0.5, agg=7
    )
    assert faces_agg0.shape[0] >= faces_agg7.shape[0]


# ---------------------------------------------------------------------------
# lossless + preserve_border on the array API
# ---------------------------------------------------------------------------


@skip_no_vtk
def test_lossless_preserves_geometry(sphere):
    triangles = sphere._connectivity_array.reshape(-1, 3)
    points, faces = fast_simplification.simplify(
        sphere.points, triangles, target_reduction=0.5, lossless=True
    )
    assert np.allclose(sphere.points, points)
    assert np.allclose(triangles, faces)


# ---------------------------------------------------------------------------
# simplify_mesh (load_from_vtk path) and non-triangle rejection
# ---------------------------------------------------------------------------


@skip_no_vtk
def test_simplify_mesh_rejects_non_triangles():
    # A plane before triangulation has quad cells; load_from_vtk must reject
    # it with the documented ValueError.
    quad_mesh = pv.Plane(i_resolution=3, j_resolution=3)
    assert not quad_mesh.is_all_triangles
    with pytest.raises(ValueError, match="only triangles"):
        fast_simplification.simplify_mesh(quad_mesh, target_reduction=0.5)


@skip_no_vtk
def test_simplify_mesh_target_count(sphere):
    out = fast_simplification.simplify_mesh(sphere, target_count=sphere.n_cells // 2)
    assert out.is_all_triangles
    assert out.n_cells == sphere.n_cells // 2


@skip_no_vtk
def test_simplify_mesh_carries_collapses(sphere):
    out = fast_simplification.simplify_mesh(sphere, target_reduction=0.5)
    collapses = out.field_data["fast_simplification_collapses"]
    assert collapses.ndim == 2 and collapses.shape[1] == 2


# ---------------------------------------------------------------------------
# Replay round-trips (offline)
# ---------------------------------------------------------------------------


def test_replay_matches_direct_plane():
    points_out, faces_out, collapses = fast_simplification.simplify(
        PLANE_POINTS, PLANE_FACES, target_reduction=0.5, return_collapses=True
    )
    replay_points, replay_faces, indice_mapping = fast_simplification.replay_simplification(
        PLANE_POINTS, PLANE_FACES, collapses
    )
    assert np.allclose(points_out, replay_points)
    assert np.array_equal(faces_out, replay_faces)
    # the indice mapping covers every input point
    assert indice_mapping.shape[0] == PLANE_POINTS.shape[0]


def test_replay_accepts_int64_collapses():
    _, _, collapses = fast_simplification.simplify(
        PLANE_POINTS, PLANE_FACES, target_reduction=0.5, return_collapses=True
    )
    rp, rf, mapping = fast_simplification.replay_simplification(
        PLANE_POINTS, PLANE_FACES, collapses.astype(np.int64)
    )
    assert rf.shape[1] == 3
    assert mapping.shape[0] == PLANE_POINTS.shape[0]


@skip_no_vtk
def test_replay_sphere_roundtrip(sphere):
    points = sphere.points
    faces = sphere.faces.reshape(-1, 4)[:, 1:]
    points_out, faces_out, collapses = fast_simplification.simplify(
        points, faces, target_reduction=0.5, return_collapses=True
    )
    replay_points, replay_faces, _ = fast_simplification.replay_simplification(
        points, faces, collapses
    )
    assert np.allclose(points_out, replay_points)
    assert np.array_equal(faces_out, replay_faces)
