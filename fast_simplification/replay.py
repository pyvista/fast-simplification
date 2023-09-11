from . import _replay


def replay(
    points,
    triangles,
    collapses,
):
    import numpy as np

    if not isinstance(points, np.ndarray):
        points = np.array(points, dtype=np.float32)
    if not isinstance(triangles, np.ndarray):
        triangles = np.array(triangles, dtype=np.int32)

    if points.ndim != 2:
        raise ValueError("``points`` array must be 2 dimensional")
    if points.shape[1] != 3:
        raise ValueError(f"Expected ``points`` array to be (n, 3), not {points.shape}")

    if triangles.ndim != 2:
        raise ValueError("``triangles`` array must be 2 dimensional")
    if triangles.shape[1] != 3:
        raise ValueError(
            f"Expected ``triangles`` array to be (n, 3), not {triangles.shape}"
        )

    n_faces = triangles.shape[0]
    # target_count = _check_args(target_reduction, target_count, n_faces)

    if not triangles.flags.c_contiguous:
        triangles = np.ascontiguousarray(triangles)

    if triangles.dtype == np.int32:
        load = _replay.load_int32
    elif triangles.dtype == np.int64:
        load = _replay.load_int64
    else:
        load = _replay.load_int32
        triangles = triangles.astype(np.int32)

    indice_mapping = _replay.compute_indice_mapping(
        collapses=collapses, n_points=points.shape[0]
    )
    dec_triangles = _replay.compute_decimated_triangles(triangles, indice_mapping)

    load(points.shape[0], n_faces, collapses.shape[0], points, triangles, collapses)
    _replay.replay()

    indice_mapping = _replay.compute_indice_mapping(
        collapses=collapses, n_points=points.shape[0]
    )

    return _replay.return_points(), dec_triangles


def indice_mapping(points, collapses):
    return _replay.compute_indice_mapping(collapses=collapses, n_points=points.shape[0])
