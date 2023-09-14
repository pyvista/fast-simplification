from . import _replay


def replay_simplification(points, triangles, collapses):
    """Replay the decimation of a triangular mesh.

    Parameters
    ----------
    points : sequence
        A ``(n, 3)`` array of points. May be a ``numpy.ndarray`` or a
        list of points. For efficiency, provide points as a float32
        array.
    triangles : sequence
        A ``(n, 3)`` array of triangle indices. May be a
        ``numpy.ndarray`` or a list of triangle indices. For
        efficiency, provide points as a float32 array.
    collapses : sequence
        The collapses to replay.
        A ``(n, 2)`` numpy.ndarray of collapses.
        ``collaspes[i] = [i0, i1]`` means that durint the i-th
        collapse, the vertex ``i1`` was collapsed into the vertex
        ``i0``.

    Returns
    -------
    np.ndarray
        Points array.
    np.ndarray
        Triangles array.
    np.ndarray
        indice_mapping array.
        A ``(n,)`` array of indices.
        ``indice_mapping[i] = j`` means that the vertex ``i`` of
        the original mesh was mapped to the vertex ``j`` of the
        decimated mesh.

    """
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
        raise ValueError(f"Expected ``triangles`` array to be (n, 3), not {triangles.shape}")

    if not triangles.flags.c_contiguous:
        triangles = np.ascontiguousarray(triangles)

    if triangles.dtype == np.int32:
        load = _replay.load_int32
    elif triangles.dtype == np.int64:
        load = _replay.load_int64
    else:
        load = _replay.load_int32
        triangles = triangles.astype(np.int32)

    n_faces = triangles.shape[0]
    n_points = points.shape[0]

    # Collapse the points
    load(points.shape[0], n_faces, collapses.shape[0], points, triangles, collapses)
    _replay.replay()
    dec_points = _replay.return_points()

    # Compute the indice mapping
    indice_mapping = _replay.compute_indice_mapping(collapses, len(points))

    # compute the new triangles
    # Apply the indice mapping to the triangles
    mapped_triangles = indice_mapping[triangles.copy()]

    # Remove degenerate triangles
    keep_triangle = (
        (mapped_triangles[:, 0] != mapped_triangles[:, 1])
        * (mapped_triangles[:, 1] != mapped_triangles[:, 2])
        * (mapped_triangles[:, 0] != mapped_triangles[:, 2])
    )
    dec_triangles = mapped_triangles[keep_triangle]

    # Compute the edges of the decimated mesh
    keep_edges0 = (mapped_triangles[:, 1] == mapped_triangles[:, 2]) * (
        mapped_triangles[:, 0] != mapped_triangles[:, 1]
    )
    keep_edges1 = (mapped_triangles[:, 0] == mapped_triangles[:, 2]) * (
        mapped_triangles[:, 1] != mapped_triangles[:, 0]
    )
    keep_edges2 = (mapped_triangles[:, 0] == mapped_triangles[:, 1]) * (
        mapped_triangles[:, 2] != mapped_triangles[:, 0]
    )
    dec_edges = np.concatenate(
        [
            mapped_triangles[keep_edges0, :][:, [0, 1]],
            mapped_triangles[keep_edges1, :][:, [1, 2]],
            mapped_triangles[keep_edges2, :][:, [0, 2]],
        ],
        axis=0,
    )

    # identify isolated points among the decimated points
    # there are the points that are in the image of
    # the indice mapping but not connected to any of
    # the decimated triangles
    isolated_points = np.setdiff1d(np.unique(indice_mapping), np.unique(dec_triangles))

    # Compute the new collapses : the isolated points
    # are collapsed to one of the points with which
    # they share an edge
    new_collapses = _replay.compute_new_collapses_from_edges(dec_edges, isolated_points)

    # Apply the new collapses)
    mapping = np.arange(dec_points.shape[0])
    for e in new_collapses:
        e0, e1 = e
        mapping[e1] = e0

    dec_triangles = mapping[dec_triangles]
    indice_mapping = mapping[indice_mapping]

    # Remove the isolated points
    isolated_points = np.sort(isolated_points)[::-1]
    mapping = np.arange(dec_points.shape[0])
    for ip in isolated_points:
        dec_points = np.delete(dec_points, ip, axis=0)
        mapping[ip:] -= 1
    indice_mapping = mapping[indice_mapping]
    dec_triangles = mapping[dec_triangles]

    return dec_points, dec_triangles, indice_mapping
