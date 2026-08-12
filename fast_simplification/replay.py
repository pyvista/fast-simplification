import numpy as np

from . import _replay
from .utils import ascontiguous


def _map_isolated_points(points, edges, triangles, return_outliers=False):
    r"""Map the isolated points to the triangles.

    (points, edges, triangles) represents a structure. The goal of this function
    is to compute a mapping array such that the points that are not in the triangles
    but are in the edges are merged into the points that are in the triangles, with
    respect to the edges. An example is given below.

          (1)
         / | \\
      (0)  |  (2)-3
         \ | /  \\
          (4)    6-9
           |
           5     8-7

    In this example, the points 5, 3, 4, 7, 8, 9 are not connected to any triangle.
    The expected mapping is:

    0 -> 0
    1 -> 1
    2 -> 2
    3 -> 2
    4 -> 4
    5 -> 4
    6 -> 2
    7 -> 7 (7 cannot be merged into any point in the triangles)
    8 -> 8 (8 cannot be merged into any point in the triangles)
    9 -> 2

    The output will be the mapping array and the merged points array. In this example,
    the mapping array is [0, 1, 2, 2, 4, 4, 2, 7, 8, 2] and the merged points array is
    [3, 5, 6, 9]. The points 7 and 8 are outliers. If return_outliers is True,
    the function will return the mapping array, the merged points array and the
    isolated points array. Else, the function will return the mapping array and the
    merged points array.

    Parameters
    ----------
        points : sequence
            array of points
        edges : sequence
            array of edges
        triangles : sequence
            array of triangles
        return_outsider : bool
            if True, return the outliers

    Returns
    -------
        np.ndarray
            mapping array
        np.ndarray
            merged points array
    """
    n_points = points.shape[0]

    # The points to connect are the points that are not in the triangles
    # but are in the edges
    points_to_connect = np.intersect1d(
        np.setdiff1d(np.arange(n_points), np.unique(triangles)), np.unique(edges)
    )
    # Start with the identity mapping
    mapping = np.arange(n_points, dtype=np.int64)

    # Remove edges that do not contains points to connect
    edges = edges[np.isin(edges, points_to_connect).any(axis=1)]

    n_edges = edges.shape[0]
    n_edges_old = 0

    # Iterate until there is no more edges to collapse
    # or until a statiionary state is reached
    while n_edges > 0 and n_edges != n_edges_old:
        n_edges_old = n_edges

        # Edges that connect two points to connect
        # are kept for the next iteration
        keep = np.isin(edges, points_to_connect).all(axis=1)

        # Edges that connect a point to connect to a point
        # that is not to connect are merged
        connexions = edges[~keep]

        a = np.isin(connexions, points_to_connect)
        merged = connexions[np.where(a)]
        target = connexions[np.where(~a)]

        # Update the mapping array and the points to connect
        mapping[merged] = mapping[target]
        points_to_connect = np.setdiff1d(points_to_connect, merged)

        # Remove the edges that are merged
        edges = edges[keep]
        # Remove edges that do not contains points to connect
        edges = edges[np.isin(edges, points_to_connect).any(axis=1)]
        n_edges = edges.shape[0]

    # The points that have been merged are the ones
    # such that mapping[i] != i
    merged_points = np.where(mapping != np.arange(len(mapping)))[0]

    if return_outliers:
        isolated_points = points_to_connect
        return mapping, merged_points, isolated_points
    return mapping, merged_points


@ascontiguous
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
        ``collapses[i] = [i0, i1]`` means that during the i-th
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

    # Vertices that are not referenced by any triangle are dropped by the
    # decimation core, but the pure-array bookkeeping below
    # (``compute_indice_mapping``) retains them and assigns them an index.
    # That shifts the index of every vertex that follows an unreferenced one,
    # so ``indice_mapping`` ends up pointing past the end of the decimated
    # points and ``_map_isolated_points`` / the final remap raise an
    # ``IndexError`` (issue #60). Strip the unreferenced vertices up front so
    # the numbering is dense, then re-expand ``indice_mapping`` to the original
    # vertex count (unreferenced vertices map to ``-1``) before returning.
    n_points_full = points.shape[0]
    referenced = np.zeros(n_points_full, dtype=bool)
    if triangles.size:
        referenced[np.unique(triangles)] = True
    unreferenced_present = not referenced.all()
    if unreferenced_present:
        kept_vertices = np.flatnonzero(referenced)
        old_to_new = np.full(n_points_full, -1, dtype=np.int64)
        old_to_new[kept_vertices] = np.arange(kept_vertices.shape[0])
        points = np.ascontiguousarray(points[kept_vertices])
        triangles = np.ascontiguousarray(old_to_new[triangles].astype(triangles.dtype, copy=False))
        collapses = np.asarray(collapses)
        if collapses.size:
            remapped = old_to_new[collapses]
            # An unreferenced vertex has no incident edge, so it can never be
            # collapsed; a negative entry here means the collapses do not
            # belong to this mesh.
            if (remapped < 0).any():
                raise ValueError(
                    "``collapses`` references a vertex that is not part of any "
                    "triangle; the collapses do not match the provided mesh."
                )
            collapses = np.ascontiguousarray(remapped.astype(np.int32, copy=False))

    if triangles.dtype == np.int32:
        load = _replay.load_int32
    elif triangles.dtype == np.int64:
        load = _replay.load_int64
    else:
        load = _replay.load_int32
        triangles = triangles.astype(np.int32)

    # Collapse the points
    n_faces = triangles.shape[0]
    n_points = points.shape[0]
    load(n_points, n_faces, collapses.shape[0], points, triangles, collapses)
    _replay.replay()
    dec_points = _replay.return_points()

    # Compute the indice mapping
    indice_mapping = _replay.compute_indice_mapping(collapses, len(points))

    # compute the new triangles
    # Apply the indice mapping to the triangles
    mapped_triangles = indice_mapping[triangles.copy()]

    # Extract the edges and the triangles
    # Edges can be repeated, but this is not a problem
    # and it is faster to do so
    dec_edges, dec_triangles = _replay.clean_triangles_and_edges(mapped_triangles)

    # Map the isolated points to the triangles
    mapping, points_to_merge, outliers = _map_isolated_points(
        dec_points, dec_edges, dec_triangles, return_outliers=True
    )

    dec_triangles = mapping[dec_triangles]
    indice_mapping = mapping[indice_mapping]

    points_to_merge = np.union1d(points_to_merge, outliers)
    # Remove the isolated points
    # isolated_points = new_collapses[:, 1]
    points_to_merge = np.sort(points_to_merge)[::-1]
    mapping = np.arange(dec_points.shape[0])
    for ip in points_to_merge:
        dec_points = np.delete(dec_points, ip, axis=0)
        mapping[ip:] -= 1
    indice_mapping = mapping[indice_mapping]
    dec_triangles = mapping[dec_triangles]

    # Re-expand the mapping to the original vertex count; vertices that were
    # unreferenced (and therefore not part of the decimated mesh) map to -1.
    if unreferenced_present:
        full_indice_mapping = np.full(n_points_full, -1, dtype=indice_mapping.dtype)
        full_indice_mapping[kept_vertices] = indice_mapping
        indice_mapping = full_indice_mapping

    return dec_points, dec_triangles, indice_mapping
