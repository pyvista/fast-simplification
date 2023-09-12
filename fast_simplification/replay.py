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


import numpy as np
from time import time


def replay2(points, triangles, collapses):
    n_points = points.shape[0]

    start_global = time()

    start_first_collapse = time()
    # Collapse the points
    dec_points, _ = replay(points=points, triangles=triangles, collapses=collapses)
    time_first_collapse = time() - start_first_collapse

    start_fitst_im = time()
    # Compute indice mapping
    # keep = np.setdiff1d(
    #     np.arange(n_points), collapses[:, 1]
    # )  # Indices of the points that must be kept after decimation
    # # start with identity mapping
    # indice_mapping = np.arange(n_points, dtype=int)
    # # First round of mapping
    # origin_indices = collapses[:, 1]
    # indice_mapping[origin_indices] = collapses[:, 0]
    # previous = np.zeros(len(indice_mapping))
    # while not np.array_equal(previous, indice_mapping):
    #     previous = indice_mapping.copy()
    #     indice_mapping[origin_indices] = indice_mapping[indice_mapping[origin_indices]]
    # application = dict([keep[i], i] for i in range(len(keep)))
    # indice_mapping = np.array([application[i] for i in indice_mapping])
    indice_mapping = _replay.compute_indice_mapping(collapses, len(points))
    time_first_im = time() - start_fitst_im

    start_dec_triangles = time()
    # compute the new triangles
    dec_triangles = indice_mapping[triangles.copy()]
    keep_triangle = (
        (dec_triangles[:, 0] != dec_triangles[:, 1])
        * (dec_triangles[:, 1] != dec_triangles[:, 2])
        * (dec_triangles[:, 0] != dec_triangles[:, 2])
    )
    # dec_triangles = dec_triangles[keep_triangle]
    time_dec_triangles = time() - start_dec_triangles

    start_find_ip = time()
    # identify isolated points
    isolated_points = np.setdiff1d(
        np.unique(indice_mapping), np.unique(dec_triangles[keep_triangle])
    )
    time_find_ip = time() - start_find_ip

    start_new_collapses = time()
    # compute the new collapses (for isolated points)
    # new_collapses = np.empty((len(isolated_points), 2), dtype=int)
    # for count, ip in enumerate(isolated_points):
    #     unique_neighbors = np.unique(
    #         np.concatenate([dec_triangles[i] for i in np.where(dec_triangles == ip)[0]])
    #     )
    #     unique_neighbors = np.setdiff1d(unique_neighbors, [ip])
    #     assert unique_neighbors[0] not in isolated_points
    #     new_collapses[count] = [unique_neighbors[0], ip]

    new_collapses = _replay.compute_new_collapses(dec_triangles, isolated_points)
    time_new_collapses = time() - start_new_collapses

    start_apply_new_collapses = time()
    # Apply the new collapses
    mapping = np.arange(dec_points.shape[0])
    for e in new_collapses:
        e0, e1 = e
        mapping[e1] = e0
        # dec_triangles[dec_triangles == e1] = e0
        # indice_mapping[indice_mapping == e1] = e0

    dec_triangles = mapping[dec_triangles]
    indice_mapping = mapping[indice_mapping]

    # for ip in isolated_points:
    #     assert np.sum(indice_mapping == ip) == 0
    #     assert np.sum(dec_triangles == ip) == 0

    time_apply_new_collapses = time() - start_apply_new_collapses

    start_remove_ip = time()
    # Remove the isolated points
    isolated_points = np.sort(isolated_points)[::-1]

    mapping = np.arange(dec_points.shape[0])
    for ip in isolated_points:
        dec_points = np.delete(dec_points, ip, axis=0)
        mapping[ip:] -= 1

    indice_mapping = mapping[indice_mapping]
    dec_triangles = mapping[dec_triangles]

    time_remove_ip = time() - start_remove_ip

    time_global = time() - start_global
    print()
    print(f"Total: {time_global}")
    print(
        f"First collapse: {time_first_collapse}, {100*time_first_collapse/time_global}%"
    )
    print(f"First indice mapping: {time_first_im}, {100*time_first_im/time_global}%")
    print(f"Dec triangles: {time_dec_triangles}, {100*time_dec_triangles/time_global}%")
    print(f"Find isolated points: {time_find_ip}, {100*time_find_ip/time_global}%")
    print(f"New collapses: {time_new_collapses}, {100*time_new_collapses/time_global}%")
    print(
        f"Apply new collapses: {time_apply_new_collapses}, {100*time_apply_new_collapses/time_global}%"
    )
    print(
        f"Remove isolated points: {time_remove_ip}, {100*time_remove_ip/time_global}%"
    )
    print()

    return dec_points, dec_triangles, indice_mapping
