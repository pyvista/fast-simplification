import pyvista as pv
from pyvista import examples
import fast_simplification
import numpy as np
from time import time


def triangles_to_faces(triangles):
    tmp = 3 * np.ones((len(triangles), 4), dtype=triangles.dtype)
    tmp[:, 1:] = triangles
    return tmp.copy().reshape(-1)


def faces_to_triangles(faces):
    return np.array(faces.reshape(-1, 4)[:, 1:], dtype=np.int64)


# load an example mesh
mesh = examples.download_louis_louvre()

points = mesh.points
faces = mesh.faces.reshape(-1, 4)[:, 1:]

start = time()
dec_points, dec_faces, collapses = fast_simplification.simplify(
    points, faces, 0.99, return_collapses=True
)
print(f"Decimation from {len(points)} points to {len(dec_points)} points")
print(f"Simplification: {time() - start}")

start = time()
dec_points2, dec_faces2 = fast_simplification.replay(
    points=points, triangles=faces, collapses=collapses
)
print(f"Replay (full): {time() - start}")


def foo(points, triangles, collapses):
    n_points = points.shape[0]

    # Collapse the points
    dec_points, _ = fast_simplification.replay(
        points=points, triangles=triangles, collapses=collapses
    )

    # Compute indice mapping
    keep = np.setdiff1d(
        np.arange(n_points), collapses[:, 1]
    )  # Indices of the points that must be kept after decimation
    # start with identity mapping
    indice_mapping = np.arange(n_points, dtype=int)
    # First round of mapping
    origin_indices = collapses[:, 1]
    indice_mapping[origin_indices] = collapses[:, 0]
    previous = np.zeros(len(indice_mapping))
    while not np.array_equal(previous, indice_mapping):
        previous = indice_mapping.copy()
        indice_mapping[origin_indices] = indice_mapping[indice_mapping[origin_indices]]
    application = dict([keep[i], i] for i in range(len(keep)))
    indice_mapping = np.array([application[i] for i in indice_mapping])

    # compute the new triangles
    dec_triangles = indice_mapping[triangles.copy()]
    keep_triangle = (
        (dec_triangles[:, 0] != dec_triangles[:, 1])
        * (dec_triangles[:, 1] != dec_triangles[:, 2])
        * (dec_triangles[:, 0] != dec_triangles[:, 2])
    )

    # identify isolated points
    isolated_points = np.setdiff1d(
        np.unique(indice_mapping), np.unique(dec_triangles[keep_triangle])
    )

    # compute the new collapses (for isolated points)
    new_collapses = np.empty((len(isolated_points), 2), dtype=int)
    for count, ip in enumerate(isolated_points):
        unique_neighbors = np.unique(
            np.concatenate([dec_triangles[i] for i in np.where(dec_triangles == ip)[0]])
        )
        unique_neighbors = np.setdiff1d(unique_neighbors, [ip])
        assert unique_neighbors[0] not in isolated_points
        new_collapses[count] = [unique_neighbors[0], ip]

    # Apply the new collapses
    for e in new_collapses:
        e0, e1 = e
        dec_triangles[dec_triangles == e1] = e0
        indice_mapping[indice_mapping == e1] = e0

    for ip in isolated_points:
        assert np.sum(indice_mapping == ip) == 0
        assert np.sum(dec_triangles == ip) == 0

    # Remove the isolated points
    isolated_points = np.sort(isolated_points)[::-1]
    for ip in isolated_points:
        dec_points = np.concatenate([dec_points[:ip], dec_points[ip + 1 :]], axis=0)
        # assert np.sum(indice_mapping == ip) == 0
        indice_mapping[indice_mapping > ip] -= 1
        dec_triangles[dec_triangles > ip] -= 1

    return dec_points, dec_triangles, indice_mapping


def compute_indice_mapping(collapses, n_points):
    # Compute the mapping from original indices to new indices
    keep = np.setdiff1d(
        np.arange(n_points), collapses[:, 1]
    )  # Indices of the points that must be kept after decimation
    # start with identity mapping
    indice_mapping = np.arange(n_points, dtype=int)
    # First round of mapping
    origin_indices = collapses[:, 1]
    indice_mapping[origin_indices] = collapses[:, 0]
    previous = np.zeros(len(indice_mapping))
    while not np.array_equal(previous, indice_mapping):
        previous = indice_mapping.copy()
        indice_mapping[origin_indices] = indice_mapping[indice_mapping[origin_indices]]
    application = dict([keep[i], i] for i in range(len(keep)))
    indice_mapping = np.array([application[i] for i in indice_mapping])

    return indice_mapping


def compute_decimated_triangles(triangles, indice_mapping, collapses):
    triangles = indice_mapping[triangles.copy()]
    # compute the new triangles
    keep_triangle = (
        (triangles[:, 0] != triangles[:, 1])
        * (triangles[:, 1] != triangles[:, 2])
        * (triangles[:, 0] != triangles[:, 2])
    )

    return triangles[keep_triangle]


indice_mapping = compute_indice_mapping(collapses, mesh.points.shape[0])
newtriangles = compute_decimated_triangles(faces, indice_mapping, collapses)
i, j = np.random.randint(0, len(mesh.points), 2)

start = time()
dec_points3, dec_faces3, indice_mapping3 = foo(
    points=points, triangles=faces, collapses=collapses
)
print(f"Replay (foo): {time() - start}")


p = pv.Plotter(shape=(2, 2))
p.subplot(0, 0)
p.add_mesh(mesh)
p.add_points(mesh.points[i], color="red", point_size=10, render_points_as_spheres=True)
p.add_points(mesh.points[j], color="blue", point_size=10, render_points_as_spheres=True)
p.add_text(
    f"Original mesh, {mesh.points.shape[0]} points, {len(np.unique(triangles_to_faces(faces)))}",
    font_size=10,
)
p.subplot(0, 1)
p.add_mesh(pv.PolyData(dec_points, faces=triangles_to_faces(dec_faces)))
p.add_points(
    dec_points[indice_mapping3[i]],
    color="red",
    point_size=10,
    render_points_as_spheres=True,
)
p.add_points(
    dec_points[indice_mapping3[j]],
    color="blue",
    point_size=10,
    render_points_as_spheres=True,
)
p.add_text(
    f"Decimated mesh, {dec_points.shape[0]} points, {len(np.unique(triangles_to_faces(dec_faces)))}",
    font_size=10,
)
p.subplot(1, 0)
p.add_mesh(pv.PolyData(dec_points2, faces=triangles_to_faces(dec_faces2)))
p.add_points(
    dec_points2[indice_mapping[i]],
    color="red",
    point_size=10,
    render_points_as_spheres=True,
)
p.add_points(
    dec_points2[indice_mapping[j]],
    color="blue",
    point_size=10,
    render_points_as_spheres=True,
)
p.add_text(
    f"Replayed mesh, {dec_points2.shape[0]} points, {len(np.unique(triangles_to_faces(dec_faces2)))}",
    font_size=10,
)

isolated_points = np.setdiff1d(
    np.arange(dec_points2.shape[0]), np.unique(triangles_to_faces(dec_faces2))
)
p.add_points(
    dec_points2[isolated_points],
    color="green",
    point_size=10,
    render_points_as_spheres=True,
)

p.subplot(1, 1)
p.add_mesh(pv.PolyData(dec_points3, faces=triangles_to_faces(dec_faces3)))
p.add_points(
    dec_points3[indice_mapping3[i]],
    color="red",
    point_size=10,
    render_points_as_spheres=True,
)
p.add_points(
    dec_points3[indice_mapping3[j]],
    color="blue",
    point_size=10,
    render_points_as_spheres=True,
)
p.add_text(
    f"Replayed mesh, {dec_points3.shape[0]} points, {len(np.unique(triangles_to_faces(dec_faces3)))}",
    font_size=10,
)


p.show()


def lexsort(points):
    return points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]


sorted_dec_points = lexsort(dec_points)
sorted_dec_points3 = lexsort(dec_points3)

print(len(mesh.points) - len(collapses))
print(np.allclose(sorted_dec_points, sorted_dec_points3))
print(sorted_dec_points - sorted_dec_points3)

p = pv.Plotter()
p.add_mesh(
    pv.PolyData(dec_points3, faces=triangles_to_faces(dec_faces3)),
    color="blue",
    opacity=0.9,
)
p.add_mesh(
    pv.PolyData(dec_points, faces=triangles_to_faces(dec_faces)),
    color="red",
    opacity=0.9,
)
p.show()
