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

start = time()
dec_points3, dec_faces3 = fast_simplification.replay(
    points=points, triangles=faces, collapses=collapses[0 : int(len(collapses) / 2)]
)
print(f"Replay (half): {time() - start}")
indice_mapping3 = fast_simplification.indice_mapping(
    mesh.points, collapses[0 : int(len(collapses) / 2)]
)

indice_mapping = fast_simplification.indice_mapping(mesh.points, collapses)
i, j = np.random.randint(0, len(mesh.points), 2)


p = pv.Plotter(shape=(2, 2))
p.subplot(0, 0)
p.add_mesh(mesh)
p.add_points(mesh.points[i], color="red", point_size=10, render_points_as_spheres=True)
p.add_points(mesh.points[j], color="blue", point_size=10, render_points_as_spheres=True)
p.subplot(0, 1)
p.add_mesh(pv.PolyData(dec_points, faces=triangles_to_faces(dec_faces)))
p.add_points(
    dec_points[indice_mapping[i]],
    color="red",
    point_size=10,
    render_points_as_spheres=True,
)
p.add_points(
    dec_points[indice_mapping[j]],
    color="blue",
    point_size=10,
    render_points_as_spheres=True,
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
p.show()
