# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True


import numpy as np

cimport numpy as np
from libc.stdint cimport int64_t
from libcpp cimport bool


cdef extern from "wrapper_replay.h" namespace "Replay":
    void load_arrays_int32(const int, const int, const int, float*, int*, int*)
    void load_arrays_int64(const int, const int, const int,  float*, int64_t*, int*)
    void replay_simplification()
    void get_points(float*)
    void get_triangles(int*)
    void get_collapses(int*)
    int get_faces_int32(int*)
    int get_faces_int32_no_padding(int*)
    int get_faces_int64(int64_t*)
    void write_obj(const char*)
    void load_obj(const char*, bool)
    int n_points()
    int n_triangles()
    int n_collapses()
    int load_triangles_from_vtk(const int, int*)
    void load_points(const int, float*)
    void load_collapses(const int, int*)

def load_int32(int n_points, int n_faces, int n_collapses, float [:, ::1] points, int [:, ::1] faces, int [:, ::1] collapses):
    load_arrays_int32(n_points, n_faces, n_collapses, &points[0, 0], &faces[0, 0], &collapses[0, 0])


def load_int64(
        int n_points, int n_faces, int n_collapses, float [:, ::1] points, int64_t [:, ::1] faces, int [:, ::1] collapses
):
    load_arrays_int64(n_points, n_faces, n_collapses, &points[0, 0], &faces[0, 0], &collapses[0, 0])


# def simplify(int target_count, double aggressiveness=7, bool verbose=False):
#     simplify_mesh(target_count, aggressiveness, verbose)

def replay():
    replay_simplification()


def save_obj(filename):
    py_byte_string = filename.encode('UTF-8')
    cdef char* c_filename = py_byte_string
    write_obj(c_filename)


def read(filename):
    py_byte_string = filename.encode('UTF-8')
    cdef char* c_filename = py_byte_string
    load_obj(c_filename, False)


def return_points():
    cdef float [:, ::1] points = np.empty((n_points(), 3), np.float32)
    get_points(&points[0, 0])
    return np.array(points)


def return_triangles():
    cdef int [:, ::1] triangles = np.empty((n_triangles(), 3), np.int32)
    get_triangles(&triangles[0, 0])
    return np.array(triangles)

def return_collapses():
    cdef int [:, ::1] collapses = np.empty((n_collapses(), 2), np.int32)
    get_collapses(&collapses[0, 0])
    return np.array(collapses)


def return_faces_int32_no_padding():
    """VTK formatted faces"""
    cdef int [::1] faces = np.empty(n_triangles()*3, np.int32)
    n_tri = get_faces_int32_no_padding(&faces[0])
    return np.array(faces[:n_tri*3])


def return_faces_int32():
    """VTK formatted faces"""
    cdef int [::1] faces = np.empty(n_triangles()*4, np.int32)
    n_tri = get_faces_int32(&faces[0])
    return np.array(faces[:n_tri*4])


def return_faces_int64():
    """VTK formatted faces"""
    cdef int64_t [::1] faces = np.empty(n_triangles()*4, np.int64)
    n_tri = get_faces_int64(&faces[0])
    return np.array(faces[:n_tri*4])


def load_from_vtk(int n_points, float [:, ::1] points, int [::1] faces, int n_faces):
    result = load_triangles_from_vtk(n_faces, &faces[0])
    if result:
        raise ValueError(
            "Input mesh ``mesh`` must consist of only triangles.\n"
            "Run ``.triangulate()`` to convert to an all triangle mesh."
        )
    load_points(n_points, &points[0, 0])

def compute_indice_mapping2(int [:, :] collapses, int n_points):

    cdef long[:] indice_mapping = np.arange(n_points, dtype=int)
    cdef int n_collapses = collapses.shape[0]
    cdef int i, j, k, lim

    lim = 0

    for i in range(n_points):

        k = 0
        while k == 0 and lim < 1000:
            lim += 1
            k = 1
            for j in range(n_collapses):
                if indice_mapping[i] == collapses[j, 1]:
                    k = 0
                    indice_mapping[i] = collapses[j, 0]
    
    return np.array(indice_mapping)



def compute_indice_mapping(collapses, n_points):

    # Compute the mapping from original indices to new indices

    # start with identity mapping
    indice_mapping = np.arange(n_points, dtype=int)
    # First round of mapping
    origin_indices = collapses[:, 1]
    indice_mapping[origin_indices] = collapses[:, 0]
    previous = np.zeros(len(indice_mapping))
    while not np.array_equal(previous, indice_mapping):
        previous = indice_mapping.copy()
        indice_mapping[origin_indices] = indice_mapping[
            indice_mapping[origin_indices]
        ]

    from time import time
    start = time()
    keep = np.setdiff1d(
        np.arange(n_points), collapses[:, 1]
    )  # Indices of the points that must be kept after decimation
    
    application = dict([keep[i], i] for i in range(len(keep)))
    indice_mapping = np.array([application[i] for i in indice_mapping])

    return indice_mapping

def compute_decimated_triangles(triangles, indice_mapping):

    triangles = indice_mapping[triangles.copy()]
    # compute the new triangles
    keep_triangle = (
        (triangles[:, 0] != triangles[:, 1])
        * (triangles[:, 1] != triangles[:, 2])
        * (triangles[:, 0] != triangles[:, 2])
    )
    return triangles[keep_triangle]


def compute_new_collapses(long [:, :] dec_triangles, long [:] isolated_points):

    cdef long[:, :] new_collapses = np.empty((len(isolated_points), 2), dtype=int)
    cdef int n_ip = len(isolated_points)
    cdef int i, j
    cdef long[:] t = np.zeros(3, dtype=int)

    for i in range(n_ip):
        new_collapses[i, 1] = isolated_points[i]
        new_collapses[i, 0] = -1

    for j in range(len(dec_triangles)):
        t = dec_triangles[j]
        for i in range(n_ip):
            if new_collapses[i, 0] == -1:
                if t[0] == isolated_points[i]:
                    if t[1] != isolated_points[i]:
                        new_collapses[i, 0] = t[1]
                    elif t[2] != isolated_points[i]:
                        new_collapses[i, 0] = t[2]

                elif t[1] == isolated_points[i]:
                    if t[0] != isolated_points[i]:
                        new_collapses[i, 0] = t[0]
                    elif t[2] != isolated_points[i]:
                        new_collapses[i, 0] = t[2]

                elif t[2] == isolated_points[i]:
                    if t[0] != isolated_points[i]:
                        new_collapses[i, 0] = t[0]
                    elif t[1] != isolated_points[i]:
                        new_collapses[i, 0] = t[1]

    return np.array(new_collapses)