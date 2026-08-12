// nanobind bindings for the Fast-Quadric-Mesh-Simplification core.
//
// This module mirrors the surface previously provided by the Cython module
// ``fast_simplification._simplify``. The C++ core (Simplify.h / wrapper.h) is
// unchanged; only the Python binding layer is reimplemented on nanobind.

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "array_support.h"
#include "wrapper.h"

namespace nb = nanobind;
using namespace nb::literals;

// ---------------------------------------------------------------------------
// Loaders
// ---------------------------------------------------------------------------

static void load_int32(int n_points, int n_faces,
                       InArray<double, 2> points,
                       InArray<int32_t, 2> faces) {
  Simplify::load_arrays_int32(n_points, n_faces, points.data(), faces.data());
}

static void load_int64(int n_points, int n_faces,
                       InArray<double, 2> points,
                       InArray<int64_t, 2> faces) {
  Simplify::load_arrays_int64(n_points, n_faces, points.data(), faces.data());
}

static void load_from_vtk(int n_points, InArray<double, 2> points,
                          InArray<int32_t, 1> faces, int n_faces) {
  int result = Simplify::load_triangles_from_vtk(n_faces, faces.data());
  if (result) {
    throw std::invalid_argument(
        "Input mesh ``mesh`` must consist of only triangles.\n"
        "Run ``.triangulate()`` to convert to an all triangle mesh.");
  }
  Simplify::load_points(n_points, points.data());
}

// ---------------------------------------------------------------------------
// Simplification
// ---------------------------------------------------------------------------

static void simplify(int target_count, double aggressiveness, bool verbose,
                     bool preserve_border) {
  Simplify::simplify_mesh(target_count, aggressiveness, verbose, preserve_border);
}

static void simplify_lossless(bool verbose, bool preserve_border) {
  Simplify::simplify_mesh_lossless(verbose, preserve_border);
}

static void save_obj(const std::string &filename) {
  Simplify::write_obj(filename.c_str());
}

static void read_obj(const std::string &filename) {
  Simplify::load_obj(filename.c_str(), false);
}

// ---------------------------------------------------------------------------
// Result accessors
// ---------------------------------------------------------------------------

static NDArray<double, 2> return_points() {
  int n = Simplify::n_points();
  auto arr = MakeNDArray<double, 2>({n, 3});
  Simplify::get_points(arr.data());
  return arr;
}

static NDArray<int32_t, 2> return_triangles() {
  int n = Simplify::n_triangles();
  auto arr = MakeNDArray<int32_t, 2>({n, 3});
  Simplify::get_triangles(arr.data());
  return arr;
}

static NDArray<int32_t, 2> return_collapses() {
  int n = Simplify::n_collapses();
  auto arr = MakeNDArray<int32_t, 2>({n, 2});
  Simplify::get_collapses(arr.data());
  return arr;
}

static NDArray<int32_t, 1> return_faces_int32_no_padding() {
  size_t cap = (size_t)Simplify::n_triangles() * 3;
  int32_t *buf = (int32_t *)std::malloc((cap ? cap : 1) * sizeof(int32_t));
  if (!buf)
    throw std::bad_alloc();
  int n_tri = Simplify::get_faces_int32_no_padding(buf);
  return WrapFlat<int32_t>(buf, (size_t)n_tri * 3);
}

static NDArray<int32_t, 1> return_faces_int32() {
  size_t cap = (size_t)Simplify::n_triangles() * 4;
  int32_t *buf = (int32_t *)std::malloc((cap ? cap : 1) * sizeof(int32_t));
  if (!buf)
    throw std::bad_alloc();
  int n_tri = Simplify::get_faces_int32(buf);
  return WrapFlat<int32_t>(buf, (size_t)n_tri * 4);
}

static NDArray<int64_t, 1> return_faces_int64() {
  size_t cap = (size_t)Simplify::n_triangles() * 4;
  int64_t *buf = (int64_t *)std::malloc((cap ? cap : 1) * sizeof(int64_t));
  if (!buf)
    throw std::bad_alloc();
  int n_tri = Simplify::get_faces_int64(buf);
  return WrapFlat<int64_t>(buf, (size_t)n_tri * 4);
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

NB_MODULE(_simplify, m) {
  m.def("load_int32", &load_int32, "n_points"_a, "n_faces"_a, "points"_a,
        "faces"_a);
  m.def("load_int64", &load_int64, "n_points"_a, "n_faces"_a, "points"_a,
        "faces"_a);
  m.def("load_from_vtk", &load_from_vtk, "n_points"_a, "points"_a, "faces"_a,
        "n_faces"_a);
  m.def("simplify", &simplify, "target_count"_a, "aggressiveness"_a = 7.0,
        "verbose"_a = false, "preserve_border"_a = false);
  m.def("simplify_lossless", &simplify_lossless, "verbose"_a = false,
        "preserve_border"_a = false);
  m.def("save_obj", &save_obj, "filename"_a);
  m.def("read", &read_obj, "filename"_a);
  m.def("return_points", &return_points);
  m.def("return_triangles", &return_triangles);
  m.def("return_collapses", &return_collapses);
  m.def("return_faces_int32_no_padding", &return_faces_int32_no_padding);
  m.def("return_faces_int32", &return_faces_int32);
  m.def("return_faces_int64", &return_faces_int64);
  m.def("n_points", &Simplify::n_points);
  m.def("n_triangles", &Simplify::n_triangles);
  m.def("n_collapses", &Simplify::n_collapses);
}
