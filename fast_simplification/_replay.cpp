// nanobind bindings for the replay half of fast-simplification.
//
// Mirrors the surface previously provided by the Cython module
// ``fast_simplification._replay``. The C++ core (Replay.h / wrapper_replay.h)
// is unchanged. The two array-processing helpers ``compute_indice_mapping``
// and ``clean_triangles_and_edges`` were pure Cython routines; they are
// reimplemented here with identical semantics so ``replay.py`` keeps working
// unchanged.

#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include "array_support.h"
#include "wrapper_replay.h"

namespace nb = nanobind;
using namespace nb::literals;

// ---------------------------------------------------------------------------
// Loaders (replay uses float32 points)
// ---------------------------------------------------------------------------

static void load_int32(int n_points, int n_faces, int n_collapses,
                       InArray<float, 2> points, InArray<int32_t, 2> faces,
                       InArray<int32_t, 2> collapses) {
  Replay::load_arrays_int32(n_points, n_faces, n_collapses, points.data(),
                            faces.data(), collapses.data());
}

static void load_int64(int n_points, int n_faces, int n_collapses,
                       InArray<float, 2> points, InArray<int64_t, 2> faces,
                       InArray<int32_t, 2> collapses) {
  Replay::load_arrays_int64(n_points, n_faces, n_collapses, points.data(),
                            faces.data(), collapses.data());
}

static void load_from_vtk(int n_points, InArray<float, 2> points,
                          InArray<int32_t, 1> faces, int n_faces) {
  int result = Replay::load_triangles_from_vtk(n_faces, faces.data());
  if (result) {
    throw std::invalid_argument(
        "Input mesh ``mesh`` must consist of only triangles.\n"
        "Run ``.triangulate()`` to convert to an all triangle mesh.");
  }
  Replay::load_points(n_points, points.data());
}

static void replay() { Replay::replay_simplification(); }

static void save_obj(const std::string &filename) {
  Replay::write_obj(filename.c_str());
}

static void read_obj(const std::string &filename) {
  Replay::load_obj(filename.c_str(), false);
}

// ---------------------------------------------------------------------------
// Result accessors (replay points are float32)
// ---------------------------------------------------------------------------

static NDArray<float, 2> return_points() {
  int n = Replay::n_points();
  auto arr = MakeNDArray<float, 2>({n, 3});
  Replay::get_points(arr.data());
  return arr;
}

static NDArray<int32_t, 2> return_triangles() {
  int n = Replay::n_triangles();
  auto arr = MakeNDArray<int32_t, 2>({n, 3});
  Replay::get_triangles(arr.data());
  return arr;
}

static NDArray<int32_t, 2> return_collapses() {
  int n = Replay::n_collapses();
  auto arr = MakeNDArray<int32_t, 2>({n, 2});
  Replay::get_collapses(arr.data());
  return arr;
}

static NDArray<int32_t, 1> return_faces_int32_no_padding() {
  size_t cap = (size_t)Replay::n_triangles() * 3;
  int32_t *buf = (int32_t *)std::malloc((cap ? cap : 1) * sizeof(int32_t));
  if (!buf)
    throw std::bad_alloc();
  int n_tri = Replay::get_faces_int32_no_padding(buf);
  return WrapFlat<int32_t>(buf, (size_t)n_tri * 3);
}

static NDArray<int32_t, 1> return_faces_int32() {
  size_t cap = (size_t)Replay::n_triangles() * 4;
  int32_t *buf = (int32_t *)std::malloc((cap ? cap : 1) * sizeof(int32_t));
  if (!buf)
    throw std::bad_alloc();
  int n_tri = Replay::get_faces_int32(buf);
  return WrapFlat<int32_t>(buf, (size_t)n_tri * 4);
}

static NDArray<int64_t, 1> return_faces_int64() {
  size_t cap = (size_t)Replay::n_triangles() * 4;
  int64_t *buf = (int64_t *)std::malloc((cap ? cap : 1) * sizeof(int64_t));
  if (!buf)
    throw std::bad_alloc();
  int n_tri = Replay::get_faces_int64(buf);
  return WrapFlat<int64_t>(buf, (size_t)n_tri * 4);
}

// ---------------------------------------------------------------------------
// Array-processing helpers (formerly pure Cython)
// ---------------------------------------------------------------------------

// Compute the mapping from original indices to new indices after collapsing
// edges. Reproduces the numpy/Cython implementation exactly, including the
// last-wins semantics of assigning through duplicated origin indices and the
// simultaneous read-then-write of numpy fancy indexing.
static NDArray<int32_t, 1>
compute_indice_mapping(InArray<int32_t, 2> collapses, int n_points) {
  size_t n_coll = collapses.shape(0);
  const int32_t *cptr = (n_coll > 0) ? collapses.data() : nullptr;

  // origin_indices = collapses[:, 1], targets = collapses[:, 0]
  std::vector<int64_t> mapping((size_t)n_points);
  for (int i = 0; i < n_points; ++i)
    mapping[(size_t)i] = i;

  // indice_mapping[origin_indices] = collapses[:, 0]  (last wins on dups)
  for (size_t k = 0; k < n_coll; ++k) {
    int64_t origin = cptr[2 * k + 1];
    int64_t target = cptr[2 * k + 0];
    mapping[(size_t)origin] = target;
  }

  // Iterate indice_mapping[origin] = indice_mapping[indice_mapping[origin]]
  // until a stationary state is reached. Each iteration reads the full
  // pre-iteration state (RHS) before writing (matches numpy semantics).
  if (n_coll > 0) {
    std::vector<int64_t> rhs(n_coll);
    bool changed = true;
    while (changed) {
      for (size_t k = 0; k < n_coll; ++k) {
        int64_t origin = cptr[2 * k + 1];
        rhs[k] = mapping[(size_t)mapping[(size_t)origin]];
      }
      changed = false;
      for (size_t k = 0; k < n_coll; ++k) {
        int64_t origin = cptr[2 * k + 1];
        if (mapping[(size_t)origin] != rhs[k]) {
          changed = true;
          mapping[(size_t)origin] = rhs[k];
        }
      }
    }
  }

  // keep = setdiff1d(arange(n_points), collapses[:, 1]) is the sorted set of
  // indices that are never an origin. application[i] is the rank of i among
  // kept indices, or 0 for removed indices.
  std::vector<char> removed((size_t)n_points, 0);
  for (size_t k = 0; k < n_coll; ++k) {
    int64_t origin = cptr[2 * k + 1];
    if (origin >= 0 && origin < n_points)
      removed[(size_t)origin] = 1;
  }
  std::vector<int32_t> application((size_t)n_points, 0);
  int32_t rank = 0;
  for (int i = 0; i < n_points; ++i) {
    if (!removed[(size_t)i]) {
      application[(size_t)i] = rank;
      ++rank;
    }
  }

  // indice_mapping = application[indice_mapping]
  auto out = MakeNDArray<int32_t, 1>({n_points});
  int32_t *optr = out.data();
  for (int i = 0; i < n_points; ++i)
    optr[i] = application[(size_t)mapping[(size_t)i]];
  return out;
}

// Split mapped triangles into genuine triangles and degenerate edges. Only the
// ``clean_edges == false`` behaviour used by replay.py is required; the
// deduplicating branch is kept for surface parity.
static nb::tuple clean_triangles_and_edges(InArray<int32_t, 2> mapped_triangles,
                                           bool clean_edges) {
  size_t N = mapped_triangles.shape(0);
  const int32_t *mt = (N > 0) ? mapped_triangles.data() : nullptr;

  int32_t *edges = (int32_t *)std::malloc(((N ? N : 1)) * 2 * sizeof(int32_t));
  int32_t *tris = (int32_t *)std::malloc(((N ? N : 1)) * 3 * sizeof(int32_t));
  if (!edges || !tris) {
    std::free(edges);
    std::free(tris);
    throw std::bad_alloc();
  }

  size_t n_edges = 0;
  size_t n_triangles = 0;
  for (size_t i = 0; i < N; ++i) {
    int32_t j = mt[3 * i + 0];
    int32_t k = mt[3 * i + 1];
    int32_t l = mt[3 * i + 2];
    if (j != k && j != l && k != l) {
      tris[3 * n_triangles + 0] = j;
      tris[3 * n_triangles + 1] = k;
      tris[3 * n_triangles + 2] = l;
      ++n_triangles;
    } else if (j != k) {
      edges[2 * n_edges + 0] = j;
      edges[2 * n_edges + 1] = k;
      ++n_edges;
    } else if (j != l) {
      edges[2 * n_edges + 0] = j;
      edges[2 * n_edges + 1] = l;
      ++n_edges;
    } else if (l != k) {
      edges[2 * n_edges + 0] = l;
      edges[2 * n_edges + 1] = k;
      ++n_edges;
    }
  }

  (void)clean_edges; // dedup branch unused by replay.py; kept for surface parity

  // Wrap the owned buffers as (n, 2) and (n, 3) arrays. The buffers were
  // allocated with a one-element floor, so data() is never null even when a
  // dimension is zero (nanobind rejects null data).
  size_t es[2] = {n_edges, 2};
  nb::capsule eowner(edges, [](void *p) noexcept { std::free(p); });
  NDArray<int32_t, 2> edges2(edges, 2, es, eowner);

  size_t ts[2] = {n_triangles, 3};
  nb::capsule towner(tris, [](void *p) noexcept { std::free(p); });
  NDArray<int32_t, 2> tris2(tris, 2, ts, towner);

  return nb::make_tuple(edges2, tris2);
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

NB_MODULE(_replay, m) {
  m.def("load_int32", &load_int32, "n_points"_a, "n_faces"_a, "n_collapses"_a,
        "points"_a, "faces"_a, "collapses"_a);
  m.def("load_int64", &load_int64, "n_points"_a, "n_faces"_a, "n_collapses"_a,
        "points"_a, "faces"_a, "collapses"_a);
  m.def("load_from_vtk", &load_from_vtk, "n_points"_a, "points"_a, "faces"_a,
        "n_faces"_a);
  m.def("replay", &replay);
  m.def("save_obj", &save_obj, "filename"_a);
  m.def("read", &read_obj, "filename"_a);
  m.def("return_points", &return_points);
  m.def("return_triangles", &return_triangles);
  m.def("return_collapses", &return_collapses);
  m.def("return_faces_int32_no_padding", &return_faces_int32_no_padding);
  m.def("return_faces_int32", &return_faces_int32);
  m.def("return_faces_int64", &return_faces_int64);
  m.def("compute_indice_mapping", &compute_indice_mapping, "collapses"_a,
        "n_points"_a);
  m.def("clean_triangles_and_edges", &clean_triangles_and_edges,
        "mapped_triangles"_a, "clean_edges"_a = false);
  m.def("n_points", &Replay::n_points);
  m.def("n_triangles", &Replay::n_triangles);
  m.def("n_collapses", &Replay::n_collapses);
}
