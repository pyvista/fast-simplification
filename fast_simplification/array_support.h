#ifndef ARRAY_SUPPORT_HEADER_H
#define ARRAY_SUPPORT_HEADER_H

#include <array>
#include <cstdlib>
#include <new>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace nb = nanobind;

// C-contiguous, NumPy-shaped, fixed-rank ndarray alias returned to
// Python. Buffers are malloc-allocated in the binding sources and wrapped
// with a capsule whose deleter calls free, so the arrays own their data
// and remain valid after the call returns.
template <typename T, size_t N>
using NDArray = nb::ndarray<nb::numpy, T, nb::ndim<N>, nb::c_contig>;

// C-contiguous ndarray accepted as input from NumPy without a copy.
template <typename T, size_t N>
using InArray = nb::ndarray<T, nb::ndim<N>, nb::c_contig, nb::device::cpu>;

// Allocate an owned, malloc-backed ndarray of the given shape. The caller
// fills it through data(). nanobind rejects a null data pointer even for
// zero-element arrays, so a minimum one-element allocation is used when the
// shape has a zero extent.
template <typename T, size_t N>
static NDArray<T, N> MakeNDArray(std::array<int, N> shape) {
  size_t total = 1;
  for (size_t i = 0; i < N; ++i)
    total *= (size_t)shape[i];
  size_t alloc = total ? total : 1;
  T *data = (T *)std::malloc(alloc * sizeof(T));
  if (!data)
    throw std::bad_alloc();
  size_t shape_[N];
  for (size_t i = 0; i < N; ++i)
    shape_[i] = (size_t)shape[i];
  nb::capsule owner(data, [](void *p) noexcept { std::free(p); });
  return NDArray<T, N>(data, N, shape_, owner);
}

// Wrap an already malloc-allocated flat buffer as an owned 1-D ndarray of
// ``len`` elements. Used for the VTK-formatted face buffers, where the
// allocation is sized for the worst case and only the populated prefix is
// exposed. A null buffer (len == 0 path) is replaced with a one-element
// allocation to satisfy nanobind.
template <typename T>
static NDArray<T, 1> WrapFlat(T *buf, size_t len) {
  if (buf == nullptr) {
    buf = (T *)std::malloc(sizeof(T));
    if (!buf)
      throw std::bad_alloc();
  }
  size_t shape_[1] = {len};
  nb::capsule owner(buf, [](void *p) noexcept { std::free(p); });
  return NDArray<T, 1>(buf, 1, shape_, owner);
}

#endif // ARRAY_SUPPORT_HEADER_H
