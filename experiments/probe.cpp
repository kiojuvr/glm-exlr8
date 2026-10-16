// Original M1 diagnostic only: no writes, ownership, exports, or implicit eval.
#include <cstdint>
#include <mlx/array.h>
#include <nanobind/nanobind.h>
namespace nb = nanobind;
NB_MODULE(m1_probe, m) {
  m.def("backing", [](const mlx::core::array& a) {
    // Caller must evaluate first. ptr() is the allocator's opaque
    // buffer handle; unlike data()/raw_ptr(), this does not request CPU access.
    return nb::make_tuple(reinterpret_cast<std::uintptr_t>(a.buffer().ptr()),
                          a.buffer_size(), a.offset());
  });
}
