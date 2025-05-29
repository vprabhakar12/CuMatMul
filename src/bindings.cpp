#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "matmul.h"

namespace py = pybind11;

void wrapper_naive(py::array_t<float> A,
                   py::array_t<float> B,
                   py::array_t<float> C,
                   int N) {
    launch_naive(A.mutable_data(), B.mutable_data(), C.mutable_data(), N);
}

void wrapper_tiled(py::array_t<float> A,
                   py::array_t<float> B,
                   py::array_t<float> C,
                   int N) {
    launch_tiled(A.mutable_data(), B.mutable_data(), C.mutable_data(), N);
}

PYBIND11_MODULE(cumatmul, m) {
    m.def("naive", &wrapper_naive, "Run naive CUDA matrix multiplication");
    m.def("tiled", &wrapper_tiled, "Run tiled CUDA matrix multiplication");
}
