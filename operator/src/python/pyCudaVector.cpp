#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FFT.h"

namespace py = pybind11;

using namespace SEP;

PYBIND11_MODULE(pyCudaOperator, clsOps) {
  py::class_<cuFFT2d, std::shared_ptr<cuFFT2d>>(clsOps, "cuFFT2d")
      .def(py::init<std::shared_ptr<hypercube>&>(),
          "Initialize cuFFT2d")

      .def("forward",
            (void (cuFFT2d::*)(bool, std::shared_ptr<complex4DReg>&, std::shared_ptr<complex4DReg>&)) &
            cuFFT2d::forward,
            "Forward operator of cuFFT2d")

      .def("adjoint",
            (void (cuFFT2d::*)(bool, std::shared_ptr<complex4DReg>&, std::shared_ptr<complex4DReg>&)) &
            cuFFT2d::adjoint,
            "Adjoint operator of cuFFT2d");

}

