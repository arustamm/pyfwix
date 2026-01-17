#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "FFT.h"
#include "Spline4D.h"
#include "complex_vector.h"
#include <cuda_runtime.h>

namespace py = pybind11;

using namespace SEP;

void init_operator(py::module_ &m) {
      py::class_<cuFFT2d, std::shared_ptr<cuFFT2d>>(m, "cuFFT2d")
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


      py::class_<Spline4D, std::shared_ptr<Spline4D>>(m, "Spline4D")
      .def(py::init<std::shared_ptr<complex4DReg>, std::shared_ptr<complex4DReg>, 
            float, float, std::vector<float>>(),
          "Initialize Spline4D")

      .def("forward",
            (void (Spline4D::*)(bool, std::shared_ptr<complex4DReg>&, std::shared_ptr<complex4DReg>&)) &
            Spline4D::forward,
            "Forward operator of Spline4D")

      .def("adjoint",
            (void (Spline4D::*)(bool, std::shared_ptr<complex4DReg>&, std::shared_ptr<complex4DReg>&)) &
            Spline4D::adjoint,
            "Adjoint operator of Spline4D");

}

