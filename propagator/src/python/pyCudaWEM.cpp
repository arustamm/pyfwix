#include <pybind11/chrono.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "PhaseShift.h"
#include "RefSampler.h"
#include "OneStep.h"
#include "Injection.h"
#include "OneWay.h"
#include "Propagator.h"
#include "StreamingPropagator.h"
#include "ExtendedBorn.h"

namespace py = pybind11;

using namespace SEP;

void init_propagator(py::module_ &m) {

py::class_<StreamingPropagator, std::shared_ptr<StreamingPropagator>>(m, "StreamingPropagator")
.def(py::init<const std::shared_ptr<hypercube>&,
            const std::shared_ptr<hypercube>&,
            std::shared_ptr<hypercube>,
            std::shared_ptr<complex2DReg>,
            const std::vector<float>&,
            const std::vector<float>&,
            const std::vector<float>&,
            const std::vector<int>&,
            const std::vector<float>&,
            const std::vector<float>&,
            const std::vector<float>&,
            const std::vector<int>&,
            std::shared_ptr<paramObj>, 
            const std::vector<int>&>(), "Initialize StreamingPropagator")

.def("forward",
    (void (StreamingPropagator::*)(bool, std::vector<std::shared_ptr<complex4DReg>>&, std::shared_ptr<complex2DReg>&)) &
    StreamingPropagator::forward,
    "Nonlinear forward operator of StreamingPropagator");

py::class_<Propagator, std::shared_ptr<Propagator>>(m, "Propagator")
    .def(py::init<const std::shared_ptr<hypercube>&,
                const std::shared_ptr<hypercube>&,
                std::shared_ptr<hypercube>,
                std::shared_ptr<complex2DReg>,
                const std::vector<float>&,
                const std::vector<float>&,
                const std::vector<float>&,
                const std::vector<int>&,
                const std::vector<float>&,
                const std::vector<float>&,
                const std::vector<float>&,
                const std::vector<int>&,
                std::shared_ptr<paramObj>>(), "Initialize Propagator")
    .def("forward",
        (void (Propagator::*)(bool, std::vector<std::shared_ptr<complex4DReg>>&, std::shared_ptr<complex2DReg>&)) &
        Propagator::forward,
        "Nonlinear forward operator of Propagator")
        
    .def("get_compression_ratio", [](Propagator &self) {
        return self.get_compression_ratio();
    }, "Get compression ratio of Propagator");

py::class_<ExtendedBorn, std::shared_ptr<ExtendedBorn>>(m, "ExtendedBorn")
    .def(py::init<const std::shared_ptr<hypercube>&,
            const std::shared_ptr<hypercube>&,
            std::vector<std::shared_ptr<complex4DReg>>,
            std::shared_ptr<Propagator>>(), "Initialize Propagator")
    .def("forward",
        (void (ExtendedBorn::*)(bool, std::vector<std::shared_ptr<complex4DReg>>&, std::shared_ptr<complex2DReg>&)) &
        ExtendedBorn::forward,
        "Nonlinear forward operator of ExtendedBorn")
    .def("adjoint",
        (void (ExtendedBorn::*)(bool, std::vector<std::shared_ptr<complex4DReg>>&, std::shared_ptr<complex2DReg>&)) &
        ExtendedBorn::adjoint,
        "Nonlinear adjoint operator of ExtendedBorn");

// py::class_<PhaseShift, std::shared_ptr<PhaseShift>>(m, "PhaseShift")
//     .def(py::init<std::shared_ptr<hypercube>, float, float &>(),
//         "Initialize PhaseShift")

//     .def("forward",
//         (void (PhaseShift::*)(bool, std::shared_ptr<complex4DReg>&, std::shared_ptr<complex4DReg>&)) &
//         PhaseShift::forward,
//         "Forward operator of PhaseShift")

//     .def("adjoint",
//         (void (PhaseShift::*)(bool, std::shared_ptr<complex4DReg>&, std::shared_ptr<complex4DReg>&)) &
//         PhaseShift::adjoint,
//         "Adjoint operator of PhaseShift")

//     .def("set_slow", [](PhaseShift &self, py::array_t<std::complex<float>, py::array::c_style> arr) {
//             auto buf = arr.request();
//             self.set_slow(static_cast<std::complex<float> *>(buf.ptr));
//         });

// py::class_<RefSampler, std::shared_ptr<RefSampler>>(m, "RefSampler")
//     .def(py::init<const std::shared_ptr<complex4DReg>&, std::shared_ptr<paramObj>&>(),
//         "Initialize RefSampler")

//     .def("get_ref_slow", [](RefSampler &self, int iz, int iref) {
//         return py::array_t<std::complex<float>>(
//             {self._nw_}, // shape
//             self.get_ref_slow(iz, iref) // pointer to data
//         );
//     })

//     .def("get_ref_labels", [](RefSampler &self, int iz) {
//         return py::array_t<int>(
//             {self._nw_, self._ny_ + self.pady, self._nx_+self.padx}, // shape
//             self.get_ref_labels(iz) // pointer to data
//         );
//     });

py::class_<PSPI, std::shared_ptr<PSPI>>(m, "PSPI")
    .def(py::init<std::shared_ptr<hypercube>&, std::shared_ptr<complex4DReg>, std::shared_ptr<paramObj>>(),
        "Initialize PSPI")

    .def("forward",
        (void (PSPI::*)(bool, std::shared_ptr<complex4DReg>&, std::shared_ptr<complex4DReg>&)) &
        PSPI::forward,
        "Forward operator of PSPI")

    .def("forward",
        (void (PSPI::*)(std::shared_ptr<complex4DReg>&)) &
        PSPI::forward,
        "Forward operator of PSPI")

    .def("adjoint",
        (void (PSPI::*)(bool, std::shared_ptr<complex4DReg>&, std::shared_ptr<complex4DReg>&)) &
        PSPI::adjoint,
        "Adjoint operator of PSPI")

    .def("set_depth", 
        (void (PSPI::*)(int)) &
        PSPI::set_depth,
        "Set depth of PSPI");

py::class_<NSPS, std::shared_ptr<NSPS>>(m, "NSPS")
    .def(py::init<std::shared_ptr<hypercube>&, std::shared_ptr<complex4DReg>, std::shared_ptr<paramObj>>(),
        "Initialize NSPS")

    .def("forward",
        (void (NSPS::*)(bool, std::shared_ptr<complex4DReg>&, std::shared_ptr<complex4DReg>&)) &
        NSPS::forward,
        "Forward operator of NSPS")

    .def("adjoint",
        (void (NSPS::*)(bool, std::shared_ptr<complex4DReg>&, std::shared_ptr<complex4DReg>&)) &
        NSPS::adjoint,
        "Adjoint operator of NSPS")

    .def("set_depth", 
        (void (NSPS::*)(int)) &
        NSPS::set_depth,
        "Set depth of NSPS");


py::class_<Injection, std::shared_ptr<Injection>>(m, "Injection")
    .def(py::init<std::shared_ptr<hypercube>&, std::shared_ptr<hypercube>&, 
        float&, float&,
        const std::vector<float>&, const std::vector<float>&, const std::vector<float>&, const std::vector<int>&>(),
        "Initialize Injection")

    .def("forward",
        (void (Injection::*)(bool, std::shared_ptr<complex2DReg>&, std::shared_ptr<complex4DReg>&)) &
        Injection::forward,
        "Forward operator of Injection")

    .def("adjoint",
        (void (Injection::*)(bool, std::shared_ptr<complex2DReg>&, std::shared_ptr<complex4DReg>&)) &
        Injection::adjoint,
        "Adjoint operator of Injection")

    .def("set_depth", 
        (void (Injection::*)(int)) &
        Injection::set_depth,
        "Set depth of Injection")

    .def("set_coords", 
        (void (Injection::*)(const std::vector<float>&, const std::vector<float>&, const std::vector<float>&, const std::vector<int>&)) &
        Injection::set_coords,
        "Set coords of Injection");

    

py::class_<Downward, std::shared_ptr<Downward>>(m, "Downward")
    .def(py::init<std::shared_ptr<hypercube>&, std::shared_ptr<complex4DReg>&, std::shared_ptr<paramObj>&>(),
        "Initialize Downward")

    .def("forward",
        (void (Downward::*)(bool, std::shared_ptr<complex4DReg>&, std::shared_ptr<complex4DReg>&)) &
        Downward::forward,
        "Forward operator of Downward")

    .def("adjoint",
        (void (Downward::*)(bool, std::shared_ptr<complex4DReg>&, std::shared_ptr<complex4DReg>&)) &
        Downward::adjoint,
        "Adjoint operator of Downward")

    .def("forward",
        (void (Downward::*)(std::shared_ptr<complex4DReg>&)) &
        Downward::forward,
        "Forward operator of Downward")

    .def("adjoint",
        (void (Downward::*)(std::shared_ptr<complex4DReg>&)) &
        Downward::adjoint,
        "Adjoint operator of Downward");

py::class_<Upward, std::shared_ptr<Upward>>(m, "Upward")
    .def(py::init<std::shared_ptr<hypercube>&, std::shared_ptr<complex4DReg>&, std::shared_ptr<paramObj>&>(),
        "Initialize Upward")

    .def("forward",
        (void (Upward::*)(bool, std::shared_ptr<complex4DReg>&, std::shared_ptr<complex4DReg>&)) &
        Upward::forward,
        "Forward operator of Upward")

    .def("adjoint",
        (void (Upward::*)(bool, std::shared_ptr<complex4DReg>&, std::shared_ptr<complex4DReg>&)) &
        Upward::adjoint,
        "Adjoint operator of Upward");

}

