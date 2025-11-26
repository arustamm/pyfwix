#include <pybind11/pybind11.h>

namespace py = pybind11;

// Forward declarations of the functions above
void init_operator(py::module_ &m);
void init_propagator(py::module_ &m);

// Define the single monolithic module name
PYBIND11_MODULE(pyFWIX, m) {
    m.doc() = "Monolithic FWIX CUDA Module";
    
    init_operator(m);
    init_propagator(m);
}