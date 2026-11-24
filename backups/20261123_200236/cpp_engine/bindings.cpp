#include "diffusion.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

PYBIND11_MODULE(simulation_engine, m) {
    py::class_<SimResult>(m, "SimResult")
        .def_readonly("spread_A", &SimResult::spread_A)
        .def_readonly("spread_B", &SimResult::spread_B)
        .def_readonly("ignorant", &SimResult::ignorant)
        .def_readonly("rounds", &SimResult::rounds);
    m.def("simulate", &simulate,
        py::arg("node_ids"), py::arg("adj"), py::arg("probs"), py::arg("polarity"),
        py::arg("strength"), py::arg("seeds_A"), py::arg("seeds_B"), py::arg("seed"));
}
