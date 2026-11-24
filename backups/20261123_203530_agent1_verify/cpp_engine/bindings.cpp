#include "diffusion.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(simulation_engine, m) {
    m.doc() = "Signed competitive Maki-Thompson diffusion engine (read-only graph, persistent)";

    py::class_<SimResult>(m, "SimResult")
        .def_readonly("spread_A", &SimResult::spread_A)
        .def_readonly("spread_B", &SimResult::spread_B)
        .def_readonly("ignorant", &SimResult::ignorant)
        .def_readonly("rounds", &SimResult::rounds)
        .def("__repr__", [](const SimResult& r) {
            return "SimResult(spread_A=" + std::to_string(r.spread_A) +
                   ", spread_B=" + std::to_string(r.spread_B) +
                   ", ignorant=" + std::to_string(r.ignorant) +
                   ", rounds=" + std::to_string(r.rounds) + ")";
        });

    py::class_<DiffusionEngine>(m, "DiffusionEngine")
        .def(py::init<const std::vector<int>&,
                      const std::vector<std::vector<int>>&,
                      const std::vector<std::vector<double>>&,
                      const std::vector<std::vector<int>>&,
                      const std::vector<std::vector<int>>&>(),
             py::arg("node_ids"), py::arg("adj"), py::arg("probs"),
             py::arg("polarity"), py::arg("strength"))
        .def("simulate", &DiffusionEngine::simulate,
             py::arg("seeds_A"), py::arg("seeds_B"), py::arg("seed"))
        .def("simulate_many", &DiffusionEngine::simulate_many,
             py::arg("seeds_A"), py::arg("seeds_B"), py::arg("seeds"),
             py::arg("n_threads") = 0)
        .def("simulate_batch", &DiffusionEngine::simulate_batch,
             py::arg("seeds_A"), py::arg("seeds_B"), py::arg("seeds"),
             py::arg("n_threads") = 0)
        .def_property_readonly("num_nodes", &DiffusionEngine::num_nodes)
        .def_property_readonly("num_edges", &DiffusionEngine::num_edges)
        .def_property_readonly("node_ids", &DiffusionEngine::node_ids);

    m.def("simulate", &simulate,
          py::arg("node_ids"), py::arg("adj"), py::arg("probs"), py::arg("polarity"),
          py::arg("strength"), py::arg("seeds_A"), py::arg("seeds_B"), py::arg("seed"),
          "One-shot simulation (builds a transient engine; prefer DiffusionEngine "
          "for repeated runs).");

    m.def("openmp_enabled", &openmp_enabled,
          "True if the extension was compiled with OpenMP support.");
}
