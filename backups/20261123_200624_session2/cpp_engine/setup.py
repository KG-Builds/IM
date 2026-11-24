from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

setup(
    name="simulation_engine",
    ext_modules=[Pybind11Extension(
        "simulation_engine",
        ["bindings.cpp", "diffusion.cpp"],
        include_dirs=["."],
        cxx_std=17,
        extra_compile_args=["-O3", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    )],
    cmdclass={"build_ext": build_ext},
)
