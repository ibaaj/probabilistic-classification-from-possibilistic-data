from pathlib import Path

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

HERE = Path(__file__).resolve().parent

ext_modules = [
    Pybind11Extension(
        "_dykstra_cpp",
        [str(HERE / "_dykstra_cpp.cpp")],
        cxx_std=17,
        extra_compile_args=["-O3"],
    )
]

setup(
    name="dykstra-cpp",
    version="0.1.0",
    description="C++ pybind11 hot loop for klbox.dykstra",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)