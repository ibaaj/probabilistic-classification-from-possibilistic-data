# klbox

KL-box construction and projection code.

## Main files

- `possibility.py`  
  Possibility ordering and antipignistic reverse mapping.

- `gaps.py`  
  Lower and upper gap construction.

- `linear_system.py`  
  Feasibility system and violation score.

- `constraints.py`  
  Prefix and gap constraints for the Python solver.

- `dykstra.py`  
  Python Dykstra KL projection.

- `dykstra_cpp.py`  
  C++ projection wrapper, including batch mode.

- `_dykstra_cpp.cpp`  
  C++ implementation of the projection loop.

- `setup_cpp.py`  
  Build script for the C++ extension.

- `protocol.py`  
  Projection sweeps and LaTeX reporting helpers.

- `benchmark_cpp_vs_python.py`  
  Benchmark script for Python vs C++.

## Notes

It provides the projection core used by both the synthetic top-k and ChaosNLI pipelines.