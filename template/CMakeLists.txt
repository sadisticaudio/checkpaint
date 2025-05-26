cmake_minimum_required(VERSION 3.15...3.27)

set(ptVer $ENV{ptVer} CACHE STRING "torch version")
set(cudaVer $ENV{cudaVer} CACHE STRING "cuda version")
set(cudnnVer $ENV{cudnnVer} CACHE STRING "cudnn version")
set(pyVer $ENV{pyVer} CACHE STRING "python version")
string(REPLACE "." "" pyName ${pyVer})

set(CUDA_HOME /media/frye/sda5/CODE/cudas/${cudaVer})
set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_HOME})

set(pythonPrefix "/opt/python/cp${pyName}-cp${pyName}")
set(Python_ROOT_DIR "${pythonPrefix}")
set(Python_EXECUTABLE "${pythonPrefix}/bin/python")
set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})

# Scikit-build-core sets these values for you, or you can just hard-code the
# name and version.
project(
  ${SKBUILD_PROJECT_NAME}
  VERSION ${SKBUILD_PROJECT_VERSION}
  LANGUAGES CXX)

set(CODE /media/frye/sda5/CODE)
set(theCudaRoot ${CODE}/cudas/${cudaVer})
set(theTorchRoot ${CODE}/torchs/${ptVer}/torch)
set(EINOPS_HOME ${CODE}/einops-cpp)
message("Torch path is \"${theTorchRoot}\"... " "CUDA path is \"${theCudaRoot}\"... ")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_VERBOSE_MAKEFILE on)
set(VERBOSE on)
set(_GLIBCXX_HAVE_STACKTRACE 1)
set(_GLIBCXX_USE_CXX11_ABI 0)
set(CMAKE_C_COMPILER /opt/rh/devtoolset-10/root/bin/gcc)
set(CMAKE_CXX_COMPILER /opt/rh/devtoolset-10/root/bin/g++)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v -allow-unsupported-compiler")
set(CXX_STANDARD_REQUIRED ON)

set(CUDA_TOOLKIT_ROOT_DIR ${theCudaRoot})
set(CUDA_HOME ${CUDA_TOOLKIT_ROOT_DIR})
set(CUDNN_ROOT_DIR ${CODE}/cudnns/${cudnnVer}_${cudaVer})
set(CUDNN_INCLUDE_PATH ${CUDNN_ROOT_DIR}/include)
set(CUDNN_LIBRARY_PATH ${CUDNN_ROOT_DIR}/lib)
set(CUDACXX ${CUDA_HOME}/bin/nvcc)
set(CMAKE_CUDA_COMPILER ${CUDACXX})
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CAFFE2_USE_CUDNN 1)
set(kineto_LIBRARY ${theTorchRoot}/../build/lib/libkineto.a)
set(Torch_INCLUDE_DIR "${theTorchRoot}/include;${theTorchRoot}/include/torch/csrc/api/include")
set(CMAKE_PREFIX_PATH ${theOriginalCMAKE_PREFIX_PATH};${theTorchRoot})
set(Torch_DIR ${theTorchRoot}/share/cmake/Torch/)
#  -faligned-new
# -L/opt/rh/devtoolset-10/root/usr/lib/gcc/x86_64-redhat-linux/10/32 -fsanitize=address -static-libasan -fsanitize=undefined -static-libubsan
# -fopenmp 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -v -std=c++17 -static-libstdc++ -fconcepts -fpic -Wno-deprecated-declarations")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -O0 -fsanitize=address -static-libasan -fpermissive -ggdb3 -rdynamic -fno-eliminate-unused-debug-symbols -fno-eliminate-unused-debug-types -fsanitize=float-divide-by-zero -Wuninitialized  -Wno-variadic-macros")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O2 -fvisibility=hidden -fvisibility-inlines-hidden -fno-omit-frame-pointer -pthread -fopenmp")

set(Python_LIBRARY "${pythonPrefix}/lib/python${pyVer}")
set(Python_INCLUDE_DIR "${pythonPrefix}/include/python${pyVer}")
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
set(pybind11_DIR /opt/python/cp${pyName}-cp${pyName}/lib/python${pyVer}/site-packages/pybind11/share/cmake/pybind11)
find_package(pybind11 CONFIG REQUIRED)

set(CMAKE_INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib:$ORIGIN/../../lib:$ORIGIN/../../../lib:$ORIGIN/../torch/lib:$ORIGIN/../../torch/lib:$ORIGIN/../../site-packages:$ORIGIN/../../../../lib:/lib")

pybind11_add_module(_core src/ccc.cpp src/Chex.cpp)
install(TARGETS _core DESTINATION checkpaint)
set_target_properties(_core PROPERTIES
    PREFIX ""
    OUTPUT_NAME _core
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
target_include_directories(_core PRIVATE
  ${Torch_INCLUDE_DIR}
  # /opt/rh/devtoolset-10/root/usr/lib/gcc/x86_64-redhat-linux/10/include/omp.h
  ${EINOPS_HOME}/include
  ${Python_INCLUDE_DIR}
  ${CUDA_HOME}/targets/x86_64-linux/include
)

target_link_libraries(_core PRIVATE
  # "-Wl,--whole-archive /lib/libasan.a -Wl,--no-whole-archive"
  # -L/opt/rh/devtoolset-10/root/usr/lib/gcc/x86_64-redhat-linux/10 -lasan
  -lstdc++
  # -L/opt/rh/devtoolset-10/root/usr/lib/gcc/x86_64-redhat-linux/10 -lgomp
  -L${theTorchRoot}/../build/lib -lc10 -lc10_cuda -ltorch -ltorch_cpu -ltorch_cuda -ltorch_python # -ltbb
  -L${theCudaRoot}/lib64/stubs -lcuda -L${theCudaRoot}/lib64 -lnvrtc -lnvToolsExt -lcudart
  pybind11::headers
)