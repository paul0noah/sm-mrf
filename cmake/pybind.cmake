# Try to find pybind11 locally first, fall back to FetchContent
execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE PYBIND11_PYTHON_RESULT
)
if(PYBIND11_PYTHON_RESULT EQUAL 0)
    list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
endif()
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    message("Found local pybind11 installation")
else()
    message("Retrieving pybind11 from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        pybind
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v3.0.0
    )
    FetchContent_MakeAvailable(pybind)
endif()
