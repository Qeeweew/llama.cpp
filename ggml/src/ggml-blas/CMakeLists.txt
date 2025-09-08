if (GGML_STATIC)
    set(BLA_STATIC ON)
endif()
#if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.22)
#    set(BLA_SIZEOF_INTEGER 8)
#endif()

ggml_add_backend_library(ggml-blas
                         ggml-blas.cpp
                         tiny-blas.cpp
                        )

target_compile_options(ggml-blas PRIVATE "-march=native")
target_include_directories(ggml-blas PRIVATE "../ggml-cpu")

find_package(OpenMP)
if (OpenMP_FOUND)
    set(GGML_OPENMP_ENABLED "ON" CACHE INTERNAL "")
    target_compile_definitions(ggml-blas PRIVATE GGML_USE_OPENMP)

    target_link_libraries(ggml-blas PRIVATE OpenMP::OpenMP_C OpenMP::OpenMP_CXX)
else()
    set(GGML_OPENMP_ENABLED "OFF" CACHE INTERNAL "")
    message(WARNING "OpenMP not found")
endif()