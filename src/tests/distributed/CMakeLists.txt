# Miscellaneous tests with Chrono::Distributed
# ------------------------------------------------------------------------------

if(NOT ENABLE_MODULE_PARALLEL OR NOT ENABLE_MODULE_DISTRIBUTED)
  return()
endif()

# ------------------------------------------------------------------------------
# List of all tests
# ------------------------------------------------------------------------------

set(TESTS
    test_DISTR_geometry
    test_DISTR_hopper
    test_DISTR_slopeplane
    test_DISTR_setstate
    test_DISTR_getforces
    test_DISTR_addbody
    test_DISTR_scaling
)

set(TESTS_OPENGL
    test_DISTR_boundary_par
    test_DISTR_slopeplane_par
    test_DISTR_hopper_par
    test_PAR_hopper
)

# ------------------------------------------------------------------------------

include_directories(${CH_PARALLEL_INCLUDES})
include_directories(${CH_DISTRIBUTED_INCLUDES})

set(COMPILER_FLAGS "${CH_CXX_FLAGS} ${CH_DISTRIBUTED_CXX_FLAGS}")

set(LINKER_FLAGS "${CH_LINKERFLAG_EXE} ${CH_DISTRIBUTED_LINK_FLAGS}")

set(LIBRARIES
    ChronoEngine
    ChronoEngine_parallel
    ChronoEngine_distributed
)

if(ENABLE_MODULE_OPENGL)
    set(ALL_TESTS ${TESTS} ${TESTS_OPENGL})
    include_directories(${CH_OPENGL_INCLUDES})
    set(LIBRARIES ${LIBRARIES} ChronoEngine_opengl)
endif()

# ------------------------------------------------------------------------------
# Add all executables
# ------------------------------------------------------------------------------

message(STATUS "Test programs for DISTRIBUTED module...")

foreach(PROGRAM ${ALL_TESTS})
    message(STATUS "...add ${PROGRAM}")

    add_executable(${PROGRAM}  "${PROGRAM}.cpp")
    source_group(""  FILES "${PROGRAM}.cpp")

    set_target_properties(${PROGRAM} PROPERTIES
        COMPILE_FLAGS "${COMPILER_FLAGS}"
        LINK_FLAGS "${LINKER_FLAGS}")

    target_link_libraries(${PROGRAM} ${LIBRARIES})
endforeach()
