# Miscellaneous tests with Chrono::Parallel
# ------------------------------------------------------------------------------

if(NOT ENABLE_MODULE_PARALLEL)
  return()
endif()  

if(NOT ENABLE_MODULE_OPENGL)
  return()
endif()  

# ------------------------------------------------------------------------------
# List of all tests
# ------------------------------------------------------------------------------

set(TESTS
    test_parallel_triangle
)

# ------------------------------------------------------------------------------

include_directories(${CH_PARALLEL_INCLUDES})
include_directories(${CH_OPENGL_INCLUDES})

set(COMPILER_FLAGS "${CH_CXX_FLAGS}")

set(LINKER_FLAGS "${CH_LINKERFLAG_EXE}")

set(LIBRARIES
    ChronoEngine
    ChronoEngine_parallel
    ChronoEngine_opengl
)

# ------------------------------------------------------------------------------
# Add all executables
# ------------------------------------------------------------------------------

message(STATUS "Test programs for PARALLEL module...")

foreach(PROGRAM ${TESTS})
    message(STATUS "...add ${PROGRAM}")

    add_executable(${PROGRAM}  "${PROGRAM}.cpp")
    source_group(""  FILES "${PROGRAM}.cpp")

    set_target_properties(${PROGRAM} PROPERTIES
        COMPILE_FLAGS "${COMPILER_FLAGS}"
        LINK_FLAGS "${LINKER_FLAGS}")

    target_link_libraries(${PROGRAM} ${LIBRARIES})
endforeach()
