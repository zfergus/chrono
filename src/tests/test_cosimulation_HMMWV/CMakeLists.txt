# Co-simulation full vehicle (HMMWV)
# ------------------------------------------------------------------------------

if(NOT ENABLE_MODULE_VEHICLE)
  return()
endif()

if(NOT ENABLE_MODULE_FEA OR NOT ENABLE_MODULE_MKL)
  return()
endif()

if(NOT ENABLE_MODULE_PARALLEL)
  return()
endif()  

# ------------------------------------------------------------------------------

include_directories(${CH_MKL_INCLUDES})
include_directories(${CH_PARALLEL_INCLUDES})
include_directories(${MPI_CXX_INCLUDE_PATH})

set(COMPILER_FLAGS "${CH_CXX_FLAGS} ${CH_PARALLEL_CXX_FLAGS}")
set(LINKER_FLAGS "${CH_LINKERFLAG_EXE}")

set(LIBRARIES
    ChronoEngine
    ChronoEngine_vehicle
    ChronoModels_vehicle
    ChronoEngine_fea
    ChronoEngine_mkl
    ChronoEngine_parallel
)

if(ENABLE_MODULE_OPENGL)
    include_directories(${CH_OPENGL_INCLUDES})
    set(LIBRARIES ${LIBRARIES} ChronoEngine_opengl)
    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
       set(LINKER_FLAGS  "${LINKER_FLAGS} -framework IOKit -framework Cocoa -framework OpenGL")
    endif()
endif()

if(ENABLE_MODULE_DISTRIBUTED)
    include_directories(${CH_DISTRIBUTED_INCLUDES})
    set(LIBRARIES ${LIBRARIES} ChronoEngine_distributed)
    set(COMPILER_FLAGS "${COMPILER_FLAGS} ${CH_DISTRIBUTED_CXX_FLAGS}")
    set(LINKER_FLAGS "${LINKER_FLAGS} ${CH_DISTRIBUTED_LINK_FLAGS}")
endif()

# ------------------------------------------------------------------------------

set(PROGRAMS test_cosimulation_HMMWV)

set(TEST_FILES 
    BaseNode.h
    BaseNode.cpp
    VehicleNode.h
    VehicleNode.cpp
    TireNode.h
    TireNode.cpp
    TerrainNode.h
    TerrainNode.cpp
    )

if(ENABLE_MODULE_DISTRIBUTED)
    set(PROGRAMS "${PROGRAMS}" test_cosimulation_HMMWV_DISTR)
    set(TEST_FILES "${TEST_FILES}" TerrainNodeDistr.h TerrainNodeDistr.cpp)
endif()

# ------------------------------------------------------------------------------

message(STATUS "Test programs for CO-SIMULATION...")

foreach(PROGRAM ${PROGRAMS})
    message(STATUS "...add ${PROGRAM}")
    add_executable(${PROGRAM} ${PROGRAM}.cpp ${TEST_FILES})    
    source_group("" FILES ${PROGRAM}.cpp ${TEST_FILES})
    set_target_properties(${PROGRAM} PROPERTIES 
                          COMPILE_FLAGS "${COMPILER_FLAGS}"
                          LINK_FLAGS "${LINKER_FLAGS}")
    target_link_libraries(${PROGRAM} ${LIBRARIES})
endforeach()