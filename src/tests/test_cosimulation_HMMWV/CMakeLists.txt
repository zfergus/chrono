# Co-simulation full vehicle (HMMWV)
# ------------------------------------------------------------------------------

if(NOT ENABLE_MODULE_VEHICLE)
  return()
endif()

if(NOT ENABLE_MODULE_FEA OR NOT ENABLE_MODULE_MKL)
  return()
endif()

if(NOT ENABLE_MODULE_PARALLEL OR NOT ENABLE_MODULE_DISTRIBUTED)
  return()
endif()  

if(NOT ENABLE_MODULE_OPENGL)
  return()
endif()  

# ------------------------------------------------------------------------------

include_directories( ${CH_MKL_INCLUDES} )
include_directories(${CH_PARALLEL_INCLUDES})
include_directories(${CH_DISTRIBUTED_INCLUDES})
include_directories(${MPI_CXX_INCLUDE_PATH})
include_directories(${CH_OPENGL_INCLUDES})

set(COMPILER_FLAGS "${CH_CXX_FLAGS} ${CH_DISTRIBUTED_CXX_FLAGS}")

set(LINKER_FLAGS "${CH_LINKERFLAG_EXE} ${CH_DISTRIBUTED_LINK_FLAGS}")

set(LIBRARIES
    ChronoEngine
    ChronoEngine_vehicle
    ChronoModels_vehicle
    ChronoEngine_fea
    ChronoEngine_mkl
    ChronoEngine_parallel
    ChronoEngine_distributed
    ChronoEngine_opengl
)

# ------------------------------------------------------------------------------

set(PROGRAM test_cosimulation_HMMWV)

set(TEST_FILES 
    test_cosimulation_HMMWV.cpp
    BaseNode.h
    BaseNode.cpp
    VehicleNode.h
    VehicleNode.cpp
    TireNode.h
    TireNode.cpp
    TerrainNode.h
    TerrainNode.cpp
    TerrainNodeDistr.h
    TerrainNodeDistr.cpp
    )

source_group("" FILES ${TEST_FILES})

message(STATUS "...add ${PROGRAM}")

add_executable(${PROGRAM} ${TEST_FILES})

set_target_properties(${PROGRAM} PROPERTIES 
    COMPILE_FLAGS "${COMPILER_FLAGS}"
    LINK_FLAGS "${LINKER_FLAGS}")

target_link_libraries(${PROGRAM} ${LIBRARIES})
