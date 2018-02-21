// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2016 projectchrono.org
// All right reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Nic Olsen
// =============================================================================

#pragma once

#include <mpi.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include "chrono/physics/ChBody.h"

#include "chrono_distributed/ChApiDistributed.h"
#include "chrono_distributed/ChDistributedDataManager.h"
#include "chrono_distributed/comm/ChCommDistributed.h"
#include "chrono_distributed/other_types.h"
#include "chrono_distributed/physics/ChDomainDistributed.h"

#include "chrono_parallel/ChDataManager.h"
#include "chrono_parallel/physics/ChSystemParallel.h"

namespace chrono {

class ChDomainDistributed;
class ChCommDistributed;
class ChDataManagerDistr;

/// This is the main user interface for Chrono::Distributed
/// Add bodies and set all settings through the system.
/// The simulation runs on all ranks given in the world parameter.
class CH_DISTR_API ChSystemDistributed : public ChSystemParallelSMC {
    friend class ChCommDistributed;

  public:
    /// world specifies on which MPI_Comm the simulation should run.
    ChSystemDistributed(MPI_Comm world, double ghost_layer, unsigned int max_objects);
    virtual ~ChSystemDistributed();

    /// Returns the number of MPI ranks the system is using.
    int GetNumRanks() { return num_ranks; }

    /// Returns the number of the rank the system is running on.
    int GetMyRank() { return my_rank; }

    int GetMasterRank() { return master_rank; }

    /// Returns the distance into the neighboring sub-domain that is considered shared.
    double GetGhostLayer() { return ghost_layer; }

    /// A running count of the number of global bodies for
    /// identification purposes
    int GetNumBodiesGlobal() { return num_bodies_global; }

    /// Should be called every add to the GLOBAL system i.e. regardless of whether
    /// the body in question is retained by the system on a particular rank, this
    /// should be called after an AddBody call.
    int IncrementGID() { return num_bodies_global++; }

    /// Returns true if pos is within this rank's sub-domain.
    bool InSub(ChVector<double> pos);

    /// Add a body to the system. Should be called on every rank for every body.
    /// Classifies the body and decides whether or not to keep it on this rank.
    void AddBody(std::shared_ptr<ChBody> newbody) override;

    /// Adds a body to the system regardless of its location. Should only be called
    /// if the caller needs a body added to the entire system. NOTE: A body crossing
    /// multiple sub-domains will not be correctly advanced.
    void AddBodyTrust(std::shared_ptr<ChBody> newbody);

    /// Removes a body from the simulation based on the ID of the body (not based on
    /// object comparison between ChBodys). Should be called on all ranks to ensure
    /// that the correct body is found and removed where it exists.
    void RemoveBody(std::shared_ptr<ChBody> body) override;

    /// Wraps the super-class Integrate_Y call and introduces a call that carries
    /// out all inter-rank communication.
    virtual bool Integrate_Y() override;

    /// Wraps super-class UpdateRigidBodies and adds a gid update.
    virtual void UpdateRigidBodies() override;

    /// Internal call for removing deactivating a body.
    /// Should not be called by the user.
    void RemoveBodyExchange(int index);

    /// Returns the ChDomainDistributed object associated with the system.
    ChDomainDistributed* GetDomain() { return domain; }

    /// Returns the ChCommDistributed object associated with the system.
    ChCommDistributed* GetComm() { return comm; }

    /// Prints msg to the user and ends execution with an MPI abort.
    void ErrorAbort(std::string msg);

    /// Prints out all valid body data. Should only be used for debugging.
    void PrintBodyStatus();

    /// Prints out all valid shape data. Should only be used for debugging.
    void PrintShapeData();

    /// Prints measures for computing efficiency.
    void PrintEfficiency();

    /// Returns the MPI communicator being used by the system.
    MPI_Comm GetMPIWorld() { return world; }

    /// Central data storages for chrono_distributed. Adds scaffolding data
    /// around ChDataManager used by chrono_parallel in order to maintain
    /// a consistent and correct view of all valid data.
    ChDistributedDataManager* ddm;

    /// Name of the node being run on.
    char node_name[50];

    /// Debugging function
    double GetLowestZ(uint* gid);

    /// Returns the highest z coordinate in the system
    double GetHighestZ();

    /// Checks for consistency in IDs in the system. Should only be used
    /// for debugging.
    void CheckIds();

    void AddBodyAllRanks(std::shared_ptr<ChBody> body);

    /// Removes all bodies below the given height - initial implementation of a
    /// deactivating boundary condition.
    void RemoveBodiesBelow(double z);

    /// Checks structures added by chrono_distributed. Prints ERROR messages at
    /// inconsistencies.
    void SanityCheck();

    /// Stores all data needed to fully update the state of a body
    typedef struct BodyState {
        BodyState(ChVector<>& p, ChQuaternion<>& r, ChVector<>& p_dt, ChQuaternion<>& r_dt)
            : pos(p), rot(r), pos_dt(p_dt), rot_dt(r_dt){};

        ChVector<> pos;
        ChQuaternion<> rot;
        ChVector<> pos_dt;
        ChQuaternion<> rot_dt;
    } BodyState;

    /// Updates the states of all bodies listed in the gids parameter
    /// Must be called on all system ranks and inputs must be complete and
    /// valid on each rank.
    /// NOTE: The change in position should be small in comparison to the ghost
    /// layer of this system.
    /// NOTE: The new states will reach the data_manager at the beginning of the
    /// next time step.
    void SetBodyStates(std::vector<uint> gids, std::vector<BodyState> states);
    void SetBodyState(uint gid, BodyState state);

    /// Updates each sphere shape associated with bodies with global ids gids.
    /// shape_idx identifies the index of the shape within its body's collisionsystem
    /// model.
    /// Must be called on all system ranks and inputs must be complete and
    /// valid on each rank.
    void SetSphereShapes(std::vector<uint> gids, std::vector<int> shape_idx, std::vector<double> radii);
    void SetSphereShape(uint gid, int shape_idx, double radius);

    /// Structure of vertex data for a triangle in the bodies existing local frame
    typedef struct TriData {
        ChVector<double> v1;
        ChVector<double> v2;
        ChVector<double> v3;
    } TriData;

    /// Updates triangle shapes associated with bodies identified by gids.
    /// shape_idx identifies the index of the shape within its body's collisionsystem
    /// model.
    /// Must be called on all system ranks and inputs must be complete and
    /// valid on each rank.
    void SetTriangleShapes(std::vector<uint> gids, std::vector<int> shape_idx, std::vector<TriData> new_shapes);
    void SetTriangleShape(uint gid, int shape_idx, TriData new_shape);

    /// Structure of force data used internally for MPI sending contact forces.
    typedef struct internal_force {
        uint gid;
        double force[3];
    } internal_force;

    /// Returns a vector of pairs of gid and corresponding contact forces.
    /// Must be called on all system ranks and inputs must be complete and valid
    /// on each rank.
    /// Returns a pair with gid == UINT_MAX if there are the gid if no contact
    /// force is found.
    std::vector<std::pair<uint, ChVector<>>> GetBodyContactForces(std::vector<uint> gids);
    std::pair<uint, ChVector<>> GetBodyContactForce(uint gid);

  protected:
    /// Number of MPI ranks
    int num_ranks;

    /// MPI rank
    int my_rank;

    /// Master MPI rank
    int master_rank;

    /// Length into the neighboring sub-domain which is considered shared.
    double ghost_layer;

    /// Number of bodies in the whole global simulation. Important for maintaining
    /// unique global IDs
    unsigned int num_bodies_global;

    /// Communicator of MPI ranks for the simulation
    MPI_Comm world;

    /// Class for domain decomposition
    ChDomainDistributed* domain;

    /// Class for MPI communication
    ChCommDistributed* comm;

    /// Internal function for adding a body from communication. Should not be
    /// called by the user.
    void AddBodyExchange(std::shared_ptr<ChBody> newbody, distributed::COMM_STATUS status);

    // Co-simulation
    MPI_Datatype InternalForceType;
};

} /* namespace chrono */
