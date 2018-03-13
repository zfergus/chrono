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
  public:
    /// Construct a distributed Chrono system using the specified MPI communicator.
    ChSystemDistributed(MPI_Comm communicator, double ghostlayer, unsigned int maxobjects);
    virtual ~ChSystemDistributed();

    /// Return the size of the group associated with the system's intra-communicator.
    int GetCommSize() const { return num_ranks; }

    /// Return the rank of the calling process in the system's intra-communicator.
    int GetCommRank() const { return my_rank; }

    /// Set the calling process as 'master' in the intra-communicator used by this system.
    /// For efficiency, certain functions report information only on this single rank.  This saves
    /// a potentially unnecessary scatter operation (if needed, such an operation should be performed
    /// in user code).  By default this is rank 0 in the system's intra-communicator.
    void SetMaster() { master_rank = my_rank; }
 
    /// Return the rank (in the system's intra-communicator) of the process marked as 'master'.
    /// Certain functions return information only on this process.
    int GetMasterRank() const { return master_rank; }

    /// Return true if the calling process is the one marked as 'master'.
    bool OnMaster() const { return my_rank == master_rank; }

    /// Return the distance into the neighboring sub-domain that is considered shared.
    double GetGhostLayer() const { return ghost_layer; }

    /// Return the current global number of bodies in the system.
    int GetNumBodiesGlobal() const { return num_bodies_global; }

    /// Increment the global number of bodies. 
    /// This function should be called *on all ranks* after a call to AddBodyTrust.
    /// Do not call this function after AddBody or AddBodyAllRanks, as these increment the counter themselves.
    void IncrementNumBodiesGlobal() { num_bodies_global++; }

    /// Return true if pos is within this rank's sub-domain.
    bool InSub(const ChVector<double>& pos) const;

    /// Create a new body, consistent with the contact method and collision model used by this system.
    /// The returned body is not added to the system.
    virtual ChBody* NewBody() override;

    /// Create a new body with non-centroidal reference frame, consistent with the contact method and
    /// collision model used by this system.  The returned body is not added to the system.
    virtual ChBodyAuxRef* NewBodyAuxRef() override;

    /// Add a body to the system. 
    /// This function should be called *on all ranks*.
    /// AddBody classifies the body and decides whether or not to keep it on each rank.
    virtual void AddBody(std::shared_ptr<ChBody> newbody) override;

    /// Add a body to the system on all ranks, regardless of its location.
    /// This body should not have associated collision geometry.
    /// NOTE: A body crossing multiple sub-domains will not be correctly advanced.
    void AddBodyAllRanks(std::shared_ptr<ChBody> body);

    /// Add the specified body on each rank on which the function is called.
    /// This function is provided as a more efficient mechanism for initializing large numbers
    /// of bodies in a distributed system (as the ChBody objects need not be created on all ranks, 
    /// but rather only on those ranks covering the subdomain in which the body is initially
    /// located).  See InSub() to decide if a 3D location is within a rank's subdomain.
    /// NOTE: after calling this function on a rank, it is the user's responsibility to increment
    /// the global number of bodies *on all ranks*, through a call to IncrementNumBodiesGlobal.
    void AddBodyTrust(std::shared_ptr<ChBody> newbody);

    /// Remove a body from the simulation based on the ID of the body (not based on
    /// object comparison between ChBodys). Should be called on all ranks to ensure
    /// that the correct body is found and removed where it exists.
    virtual void RemoveBody(std::shared_ptr<ChBody> body) override;

    /// Wraps the super-class Integrate_Y call and introduces a call that carries
    /// out all inter-rank communication.
    virtual bool Integrate_Y() override;

    /// Wraps super-class UpdateRigidBodies and adds a gid update.
    virtual void UpdateRigidBodies() override;

    /// Internal call for removing deactivating a body.
    /// Should not be called by the user.
    void RemoveBodyExchange(int index);

    /// Returns the ChDomainDistributed object associated with the system.
    ChDomainDistributed* GetDomain() const { return domain; }

    /// Returns the ChCommDistributed object associated with the system.
    ChCommDistributed* GetComm() const { return comm; }

    /// Prints msg to the user and ends execution with an MPI abort.
    void ErrorAbort(std::string msg);

    /// Prints out all valid body data. Should only be used for debugging.
    void PrintBodyStatus();

    /// Prints out all valid shape data. Should only be used for debugging.
    void PrintShapeData();

    /// Prints measures for computing efficiency.
    void PrintEfficiency();

    /// Returns the MPI communicator being used by the system.
    MPI_Comm GetMPIWorld() const { return world; }

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

    /// Removes all bodies below the given height - initial implementation of a
    /// deactivating boundary condition.
    int RemoveBodiesBelow(double z);

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

    friend class ChCommDistributed;
    friend class ChDomainDistributed;
};

} /* namespace chrono */
