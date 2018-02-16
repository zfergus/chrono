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

#include "chrono_distributed/physics/ChSystemDistributed.h"
#include "chrono_parallel/collision/ChCollisionModelParallel.h"

#include "chrono/physics/ChBody.h"
#include "chrono/physics/ChSystem.h"

namespace chrono {

/// This class implements a custom collison callback that can be used to add
/// collision events with a plane to a system at each timestep.
class ChPlaneCB : public ChSystem::CustomCollisionCallback {
  public:
    ChPlaneCB(ChSystemDistributed* sys,  ///< Main system pointer
              ChBody* body,              ///< Associate body
              ChVector<> center,         ///< Center of plane
              ChVector<> u,              ///< Vector from center to one edge
              ChVector<> w,              ///< Vector from center to other edge. Should be orthogonal to u
              ChVector<> n               ///< Inward normal
              )
        : m_sys(sys), m_body(body), m_center(center) {
        m_hu = u.Length();
        m_hw = w.Length();
        m_u = u.GetNormalized();
        m_w = w.GetNormalized();
        m_n = n.GetNormalized();
    }

    // TODO
    void SetPos(ChVector<> center) { m_center = center; }
    ChVector<double> GetPos() { return m_center; }
    /// Main method called by the system at each collision detection phase
    virtual void OnCustomCollision(ChSystem* system) override;

  private:
    /// Checks for collision between the plane and a sphere
    void CheckSphereProfile(std::shared_ptr<ChBody> sphere);
    /// Find the nearest point in the bounded plane to the given pt.
    ChVector<double> SnapTo(ChVector<> pt);
    ChSystemDistributed* m_sys;  ///< Associated ChSystem
    ChBody* m_body;              ///< Associated ChBody object
    ChVector<> m_center;         ///< Center of plane
    ChVector<> m_u;              ///< Unit vector from center to one edge
    double m_hu;                 ///< Distance from center to u edge
    ChVector<> m_w;              ///< Unit vector from center to other edge
    double m_hw;                 ///< Distance from center to w edge
    ChVector<> m_n;              ///< Unit inward normal
};                               // namespace chrono

/// Function called once every timestep by the system to add all custom collisions
/// associated with the callback to the system.
void ChPlaneCB::OnCustomCollision(ChSystem* sys) {
    // Loop over all bodies in the system
    for (int i = 0; i < m_sys->data_manager->body_list->size(); i++) {
        auto sphere = (*m_sys->data_manager->body_list)[i];
        // TODO switch on shape type to be added later
        // Avoid colliding with other planes
        if ((std::dynamic_pointer_cast<collision::ChCollisionModelParallel>(sphere->GetCollisionModel()))
                ->GetNObjects() > 0)
            CheckSphereProfile(sphere);
    }
}

/// Checks for collision between a sphere shape and the plane
/// Adds a new contact to the associated system if there is contact
void ChPlaneCB::CheckSphereProfile(std::shared_ptr<ChBody> sphere) {
    // Mini broad-phase
    ChVector<> centerS(sphere->GetPos());
    auto pmodel = std::dynamic_pointer_cast<collision::ChCollisionModelParallel>(sphere->GetCollisionModel());
    double radius = pmodel->mData[0].B[0];

    // Broadphase
    ChVector<> delta = m_center - centerS;  // Displacement from sphere center to plane center
    if (std::abs(delta.Dot(m_u)) > m_hu + radius || std::abs(delta.Dot(m_w)) > m_hw + radius ||
        std::abs(delta.Dot(m_n)) > radius)
        return;

    // Narrowphase
    ChVector<double> snap = SnapTo(centerS);
    if ((snap - centerS).Length() > radius)
        return;

    // Fill in contact information and add the contact to the system.
    // Express all vectors in the global frame
    collision::ChCollisionInfo contact;

    // Off an edge
    if (std::abs(delta.Dot(m_u)) > m_hu || std::abs(delta.Dot(m_w)) > m_hw) {
        ChVector<> edge_pt = snap;
        ChVector<> vN = (centerS - edge_pt).GetNormalized();
        contact.vN = vN;
        contact.vpA = edge_pt;
        contact.vpB = centerS - radius * vN;
        contact.distance = (edge_pt - centerS).Length() - radius;
    } else {
        // Contact point on the plane
        ChVector<> vpA =
            m_center - (delta + (delta.Dot(m_n)) * m_n);  // TODO check this: negative because direction of delta

        // Contact point on sphere
        ChVector<> vpB = centerS - radius * m_n;
        contact.vN = m_n;
        contact.vpA = vpA;
        contact.vpB = vpB;
        contact.distance = std::abs(delta.Dot(m_n)) - radius;  // Should be negative
    }

    contact.modelA = m_body->GetCollisionModel().get();
    contact.modelB = sphere->GetCollisionModel().get();
    m_sys->data_manager->host_data.erad_rigid_rigid.push_back(radius);

    m_sys->GetContactContainer()->AddContact(contact);  // NOTE: Not thread-safe
}

ChVector<double> ChPlaneCB::SnapTo(ChVector<> pt) {
    ChVector<double> delta = pt - m_center;
    delta = delta - delta.Dot(m_n) * m_n;  // Puts delta in the plane

    double du = delta.Dot(m_u);
    // Snap to u edge
    if (du > m_hu) {
        delta -= (du - m_hu) * m_u;
    } else if (du < -m_hu) {
        delta += (-du - m_hu) * m_u;
    }

    double dw = delta.Dot(m_w);
    if (dw > m_hw) {
        delta -= (dw - m_hw) * m_w;
    } else if (dw < -m_hw) {
        delta += (-dw - m_hw) * m_w;
    }

    return m_center + delta;
}
}  // end namespace chrono