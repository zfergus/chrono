// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Radu Serban
// =============================================================================
//
// ChronoParallel test program using SMC method for frictional contact.
//
// The model simulated here consists of a number of objects falling onto a
// sloped surface.
//
// IMPORTANT PARAMETERS:
//  - time step
//  - particle radius ( => inertial properties)
//  - Young's modulus
//
// The global reference frame has Z up.
// =============================================================================

#include <cmath>
#include <cstdio>

#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsGenerators.h"
#include "chrono/utils/ChUtilsSamplers.h"

#include "chrono_parallel/physics/ChSystemParallel.h"

#include "chrono_distributed/collision/ChBoundary.h"

#include "chrono_opengl/ChOpenGLWindow.h"

using namespace chrono;
using namespace chrono::collision;

// Granular Properties
float Y = 2e6f;
float mu = 0.4f;
float cr = 0.05f;
double gran_radius = 0.025;
double rho = 1000;
double mass = 4.0 / 3.0 * CH_C_PI * gran_radius * gran_radius * gran_radius;
ChVector<> inertia = (2.0 / 5.0) * mass * gran_radius * gran_radius * ChVector<>(1, 1, 1);
double spacing = 2.5 * gran_radius;  // Distance between adjacent centers of particles

double slope_angle = CH_C_PI / 4;  // Angle of sloped wall from the horizontal

// Simulation
double time_step = 2e-5;

unsigned int max_iteration = 100;  // not relevant here (SMC, no joints)
double tolerance = 1e-4;           // not relevant here (SMC, no joints)

std::shared_ptr<ChBoundary> AddSlopedWall(ChSystemParallel* sys) {
    int binId = -200;

    auto mat = std::make_shared<ChMaterialSurfaceSMC>();
    mat->SetYoungModulus(Y);
    mat->SetFriction(mu);
    mat->SetRestitution(cr);

    auto container = std::make_shared<ChBody>(std::make_shared<ChCollisionModelParallel>(), ChMaterialSurface::SMC);
    container->SetMaterialSurface(mat);
    container->SetIdentifier(binId);
    container->SetMass(1);
    container->SetPos(ChVector<>(0));
    container->SetCollide(false);
    container->SetBodyFixed(true);
    container->GetCollisionModel()->ClearModel();

    sys->AddBody(container);

    auto boundary = std::shared_ptr<ChBoundary>(new ChBoundary(container));
    boundary->AddPlane(ChFrame<>(ChVector<>(0, 0, 0), Q_from_AngY(-slope_angle)),
                       ChVector2<>(100 * gran_radius, 100 * gran_radius));
    boundary->AddVisualization(3 * gran_radius);

    return boundary;
}

size_t AddFallingBalls(ChSystemParallel* sys) {
    double hx = 20 * gran_radius;
    double hy = 4 * gran_radius;
    double hz = 2 * gran_radius;
    ChVector<double> box_center(0, 0, hx * std::tan(slope_angle) + 5 * gran_radius);

    utils::HCPSampler<> sampler(spacing);
    auto points = sampler.SampleBox(box_center, ChVector<>(hx, hy, hz));

    auto ballMat = std::make_shared<ChMaterialSurfaceSMC>();
    ballMat->SetYoungModulus(Y);
    ballMat->SetFriction(mu);
    ballMat->SetRestitution(cr);
    ballMat->SetAdhesion(0);

    int ballId = 0;
    for (int i = 0; i < points.size(); i++) {
        auto ball = std::make_shared<ChBody>(std::make_shared<ChCollisionModelParallel>(), ChMaterialSurface::SMC);
        ball->SetMaterialSurface(ballMat);

        ball->SetIdentifier(ballId++);
        ball->SetMass(mass);
        ball->SetInertiaXX(inertia);
        ball->SetPos(points[i]);
        ball->SetRot(ChQuaternion<>(1, 0, 0, 0));
        ball->SetBodyFixed(false);
        ball->SetCollide(true);

        ball->GetCollisionModel()->ClearModel();
        utils::AddSphereGeometry(ball.get(), gran_radius);
        ball->GetCollisionModel()->BuildModel();

        sys->AddBody(ball);
    }

    return points.size();
}

int main(int argc, char* argv[]) {
    GetLog() << "Copyright (c) 2017 projectchrono.org\nChrono version: " << CHRONO_VERSION << "\n\n";

    int threads = 2;

    // Create system
    ChSystemParallelSMC sys;
    sys.Set_G_acc(ChVector<>(0, 0, -9.8));

    // Set number of threads.
    int max_threads = CHOMPfunctions::GetNumProcs();
    if (threads > max_threads)
        threads = max_threads;
    sys.SetParallelThreadNumber(threads);
    CHOMPfunctions::SetNumThreads(threads);

    // Set solver parameters
    sys.GetSettings()->solver.max_iteration_bilateral = max_iteration;
    sys.GetSettings()->solver.tolerance = tolerance;

    sys.GetSettings()->collision.narrowphase_algorithm = NarrowPhaseType::NARROWPHASE_HYBRID_MPR;
    sys.GetSettings()->collision.bins_per_axis = vec3(10, 10, 10);

    sys.GetSettings()->solver.contact_force_model = ChSystemSMC::ContactForceModel::Hooke;
    sys.GetSettings()->solver.adhesion_force_model = ChSystemSMC::AdhesionForceModel::Constant;

    // Create objects
    auto boundary = AddSlopedWall(&sys);
    auto actual_num_bodies = AddFallingBalls(&sys);
    std::cout << "Created " << actual_num_bodies << " balls." << std::endl;

    // Perform the simulation
    opengl::ChOpenGLWindow& gl_window = opengl::ChOpenGLWindow::getInstance();
    gl_window.Initialize(1280, 720, "Boundary test SMC", &sys);
    gl_window.SetCamera(ChVector<>(-20 * gran_radius, -100 * gran_radius, 0), ChVector<>(0, 0, 0), ChVector<>(0, 0, 1), 0.01f);
    gl_window.SetRenderMode(opengl::WIREFRAME);

    while (gl_window.Active()) {
        gl_window.DoStepDynamics(time_step);
        gl_window.Render();
    }

    return 0;
}

