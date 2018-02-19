// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2014 projectchrono.org
// All right reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
//
// Test triangle collision shape in Chrono::Parallel
//
// The global reference frame has Z up.
// All units SI.
// =============================================================================

#include <cmath>
#include <cstdio>
#include <vector>

#include "chrono/ChConfig.h"
#include "chrono/core/ChFileutils.h"
#include "chrono/core/ChStream.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsGeometry.h"
#include "chrono/utils/ChUtilsInputOutput.h"

#include "chrono_parallel/physics/ChSystemParallel.h"
#include "chrono_parallel/solver/ChSystemDescriptorParallel.h"
#include "chrono_parallel/collision/ChNarrowphaseRUtils.h"

// Note: CHRONO_OPENGL is defined in ChConfig.h
#ifdef CHRONO_OPENGL
#include "chrono_opengl/ChOpenGLWindow.h"
#endif

using namespace chrono;
using namespace chrono::collision;

using std::cout;
using std::endl;

// -----------------------------------------------------------------------------
// Problem setup
// -----------------------------------------------------------------------------

// Comment the following line to use NSC contact
#define USE_SMC

ChVector<> initPos(0.1, 0.1, 0.05);
ChQuaternion<> initRot(1.0, 0.0, 0.0, 0.0);
//ChQuaternion<> initRot = Q_from_AngAxis(CH_C_PI / 3, ChVector<>(1, 0, 0));

ChVector<> initLinVel(0.0, 0.0, 0.0);
ChVector<> initAngVel(0.0, 0.0, 0.0);

// -----------------------------------------------------------------------------
// Simulation parameters
// -----------------------------------------------------------------------------

// Desired number of OpenMP threads (will be clamped to maximum available)
int threads = 10;

// Perform dynamic tuning of number of threads?
bool thread_tuning = true;

// Simulation duration.
double time_end = 10;

// Solver parameters
#ifdef USE_SMC
double time_step = 1e-4;
int max_iteration = 20;
#else
double time_step = 1e-3;
int max_iteration_normal = 30;
int max_iteration_sliding = 20;
int max_iteration_spinning = 0;
float contact_recovery_speed = 0.1;
#endif

// Output
int out_fps = 60;

// =============================================================================
// Create ground body
// =============================================================================
void CreateGround(ChSystemParallel* system) {
#ifdef USE_SMC
    auto mat_g = std::make_shared<ChMaterialSurfaceSMC>();
    mat_g->SetYoungModulus(1e7f);
    mat_g->SetFriction(0.7f);
    mat_g->SetRestitution(0.01f);

    auto ground = std::make_shared<ChBody>(std::make_shared<ChCollisionModelParallel>(), ChMaterialSurface::SMC);
    ground->SetMaterialSurface(mat_g);
#else
    auto mat_g = std::make_shared<ChMaterialSurfaceNSC>();
    mat_g->SetFriction(0.7f);

    auto ground = std::make_shared<ChBody>(std::make_shared<ChCollisionModelParallel>());
    ground->SetMaterialSurface(mat_g);
#endif

    ground->SetIdentifier(-1);
    ground->SetMass(1);
    ground->SetPos(ChVector<>(0, 0, 0));
    ground->SetRot(ChQuaternion<>(1, 0, 0, 0));
    ground->SetBodyFixed(true);
    ground->SetCollide(true);

    // Set fixed contact shapes (grid of 10x10 spheres)
    double spacing = 0.6;
    double bigR = 1;
    ground->GetCollisionModel()->ClearModel();
    for (int ix = -5; ix < 5; ix++) {
        for (int iy = -5; iy < 5; iy++) {
            ChVector<> pos(ix * spacing, iy * spacing, -bigR);
            utils::AddSphereGeometry(ground.get(), bigR, pos);
        }
    }
    ground->GetCollisionModel()->BuildModel();

    system->AddBody(ground);
}

// =============================================================================
// Create falling object
// =============================================================================
std::shared_ptr<ChBody> CreateObject(ChSystemParallel* system) {
    double rho_o = 2000.0;

#ifdef USE_SMC
    auto mat_o = std::make_shared<ChMaterialSurfaceSMC>();
    mat_o->SetYoungModulus(1e7f);
    mat_o->SetFriction(0.7f);
    mat_o->SetRestitution(0.01f);

    auto obj = std::make_shared<ChBody>(std::make_shared<ChCollisionModelParallel>(), ChMaterialSurface::SMC);
    obj->SetMaterialSurface(mat_o);
#else
    auto mat_o = std::make_shared<ChMaterialSurfaceNSC>();
    mat_o->SetFriction(0.7f);

    auto obj = std::make_shared<ChBody>(std::make_shared<ChCollisionModelParallel>());
    obj->SetMaterialSurface(mat_o);
#endif

    obj->SetIdentifier(1);
    obj->SetCollide(true);
    obj->SetBodyFixed(false);

    // Mass and inertia
    double mass = 1;
    ChVector<> inertia = 1e-3 * mass * ChVector<>(0.1, 0.1, 0.1);
    obj->SetMass(mass);
    obj->SetInertia(inertia);

    // Set contact and visualization shape
    obj->GetCollisionModel()->ClearModel();
    double len = 1;
    ChVector<> A(len, -len, 0);
    ChVector<> B(-len, -len, 0);
    ChVector<> C(0, len, 0);	
    utils::AddTriangle(obj.get(), A, B, C, "triangle");
    obj->GetCollisionModel()->BuildModel();

    // Set initial state.
    obj->SetPos(initPos);
    obj->SetRot(initRot);
    obj->SetPos_dt(initLinVel);
    obj->SetWvel_loc(initAngVel);

    // Add object to system.
    system->AddBody(obj);

    return obj;
}

// =============================================================================
// =============================================================================
int main(int argc, char* argv[]) {
    // Create system.
    char title[100];
#ifdef USE_SMC
    sprintf(title, "Object Drop >> SMC");
    cout << "Create SMC system" << endl;
    ChSystemParallelSMC* msystem = new ChSystemParallelSMC();
#else
    sprintf(title, "Object Drop >> NSC");
    cout << "Create NSC system" << endl;
    ChSystemParallelNSC* msystem = new ChSystemParallelNSC();
#endif

    msystem->Set_G_acc(ChVector<>(0, 0, -9.81));

    // ----------------------
    // Set number of threads.
    // ----------------------

    int max_threads = CHOMPfunctions::GetNumProcs();
    if (threads > max_threads)
        threads = max_threads;
    msystem->SetParallelThreadNumber(threads);
    CHOMPfunctions::SetNumThreads(threads);
    cout << "Using " << threads << " threads" << endl;

    // ---------------------
    // Edit system settings.
    // ---------------------

    msystem->GetSettings()->solver.tolerance = 1e-3;

#ifdef USE_SMC
    msystem->GetSettings()->collision.narrowphase_algorithm = NarrowPhaseType::NARROWPHASE_HYBRID_MPR;
#else
    msystem->GetSettings()->solver.solver_mode = SolverMode::SLIDING;
    msystem->GetSettings()->solver.max_iteration_normal = max_iteration_normal;
    msystem->GetSettings()->solver.max_iteration_sliding = max_iteration_sliding;
    msystem->GetSettings()->solver.max_iteration_spinning = max_iteration_spinning;
    msystem->GetSettings()->solver.alpha = 0;
    msystem->GetSettings()->solver.contact_recovery_speed = contact_recovery_speed;
    msystem->ChangeSolverType(SolverType::APGDREF);

    msystem->GetSettings()->solver.contact_recovery_speed = 1;
#endif

    msystem->GetSettings()->collision.bins_per_axis = vec3(10, 10, 10);

    // Create bodies.
    CreateGround(msystem);
    auto obj = CreateObject(msystem);

#ifdef CHRONO_OPENGL
    // Initialize OpenGL
    opengl::ChOpenGLWindow& gl_window = opengl::ChOpenGLWindow::getInstance();
    gl_window.Initialize(1280, 720, title, msystem);
    gl_window.SetCamera(ChVector<>(0, -5, 2), ChVector<>(0, 0, 0), ChVector<>(0, 0, 1));
    gl_window.SetRenderMode(opengl::WIREFRAME);
#endif

    // Run simulation for specified time.
    int out_steps = (int)std::ceil((1.0 / time_step) / out_fps);

    double time = 0;
    int sim_frame = 0;
    int out_frame = 0;
    int next_out_frame = 0;
    double exec_time = 0;
    int num_contacts = 0;

    while (time < time_end) {
        if (sim_frame == next_out_frame) {
            cout << "------------ Output frame:   " << out_frame << endl;
            cout << "             Sim frame:      " << sim_frame << endl;
            cout << "             Time:           " << time << endl;
            cout << "             Avg. contacts:  " << num_contacts / out_steps << endl;
            cout << "             Execution time: " << exec_time << endl;
            cout << endl;
            cout << obj->GetPos().z() << endl;
            out_frame++;
            next_out_frame += out_steps;
            num_contacts = 0;
        }

        obj->SetRot(QUNIT);

#ifdef CHRONO_OPENGL
        // OpenGL simulation step
        if (gl_window.Active()) {
            gl_window.DoStepDynamics(time_step);
            gl_window.Render();
        } else
            break;
#else
        // Advance dynamics.
        msystem->DoStepDynamics(time_step);
#endif

        // Update counters.
        time += time_step;
        sim_frame++;
        exec_time += msystem->GetTimerStep();
        num_contacts += msystem->GetNcontacts();
    }

    // Final stats
    cout << "==================================" << endl;
    cout << "Simulation time:   " << exec_time << endl;
    cout << "Number of threads: " << threads << endl;

    return 0;
}
