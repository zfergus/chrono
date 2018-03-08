#include <cmath>
#include <iostream>

#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsGenerators.h"
#include "chrono/utils/ChUtilsSamplers.h"

#include "chrono_distributed/collision/ChBoundary.h"
#include "chrono_parallel/physics/ChSystemParallel.h"
#include "chrono_parallel/solver/ChIterativeSolverParallel.h"

#include "chrono_opengl/ChOpenGLWindow.h"

using namespace chrono;
using namespace chrono::collision;

// Granular Properties
float Y = 2e6f;
float mu = 0.4f;
float cr = 0.05f;
double gran_radius = 0.0025;  // 2.5mm radius
double rho = 4000;
double mass = 4.0 / 3.0 * CH_C_PI * gran_radius * gran_radius *
              gran_radius;  // TODO shape dependent: more complicated than you'd think...
ChVector<> inertia = (2.0 / 5.0) * mass * gran_radius * gran_radius * ChVector<>(1, 1, 1);
double spacing = 4.5 * gran_radius;  // Distance between adjacent centers of particles

// Dimensions
double hy = 50 * gran_radius;                // Half y dimension
double height = 150 * gran_radius;            // Height of the box
double slope_angle = CH_C_PI / 4;            // Angle of sloped wall from the horizontal
double dx = height / std::tan(slope_angle);  // x width of slope
double settling_gap = 0 * gran_radius;       // Width of opening of the hopper during settling phase
int split_axis = 1;                          // Split domain along y axis
double pouring_gap = 4 * gran_radius;        // Width of opening of the hopper during pouring phase
double settling_time = 0.25;

size_t high_x_wall;

// Simulation
double time_step = 1e-5;
double out_fps = 120;
unsigned int max_iteration = 100;
double tolerance = 1e-4;

void Monitor(chrono::ChSystemParallel* system, int rank) {
    double TIME = system->GetChTime();
    double STEP = system->GetTimerStep();
    double BROD = system->GetTimerCollisionBroad();
    double NARR = system->GetTimerCollisionNarrow();
    double SOLVER = system->GetTimerSolver();
    double UPDT = system->GetTimerUpdate();
    double EXCH = system->data_manager->system_timer.GetTime("Exchange");
    int BODS = system->GetNbodies();
    int CNTC = system->GetNcontacts();
    double RESID = std::static_pointer_cast<chrono::ChIterativeSolverParallel>(system->GetSolver())->GetResidual();
    int ITER = std::static_pointer_cast<chrono::ChIterativeSolverParallel>(system->GetSolver())->GetTotalIterations();

    printf("%d|   %8.5f | %7.4f | E%7.4f | B%7.4f | N%7.4f | %7.4f | %7.4f | %7d | %7d | %7d | %7.4f\n", rank, TIME,
           STEP, EXCH, BROD, NARR, SOLVER, UPDT, BODS, CNTC, ITER, RESID);
}

std::shared_ptr<ChBoundary> AddContainer(ChSystemParallelSMC* sys) {
    int binId = -200;

    auto mat = std::make_shared<ChMaterialSurfaceSMC>();
    mat->SetYoungModulus(Y);
    mat->SetFriction(mu);
    mat->SetRestitution(cr);

    auto bin = std::make_shared<ChBody>(std::make_shared<ChCollisionModelParallel>(), ChMaterialSurface::SMC);
    bin->SetMaterialSurface(mat);
    bin->SetIdentifier(binId);
    bin->SetMass(1);
    bin->SetPos(ChVector<>(0, 0, 0));
    bin->SetCollide(true);
    bin->SetBodyFixed(true);
    sys->AddBody(bin);

    auto cb = std::make_shared<ChBoundary>(bin);
    // Sloped Wall
    cb->AddPlane(ChFrame<>(ChVector<>(settling_gap + dx / 2, 0, height / 2), Q_from_AngY(-slope_angle)),
                 ChVector2<>(std::sqrt(dx * dx + height * height), 2 * hy));
    high_x_wall = 0;

    // Vertical wall
    cb->AddPlane(ChFrame<>(ChVector<>(0, 0, height / 2), Q_from_AngY(CH_C_PI_2)), ChVector2<>(height, 2 * hy));

    // Parallel vertical walls
    cb->AddPlane(ChFrame<>(ChVector<>((settling_gap + dx) / 2, -hy, height / 2), Q_from_AngX(-CH_C_PI_2)),
                 ChVector2<>(settling_gap + dx, height));
    cb->AddPlane(ChFrame<>(ChVector<>((settling_gap + dx) / 2, hy, height / 2), Q_from_AngX(CH_C_PI_2)),
                 ChVector2<>(settling_gap + dx, height));
    cb->AddVisualization(0, 3 * gran_radius);
    cb->AddVisualization(1, 3 * gran_radius);

    return cb;
}

inline std::shared_ptr<ChBody> CreateBall(const ChVector<>& pos,
                                          std::shared_ptr<ChMaterialSurfaceSMC> ballMat,
                                          int* ballId,
                                          double m,
                                          ChVector<> inertia,
                                          double radius) {
    auto ball = std::make_shared<ChBody>(std::make_shared<ChCollisionModelParallel>(), ChMaterialSurface::SMC);
    ball->SetMaterialSurface(ballMat);

    ball->SetIdentifier(*ballId++);
    ball->SetMass(m);
    ball->SetInertiaXX(inertia);
    ball->SetPos(pos);
    ball->SetRot(ChQuaternion<>(1, 0, 0, 0));
    ball->SetBodyFixed(false);
    ball->SetCollide(true);

    ball->GetCollisionModel()->ClearModel();
    utils::AddSphereGeometry(ball.get(), radius);
    ball->GetCollisionModel()->BuildModel();
    return ball;
}

size_t AddFallingBalls(ChSystemParallelSMC* sys) {
    double first_layer_width = 10 * gran_radius;

    // utils::GridSampler<> sampler(spacing);
    utils::HCPSampler<> sampler(spacing);
    size_t count = 0;
    auto ballMat = std::make_shared<ChMaterialSurfaceSMC>();
    ballMat->SetYoungModulus(Y);
    ballMat->SetFriction(mu);
    ballMat->SetRestitution(cr);
    ballMat->SetAdhesion(0);

    ChVector<double> box_center(dx / 2, 0, height / 2);
    ChVector<double> h_dims(dx / 2, hy, height / 2);
    ChVector<double> padding = spacing * ChVector<double>(1, 1, 1);
    ChVector<double> half_dims = h_dims - padding;
    auto points = sampler.SampleBox(box_center, half_dims);
    int ballId = 0;
    for (int i = 0; i < points.size(); i++) {
        if (points[i].z() > (height * points[i].x()) / dx + 3 * gran_radius && points[i].x() < 10 * gran_radius) {
            auto ball = CreateBall(points[i], ballMat, &ballId, mass, inertia, gran_radius);
            sys->AddBody(ball);
            count++;
        }
    }

    return count;
}

int main(int argc, char* argv[]) {
    int num_threads = 8;
    double time_end = 4;
    bool verbose = true;
    bool monitor = true;

    if (verbose) {
        std::cout << "Number of threads:          " << num_threads << std::endl;
        std::cout << "Simulation length:          " << time_end << std::endl;
        std::cout << "Monitor?                    " << monitor << std::endl;
    }

    // Create distributed system
    ChSystemParallelSMC my_sys;  // TODO

    my_sys.SetParallelThreadNumber(num_threads);
    CHOMPfunctions::SetNumThreads(num_threads);

    my_sys.Set_G_acc(ChVector<double>(0, 0, -9.8));

    // Set solver parameters
    my_sys.GetSettings()->solver.max_iteration_bilateral = max_iteration;
    my_sys.GetSettings()->solver.tolerance = tolerance;

    my_sys.GetSettings()->solver.contact_force_model = ChSystemSMC::ContactForceModel::Hooke;
    my_sys.GetSettings()->solver.adhesion_force_model = ChSystemSMC::AdhesionForceModel::Constant;

    my_sys.GetSettings()->collision.narrowphase_algorithm = NarrowPhaseType::NARROWPHASE_HYBRID_MPR;

    my_sys.GetSettings()->collision.bins_per_axis = vec3(10, 3, 10);

    auto cb = AddContainer(&my_sys);
    auto actual_num_bodies = AddFallingBalls(&my_sys);

    // Run simulation for specified time
    int num_steps = (int)std::ceil(time_end / time_step);
    int out_steps = (int)std::ceil((1 / time_step) / out_fps);
    int out_frame = 0;

    double time = 0;

    std::cout << "Starting Simulation" << std::endl;

    bool settling = true;

    // Perform the simulation
    opengl::ChOpenGLWindow& gl_window = opengl::ChOpenGLWindow::getInstance();
    gl_window.Initialize(1280, 720, "Boundary test SMC", &my_sys);
    gl_window.SetCamera(ChVector<>(-20 * gran_radius, -100 * gran_radius, 0), ChVector<>(0, 0, 0), ChVector<>(0, 0, 1),
                        0.01f);
    gl_window.SetRenderMode(opengl::WIREFRAME);
    for (int i = 0; gl_window.Active(); i++) {
        gl_window.DoStepDynamics(time_step);
        time += time_step;
        if (settling && time >= settling_time) {
            settling = false;
            cb->UpdatePlane(high_x_wall,
                            ChFrame<>(ChVector<>(pouring_gap + dx / 2, 0, height / 2), Q_from_AngY(-slope_angle)));
        }
        gl_window.Render();
    }

    return 0;
}