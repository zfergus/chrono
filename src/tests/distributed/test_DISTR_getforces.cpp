#include <mpi.h>

#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <vector>

#include "chrono_distributed/collision/ChBoundary.h"
#include "chrono_distributed/collision/ChCollisionModelDistributed.h"
#include "chrono_distributed/physics/ChSystemDistributed.h"

#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsGenerators.h"
#include "chrono/utils/ChUtilsSamplers.h"

#include "chrono_parallel/solver/ChIterativeSolverParallel.h"

using namespace chrono;
using namespace chrono::collision;

#define MASTER 0

// Granular Properties
float Y = 2e6f;
float mu = 0.4f;
float cr = 0.05f;
double r = 0.01;
double rho = 4000;
double spacing = 2.5 * r;
double mass = rho * 4.0 / 3.0 * CH_C_PI * r * r * r;
ChVector<> inertia = (2.0 / 5.0) * mass * r * r * ChVector<>(1, 1, 1);

// Dimensions
double hx = 20 * r;
double hy = 20 * r;
double height = 20 * r;

// Simulation
double time_step = 1e-4;
double out_fps = 120;
unsigned int max_iteration = 100;
double tolerance = 1e-4;
double time_end = 2;

void AddContainer(ChSystemDistributed* sys) {
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
    sys->AddBodyAllRanks(bin);

    auto cb = new ChBoundary(bin);
    cb->AddPlane(ChFrame<>(ChVector<>(0, 0, 0), QUNIT), ChVector2<>(2.0 * hx, 2.0 * hy));
    cb->AddPlane(ChFrame<>(ChVector<>(-hx, 0, height / 2.0), Q_from_AngY(CH_C_PI_2)), ChVector2<>(height, 2.0 * hy));
    cb->AddPlane(ChFrame<>(ChVector<>(hx, 0, height / 2.0), Q_from_AngY(-CH_C_PI_2)), ChVector2<>(height, 2.0 * hy));
    cb->AddPlane(ChFrame<>(ChVector<>(0, -hy, height / 2.0), Q_from_AngX(-CH_C_PI_2)), ChVector2<>(2.0 * hx, height));
    cb->AddPlane(ChFrame<>(ChVector<>(0, hy, height / 2.0), Q_from_AngX(CH_C_PI_2)), ChVector2<>(2.0 * hx, height));
}

inline std::shared_ptr<ChBody> CreateBall(const ChVector<>& pos,
                                          std::shared_ptr<ChMaterialSurfaceSMC> ballMat,
                                          int* ballId,
                                          double m,
                                          ChVector<> inertia,
                                          double radius) {
    auto ball = std::make_shared<ChBody>(std::make_shared<ChCollisionModelDistributed>(), ChMaterialSurface::SMC);
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

size_t AddFallingBalls(ChSystemDistributed* sys) {
    double lowest = r;
    ChVector<double> box_center(0, 0, lowest + (height - lowest) / 2.0);
    ChVector<double> half_dims(hx - spacing, hy - spacing, (height - lowest) / 2.0);

    utils::GridSampler<> sampler(spacing);

    auto points = sampler.SampleBox(box_center, half_dims);

    auto ballMat = std::make_shared<ChMaterialSurfaceSMC>();
    ballMat->SetYoungModulus(Y);
    ballMat->SetFriction(mu);
    ballMat->SetRestitution(cr);
    ballMat->SetAdhesion(0);

    // Create the falling balls
    int ballId = 0;
    for (int i = 0; i < points.size(); i++) {
        auto ball = CreateBall(points[i], ballMat, &ballId, mass, inertia, r);
        sys->AddBody(ball);
    }

    return points.size();
}

int main(int argc, char* argv[]) {
    int num_ranks;
    int my_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    int num_threads = 2;

    if (my_rank == MASTER) {
        std::cout << "Number of threads:          " << num_threads << std::endl;
        std::cout << "Domain:                     " << 2 * hx << " x " << 2 * hy << " x " << 2 * height << std::endl;
        std::cout << "Simulation length:          " << time_end << std::endl;
    }

    // Create distributed system
    ChSystemDistributed my_sys(MPI_COMM_WORLD, r * 2, 1000);  // TODO

    if (my_rank == MASTER)
        std::cout << "Running on " << num_ranks << " MPI ranks" << std::endl;

    std::cout << "Rank: " << my_rank << " Node name: " << my_sys.node_name << std::endl;

    my_sys.SetParallelThreadNumber(num_threads);
    CHOMPfunctions::SetNumThreads(num_threads);

    my_sys.Set_G_acc(ChVector<double>(0, 0.0, -9.8));

    // Domain decomposition
    ChVector<double> domlo(-hx - spacing, -hy - spacing, 0 - spacing);
    ChVector<double> domhi(hx + spacing, hy + spacing, height + spacing);
    my_sys.GetDomain()->SetSplitAxis(1);  // Split along the y-axis
    my_sys.GetDomain()->SetSimDomain(domlo.x(), domhi.x(), domlo.y(), domhi.y(), domlo.z(), domhi.z());

    my_sys.GetDomain()->PrintDomain();

    // Set solver parameters
    my_sys.GetSettings()->solver.max_iteration_bilateral = max_iteration;
    my_sys.GetSettings()->solver.tolerance = tolerance;

    my_sys.GetSettings()->solver.contact_force_model = ChSystemSMC::ContactForceModel::Hertz;
    my_sys.GetSettings()->solver.adhesion_force_model = ChSystemSMC::AdhesionForceModel::Constant;

    my_sys.GetSettings()->collision.narrowphase_algorithm = NarrowPhaseType::NARROWPHASE_HYBRID_MPR;

    my_sys.GetSettings()->collision.bins_per_axis = vec3(10, 10, 10);

    AddContainer(&my_sys);
    size_t num_bodies = AddFallingBalls(&my_sys);
    std::vector<uint> target_bodies = {1, 10, (uint)num_bodies - 1};  // Body to be watched

    MPI_Barrier(my_sys.GetCommunicator());
    if (my_rank == MASTER)
        std::cout << "Total number of particles: " << num_bodies << std::endl;

    MPI_Barrier(my_sys.GetCommunicator());

    // Run simulation for specified time
    int num_steps = (int)std::ceil(time_end / time_step);
    int out_steps = (int)std::ceil((1 / time_step) / out_fps);
    int out_frame = 0;
    double time = 0;

    if (my_rank == MASTER)
        std::cout << "Starting Simulation" << std::endl;

    double t_start = MPI_Wtime();
    for (int i = 0; i < num_steps; i++) {
        my_sys.DoStepDynamics(time_step);
        time += time_step;
        // Single body force
        auto forces = my_sys.GetBodyContactForces(target_bodies);
        for (auto force : forces) {
            std::cout << "GID " << force.first << " (" << force.second.x() << ", " << force.second.y() << ", "
                      << force.second.z() << ")" << std::endl;
        }
    }
    double elapsed = MPI_Wtime() - t_start;

    if (my_rank == MASTER)
        std::cout << "\n\nTotal elapsed time = " << elapsed << std::endl;

    MPI_Finalize();
    return 0;
}