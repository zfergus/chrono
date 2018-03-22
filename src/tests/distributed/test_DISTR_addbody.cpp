#include <mpi.h>
#include <string>

#include "chrono_distributed/collision/ChCollisionModelDistributed.h"
#include "chrono_distributed/physics/ChSystemDistributed.h"

#include "chrono/ChConfig.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsGenerators.h"
#include "chrono/utils/ChUtilsSamplers.h"

using namespace chrono;
using namespace chrono::collision;

double radius = 0.025;
double spacing = 2.01 * radius;

double hx = 1e3 * radius;
double hy = 1e2 * radius;
double hz = 1e1 * radius;

void SetBallParams(std::shared_ptr<ChBody> ball, const ChVector<>& pos, std::shared_ptr<ChMaterialSurface> mat) {
    ball->SetMaterialSurface(mat);
    ball->SetMass(1);
    ball->SetInertiaXX(ChVector<>(1, 1, 1));
    ball->SetPos(pos);
    ball->SetRot(ChQuaternion<>(1, 0, 0, 0));
    ball->SetBodyFixed(false);
    ball->SetCollide(true);

    ball->GetCollisionModel()->ClearModel();
    utils::AddSphereGeometry(ball.get(), 1);
    ball->GetCollisionModel()->BuildModel();
}

size_t AddBalls(ChSystem* sys) {
    ChVector<double> box_center(0, 0, 0);
    ChVector<double> half_dims(hx, hy, hz);

    utils::GridSampler<> sampler(spacing);
    auto points = sampler.SampleBox(box_center, half_dims);

    auto mat = std::make_shared<ChMaterialSurfaceSMC>();
    mat->SetYoungModulus(2e6f);
    mat->SetFriction(0.5f);
    mat->SetRestitution(0.01f);
    mat->SetAdhesion(0);

    // Create the falling balls
    int ballId = 0;

    for (int i = 0; i < points.size(); i++) {
        auto ball = std::shared_ptr<ChBody>(sys->NewBody());
        SetBallParams(ball, points[i], mat);
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

    ////{
    ////    ChSystemParallelSMC sysP;

    ////    double t_start = MPI_Wtime();
    ////    auto num_balls = AddBalls(&sysP);
    ////    MPI_Barrier(MPI_COMM_WORLD);
    ////    double elapsed = MPI_Wtime() - t_start;

    ////    if (my_rank == 0) {
    ////        std::cout << "Number balls: " << num_balls << std::endl;
    ////        std::cout << "Total elapsed time: " << elapsed << std::endl;
    ////    }

    ////    MPI_Finalize();
    ////    return 0;
    ////}

    // Read program argument (method for adding bodies to the system)

    // Create distributed system
    ChSystemDistributed my_sys(MPI_COMM_WORLD, radius * 2, 1000000);

    // Domain decomposition
    ChVector<double> domlo(-hx - spacing, -hy - spacing, -hz - spacing);
    ChVector<double> domhi(+hx + spacing, +hy + spacing, +hz + spacing);
    my_sys.GetDomain()->SetSplitAxis(0);
    my_sys.GetDomain()->SetSimDomain(domlo.x(), domhi.x(), domlo.y(), domhi.y(), domlo.z(), domhi.z());

    // Create objects
    size_t num_balls;
    double t_start = MPI_Wtime();
    MPI_Barrier(my_sys.GetCommunicator());
    double elapsed = MPI_Wtime() - t_start;

    if (my_rank == 0) {
        std::cout << "Number balls: " << num_balls << std::endl;
        std::cout << "Number bodies (global): " << my_sys.GetNumBodiesGlobal() << std::endl;
        std::cout << "Total elapsed time: " << elapsed << std::endl;
    }

    MPI_Finalize();
    return 0;
}
