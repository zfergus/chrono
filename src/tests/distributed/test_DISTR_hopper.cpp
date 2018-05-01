#ifndef PAR
#define MASTER 0
#include <mpi.h>
#include "chrono_distributed/collision/ChBoundary.h"
#include "chrono_distributed/collision/ChCollisionModelDistributed.h"
#include "chrono_distributed/physics/ChSystemDistributed.h"
#endif

#include <omp.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsGenerators.h"
#include "chrono/utils/ChUtilsSamplers.h"

#include "chrono_thirdparty/filesystem/path.h"
#include "chrono_thirdparty/filesystem/resolver.h"

#include "chrono_thirdparty/SimpleOpt/SimpleOpt.h"

#include "GeometryUtils.h"
#include "chrono_parallel/solver/ChIterativeSolverParallel.h"

//#undef CHRONO_OPENGL

#ifdef CHRONO_OPENGL
#include "chrono_opengl/ChOpenGLWindow.h"
#endif

using namespace chrono;
using namespace chrono::collision;

// ID values to identify command line arguments
enum { OPT_HELP, OPT_THREADS, OPT_TIME, OPT_MONITOR, OPT_OUTPUT_DIR, OPT_VERBOSE, OPT_MIX };

// Table of CSimpleOpt::Soption structures. Each entry specifies:
// - the ID for the option (returned from OptionId() during processing)
// - the option as it should appear on the command line
// - type of the option
// The last entry must be SO_END_OF_OPTIONS
CSimpleOptA::SOption g_options[] = {
    {OPT_HELP, "--help", SO_NONE}, {OPT_HELP, "-h", SO_NONE},     {OPT_THREADS, "-n", SO_REQ_CMB},
    {OPT_TIME, "-t", SO_REQ_CMB},  {OPT_MONITOR, "-m", SO_NONE},  {OPT_OUTPUT_DIR, "-o", SO_REQ_CMB},
    {OPT_VERBOSE, "-v", SO_NONE},  {OPT_MIX, "-mix", SO_REQ_CMB}, SO_END_OF_OPTIONS};

bool GetProblemSpecs(int argc,
                     char** argv,
                     int rank,
                     int& num_threads,
                     double& time_end,
                     bool& monitor,
                     bool& verbose,
                     bool& output_data,
                     std::string& outdir);

void ShowUsage();

// Material Properties
float Y = 1e7f;
float mu = 0.3f;
float cr = 0.1f;

// Radius generation
double rad_min = 0.05;
double rad_max = 0.075;
double mean_radius = (rad_max - rad_min) / 2.0;
double stddev_radius = (rad_max - rad_min) / 4.0;
unsigned seed = 132;

std::default_random_engine generator(seed);
std::normal_distribution<double> rad_dist(mean_radius, stddev_radius);
inline double GetRadius() {
    double r = rad_dist(generator);
    if (r >= rad_min && r <= rad_max)
        return r;
    else if (r < rad_min)
        return rad_min;
    else
        return rad_max;
}

double rho = 2500.0;
inline double GetMass(double r) {
    return 4.0 * CH_C_PI * r * r * r / 3.0;  // TODO shape dependent: more complicated than you'd think...
}

inline ChVector<> GetInertia(double m, double r) {
    return (2.0 * m * r * r / 5.0) * ChVector<>(1, 1, 1);
}

double spacing = 2 * rad_max;  // Distance between adjacent centers of particles

// Dimensions
double hy = 50 * rad_max;                    // 50             // Half y dimension
double height = 200 * rad_max;               // 150          // Height of the box
double slope_angle = CH_C_PI / 5;            // Angle of sloped wall from the horizontal
double dx = height / std::tan(slope_angle);  // x width of slope
double settling_gap = 0 * rad_max;           // Width of opening of the hopper during settling phase
double pouring_gap = 6 * rad_max;            // Width of opening of the hopper during pouring phase
double settling_time = 1.5;
#ifndef PAR
int split_axis = 2;  // Split domain along z axis // TODO
#endif
size_t high_x_wall;

// Simulation
double time_step = 1e-5;
double out_fps = 120;
unsigned int max_iteration = 100;
double tolerance = 1e-4;
double remove_fps = 60;

// For layered addition
double layer_thickness = 3 * spacing;
int num_layers = (int)(height / layer_thickness);
double layer_fall_time = std::sqrt(2 * (layer_thickness + 2 * rad_max) / 9.8);
int layer_fall_steps = (int)(layer_fall_time / time_step);

// Geometry
enum { SPHERE_H, BISPHERE_H, SPHERE_BISPHERE_H, ASYM_H, SPHERE_ASYM_H, SPHERE_BISPHERE_ASYM_H };
int mix = SPHERE_H;

void my_abort() {
#ifndef PAR
    MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
#else
    std::abort(1);
#endif
}

int GetGeometry(int id, ChBody* body, double r) {
    switch (mix) {
        case SPHERE_H:
            utils::AddSphereGeometry(body, r);
            break;

        case BISPHERE_H:
            AddBiSphere(body, r);
            break;

        case SPHERE_BISPHERE_H:
            if (id % 2 == 0)
                utils::AddSphereGeometry(body, r);
            else
                AddBiSphere(body, r);
            break;

        case ASYM_H:
            AddAsymmetricBisphere(body, r);
            break;

        case SPHERE_ASYM_H:
            if (id % 2 == 0)
                utils::AddSphereGeometry(body, r);
            else
                AddAsymmetricBisphere(body, r);
            break;

        case SPHERE_BISPHERE_ASYM_H:
            if (id % 3 == 0)
                utils::AddSphereGeometry(body, r);
            else if (id % 3 == 1)
                AddBiSphere(body, r);
            else
                AddAsymmetricBisphere(body, r);
            break;

        default:
            std::cout << "Select a mix type." << std::endl;
            my_abort();
            break;
    }

    return 0;
}

void WriteCSV(std::ofstream* file, int timestep_i, ChSystemDistributed* sys) {
    std::stringstream ss_particles;

    int i = 0;
    auto bl_itr = sys->data_manager->body_list->begin();

    for (; bl_itr != sys->data_manager->body_list->end(); bl_itr++, i++) {
        if (sys->ddm->comm_status[i] != chrono::distributed::EMPTY) {
            ChVector<> pos = (*bl_itr)->GetPos();
            ChVector<> vel = (*bl_itr)->GetPos_dt();

            ss_particles << timestep_i << "," << (*bl_itr)->GetGid() << "," << pos.x() << "," << pos.y() << ","
                         << pos.z() << "," << vel.Length() << std::endl;
        }
    }

    *file << ss_particles.str();
}

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

std::shared_ptr<ChBoundary> AddContainer(ChSystemDistributed* sys) {
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

    auto cb = std::make_shared<ChBoundary>(bin);
    // Sloped Wall
    cb->AddPlane(ChFrame<>(ChVector<>(settling_gap + dx / 2, 0, height / 2), Q_from_AngY(-slope_angle)),
                 ChVector2<>(std::sqrt(dx * dx + height * height), 2.01 * hy));
    high_x_wall = 0;

    // Vertical wall
    cb->AddPlane(ChFrame<>(ChVector<>(0, 0, height / 2), Q_from_AngY(CH_C_PI_2)),
                 ChVector2<>(height + 2 * rad_max, 2.01 * hy));

    // Parallel vertical walls
    cb->AddPlane(ChFrame<>(ChVector<>((settling_gap + dx) / 2, -hy, height / 2), Q_from_AngX(-CH_C_PI_2)),
                 ChVector2<>(settling_gap + dx, height + 2 * rad_max));
    cb->AddPlane(ChFrame<>(ChVector<>((settling_gap + dx) / 2, hy, height / 2), Q_from_AngX(CH_C_PI_2)),
                 ChVector2<>(settling_gap + dx, height + 2 * rad_max));
    cb->AddVisualization(0, 0.1 * rad_max);
    cb->AddVisualization(1, 0.1 * rad_max);

    return cb;
}

std::shared_ptr<ChBody> CreateBall(const ChVector<>& pos,
                                   std::shared_ptr<ChMaterialSurfaceSMC> ballMat,
                                   int& ballId,
                                   double r) {
    auto ball = std::make_shared<ChBody>(std::make_shared<ChCollisionModelDistributed>(), ChMaterialSurface::SMC);
    ball->SetMaterialSurface(ballMat);
    ball->SetIdentifier(ballId++);
    double mass = GetMass(r);
    ball->SetMass(mass);
    ball->SetInertiaXX(GetInertia(mass, r));
    ball->SetPos(pos);
    ball->SetRot(ChQuaternion<>(1, 0, 0, 0));
    ball->SetBodyFixed(false);
    ball->SetCollide(true);

    ball->GetCollisionModel()->ClearModel();
    GetGeometry(ballId, ball.get(), r);
    ball->GetCollisionModel()->BuildModel();

    return ball;
}

size_t AddFallingBalls(ChSystemDistributed* sys) {
    utils::HCPSampler<> sampler(spacing);
    size_t count = 0;
    auto ballMat = std::make_shared<ChMaterialSurfaceSMC>();
    ballMat->SetYoungModulus(Y);
    ballMat->SetFriction(mu);
    ballMat->SetRestitution(cr);
    ballMat->SetAdhesion(0);

    ChVector<double> box_center(dx / 2, 0, height / 2);
    ChVector<double> h_dims(dx / 2, hy, height / 2);
    ChVector<double> padding = 2 * rad_max * ChVector<double>(1, 1, 1);
    ChVector<double> half_dims = h_dims - padding;
    auto points = sampler.SampleBox(box_center, half_dims);
    int ballId = 0;
    double dz = rad_max * std::sin(CH_C_PI_2 - slope_angle);
    for (int i = 0; i < points.size(); i++) {
        if (points[i].z() > (height * points[i].x()) / dx + 2 * dz) {
            double r = GetRadius();
            auto ball = CreateBall(points[i], ballMat, ballId, r);
            sys->AddBody(ball);
            count++;
        }
    }

    return count;
}

size_t AddLayerOfBalls(ChSystemDistributed* sys) {
    std::cout << "LAYER\n";
    utils::HCPSampler<> sampler(spacing);
    size_t count = 0;
    auto ballMat = std::make_shared<ChMaterialSurfaceSMC>();
    ballMat->SetYoungModulus(Y);
    ballMat->SetFriction(mu);
    ballMat->SetRestitution(cr);
    ballMat->SetAdhesion(0);

    ChVector<double> box_center(dx / 2, 0, height);
    ChVector<double> h_dims(dx / 2, hy, layer_thickness / 2.0);
    ChVector<double> padding = 2 * rad_max * ChVector<double>(1, 1, 0);
    ChVector<double> half_dims = h_dims - padding;
    auto points = sampler.SampleBox(box_center, half_dims);
    int ballId = 0;
    double dz = rad_max * std::sin(CH_C_PI_2 - slope_angle);
    for (int i = 0; i < points.size(); i++) {
        if (points[i].z() > (height * points[i].x() - 10 * spacing) / dx + 2) {
            double r = GetRadius();
            auto ball = CreateBall(points[i], ballMat, ballId, r);
            sys->AddBody(ball);
            count++;
        }
    }

    return count;
}

int main(int argc, char* argv[]) {
    int num_ranks;
    int my_rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // Parse program arguments
    int num_threads;
    double time_end;
    std::string outdir;
    bool verbose;
    bool monitor;
    bool output_data;
    if (!GetProblemSpecs(argc, argv, my_rank, num_threads, time_end, monitor, verbose, output_data, outdir)) {
        MPI_Finalize();
        return 1;
    }

    // if (my_rank == 0) {
    // 	int foo;
    // 	std::cout << "Enter something too continue..." << std::endl;
    // 	std::cin >> foo;
    // }
    // MPI_Barrier(MPI_COMM_WORLD);
    if (my_rank == MASTER) {
        std::cout << "MIX TYPE: " << mix << std::endl;
    }

    // Output directory and files
    std::ofstream outfile;
    if (output_data) {
        // Create output directory
        if (my_rank == MASTER) {
            bool out_dir_exists = filesystem::path(outdir).exists();
            if (out_dir_exists) {
                std::cout << "Output directory already exists" << std::endl;
                my_abort();
                return 1;
            } else if (filesystem::create_directory(filesystem::path(outdir))) {
                if (verbose) {
                    std::cout << "Create directory = " << filesystem::path(outdir).make_absolute() << std::endl;
                }
            } else {
                std::cout << "Error creating output directory" << std::endl;
                my_abort();
                return 1;
            }
        }
    } else if (verbose && my_rank == MASTER) {
        std::cout << "Not writing data files" << std::endl;
    }

    if (verbose && my_rank == MASTER) {
        std::cout << "Number of threads:          " << num_threads << std::endl;
        // std::cout << "Domain:                     " << 2 * h_x << " x " << 2 * h_y << " x " << 2 * h_z <<
        // std::endl;
        std::cout << "Simulation length:          " << time_end << std::endl;
        std::cout << "Monitor?                    " << monitor << std::endl;
        std::cout << "Output?                     " << output_data << std::endl;
        if (output_data)
            std::cout << "Output directory:           " << outdir << std::endl;
    }

    // Create distributed system
    ChSystemDistributed my_sys(MPI_COMM_WORLD, rad_max * 2, 100000);  // TODO

    if (verbose) {
        if (my_rank == MASTER)
            std::cout << "Running on " << num_ranks << " MPI ranks" << std::endl;
        std::cout << "Rank: " << my_rank << " Node name: " << my_sys.node_name << std::endl;
    }

    my_sys.SetParallelThreadNumber(num_threads);
    CHOMPfunctions::SetNumThreads(num_threads);

    my_sys.Set_G_acc(ChVector<double>(0, 0, -9.8));

    // Domain decomposition
    ChVector<double> domlo(-2 * rad_max, -hy, -4 * rad_max);
    ChVector<double> domhi(pouring_gap + dx, hy, height + 2 * rad_max);
    my_sys.GetDomain()->SetSplitAxis(split_axis);
    my_sys.GetDomain()->SetSimDomain(domlo.x(), domhi.x(), domlo.y(), domhi.y(), domlo.z(), domhi.z());

    if (verbose)
        my_sys.GetDomain()->PrintDomain();

    // Set solver parameters
    my_sys.GetSettings()->solver.max_iteration_bilateral = max_iteration;
    my_sys.GetSettings()->solver.tolerance = tolerance;

    my_sys.GetSettings()->solver.contact_force_model = ChSystemSMC::ContactForceModel::Hooke;
    my_sys.GetSettings()->solver.adhesion_force_model = ChSystemSMC::AdhesionForceModel::Constant;

    my_sys.GetSettings()->collision.narrowphase_algorithm = NarrowPhaseType::NARROWPHASE_HYBRID_MPR;

    double factor = 3;
    ChVector<> subhi = my_sys.GetDomain()->GetSubHi();
    ChVector<> sublo = my_sys.GetDomain()->GetSubLo();
    ChVector<> subsize = (subhi - sublo) / (2 * rad_max);

    int binX = std::max((int)std::ceil(subsize.x() / factor), 1);
    int binY = std::max((int)std::ceil(subsize.y() / factor), 1);
    int binZ = std::max((int)std::ceil(subsize.z() / factor), 1);

    my_sys.GetSettings()->collision.bins_per_axis = vec3(binX, binY, binZ);
    if (verbose)
        printf("Rank: %d   bins: %d %d %d\n", my_rank, binX, binY, binZ);

    auto cb = AddContainer(&my_sys);
    auto actual_num_bodies = AddFallingBalls(&my_sys);
    MPI_Barrier(my_sys.GetCommunicator());

    if (my_rank == MASTER)
        std::cout << "Total number of particles: " << actual_num_bodies << std::endl;

    // Once the directory has been created, all ranks can make their output files
    MPI_Barrier(my_sys.GetCommunicator());
    std::string out_file_name = outdir + "/Rank" + std::to_string(my_rank) + ".csv";
    outfile.open(out_file_name);
    outfile << "t,gid,x,y,z,U\n" << std::flush;
    if (verbose)
        std::cout << "Rank: " << my_rank << "  Output file name: " << out_file_name << std::endl;

    // Run simulation for specified time
    int num_steps = (int)std::ceil(time_end / time_step);
    int out_steps = (int)std::ceil((1 / time_step) / out_fps);
    int out_frame = 0;
    int remove_steps = (int)std::ceil((1 / time_step) / remove_fps);

    double time = 0;

    if (verbose && my_rank == MASTER)
        std::cout << "Starting Simulation" << std::endl;

    bool settling = true;
    double t_start = MPI_Wtime();

    // Perform the simulation

#ifdef CHRONO_OPENGL

    opengl::ChOpenGLWindow& gl_window = opengl::ChOpenGLWindow::getInstance();
    gl_window.Initialize(1280, 720, "Boundary test SMC", &my_sys);
    gl_window.SetCamera(ChVector<>(-20 * rad_max, -100 * rad_max, 0), ChVector<>(0, 0, 0), ChVector<>(0, 0, 1), 0.01f);
    gl_window.SetRenderMode(opengl::WIREFRAME);
    for (int i = 0; gl_window.Active(); i++) {
        gl_window.DoStepDynamics(time_step);
        time += time_step;
        if (settling && time >= settling_time) {
            settling = false;
            cb->UpdatePlane(high_x_wall,
                            ChFrame<>(ChVector<>(pouring_gap + dx / 2, 0, height / 2), Q_from_AngY(-slope_angle)));
        }
        if (i % remove_steps == 0)
            my_sys.RemoveBodiesBelow(-2 * spacing);
        gl_window.Render();
    }

#else

     for (int i = 0; i < num_steps; i++) {
         my_sys.DoStepDynamics(time_step);
         time += time_step;
    
         if (i % out_steps == 0) {
             if (my_rank == MASTER)
                 std::cout << "Time: " << time << "    elapsed: " << MPI_Wtime() - t_start << std::endl;
             if (output_data) {
                 WriteCSV(&outfile, out_frame, &my_sys);
                 out_frame++;
             }
         }
    
         if (monitor)
             Monitor(&my_sys, my_rank);
         if (settling && time >= settling_time) {
             settling = false;
             cb->UpdatePlane(high_x_wall,
                             ChFrame<>(ChVector<>(pouring_gap + dx / 2, 0, height / 2), Q_from_AngY(-slope_angle)));
         }
         if (!settling && i % remove_steps == 0) {
             int remove_count = my_sys.RemoveBodiesBelow(-2 * spacing);
             if (my_rank == MASTER)
                 std::cout << remove_count << " bodies removed" << std::endl;
         }
     }
     double elapsed = MPI_Wtime() - t_start;
    
     if (my_rank == MASTER)
         std::cout << "\n\nTotal elapsed time = " << elapsed << std::endl;

#endif

    if (output_data)
        outfile.close();

    MPI_Finalize();
    return 0;
}

bool GetProblemSpecs(int argc,
                     char** argv,
                     int rank,
                     int& num_threads,
                     double& time_end,
                     bool& monitor,
                     bool& verbose,
                     bool& output_data,
                     std::string& outdir) {
    // Initialize parameters.
    num_threads = -1;
    time_end = -1;
    verbose = false;
    monitor = false;
    output_data = false;

    // Create the option parser and pass it the program arguments and the array of valid options.
    CSimpleOptA args(argc, argv, g_options);

    // Then loop for as long as there are arguments to be processed.
    while (args.Next()) {
        // Exit immediately if we encounter an invalid argument.
        if (args.LastError() != SO_SUCCESS) {
            if (rank == MASTER) {
                std::cout << "Invalid argument: " << args.OptionText() << std::endl;
                ShowUsage();
            }
            return false;
        }

        // Process the current argument.
        switch (args.OptionId()) {
            case OPT_HELP:
                if (rank == MASTER)
                    ShowUsage();
                return false;

            case OPT_THREADS:
                num_threads = std::stoi(args.OptionArg());
                break;

            case OPT_OUTPUT_DIR:
                output_data = true;
                outdir = args.OptionArg();
                break;

            case OPT_TIME:
                time_end = std::stod(args.OptionArg());
                break;

            case OPT_MONITOR:
                monitor = true;
                break;

            case OPT_VERBOSE:
                verbose = true;
                break;

            case OPT_MIX:
                mix = std::stoi(args.OptionArg());
        }
    }

    // Check that required parameters were specified
    if (num_threads == -1 || time_end <= 0) {
        if (rank == MASTER) {
            std::cout << "Invalid parameter or missing required parameter." << std::endl;
            ShowUsage();
        }
        return false;
    }

    return true;
}

void ShowUsage() {
    std::cout << "Usage: mpirun -np <num_ranks> ./demo_DISTR_scaling [ARGS]" << std::endl;
    std::cout << "-n=<nthreads>   Number of OpenMP threads on each rank [REQUIRED]" << std::endl;
    std::cout << "-t=<end_time>   Simulation length [REQUIRED]" << std::endl;
    std::cout << "-o=<outdir>     Output directory (must not exist)" << std::endl;
    std::cout << "-m              Enable performance monitoring (default: false)" << std::endl;
    std::cout << "-v              Enable verbose output (default: false)" << std::endl;
    std::cout << "-h              Print usage help" << std::endl;
}
