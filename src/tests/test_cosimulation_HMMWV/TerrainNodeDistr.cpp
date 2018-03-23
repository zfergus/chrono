// =============================================================================
// PROJECT CHRONO - http://projectchrono.org
//
// Copyright (c) 2015 projectchrono.org
// All right reserved.
//
// Use of this source code is governed by a BSD-style license that can be found
// in the LICENSE file at the top level of the distribution and at
// http://projectchrono.org/license-chrono.txt.
//
// =============================================================================
// Authors: Radu Serban
// =============================================================================
//
// Definition of the TERRAIN NODE.
//
// The global reference frame has Z up, X towards the front of the vehicle, and
// Y pointing to the left.
//
// =============================================================================

#include <algorithm>
#include <cmath>
#include <vector>
#include <set>
#include <unordered_map>

#include "chrono/ChConfig.h"
#include "chrono/assets/ChLineShape.h"
#include "chrono/geometry/ChLineBezier.h"

#ifdef CHRONO_OPENGL
#include "chrono_opengl/ChOpenGLWindow.h"
#endif

#include "TerrainNodeDistr.h"

using std::cout;
using std::endl;

using namespace chrono;

const std::string TerrainNodeDistr::m_checkpoint_filename = "checkpoint.dat";

// -----------------------------------------------------------------------------
// Free functions in the cosim namespace
// -----------------------------------------------------------------------------

namespace cosim {

static MPI_Comm terrain_comm = MPI_COMM_NULL;

int Initialize(int num_tires) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (world_size < 2 + num_tires) {
        return MPI_ERR_OTHER;
    }

    // Get the group for MPI_COMM_WORLD
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    // Set list of excluded ranks (vehicle and tire nodes)
    std::vector<int> excluded;
    excluded.push_back(VEHICLE_NODE_RANK);
    for (int i = 0; i < num_tires; i++)
        excluded.push_back(TIRE_NODE_RANK(i));

    // Create the group of ranks for terrain simulation
    MPI_Group terrain_group;
    MPI_Group_excl(world_group, 1 + num_tires, excluded.data(), &terrain_group);

    // Create and return a communicator from the terrain group
    MPI_Comm_create(MPI_COMM_WORLD, terrain_group, &terrain_comm);

    return MPI_SUCCESS;
}

bool IsInitialized() {
    return terrain_comm != MPI_COMM_NULL;
}

MPI_Comm GetTerrainIntracommunicator() {
    return terrain_comm;
}

}  // end namespace cosim

// -----------------------------------------------------------------------------
// Construction of the terrain node:
// - create the (distributed) Chrono system and set solver parameters
// - create the OpenGL visualization window
// -----------------------------------------------------------------------------
TerrainNodeDistr::TerrainNodeDistr(MPI_Comm terrain_comm, int num_tires, bool render, int num_threads)
    : BaseNode("TERRAIN"),
      m_num_tires(num_tires),
      m_render(render),
      m_constructed(false),
      m_settling_output(false),
      m_num_particles(0),
      m_particles_start_index(0) {
    MPI_Comm_rank(MPI_COMM_WORLD, &m_world_rank);
    if (cosim::IsInitialized()) {
        MPI_Comm_rank(terrain_comm, &m_terrain_rank);
        cout << "Terrain process:  " << m_world_rank << " (" << m_terrain_rank << ")" << endl;
    } else if (m_world_rank == TERRAIN_NODE_RANK) {
        cout << "Co-simulation framework not initialized!" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // ------------------------
    // Default model parameters
    // ------------------------

    // Default platform and container dimensions
    m_hdimX = 1.0;
    m_hdimY = 0.25;
    m_hdimZ = 0.5;

    // Default granular material properties
    m_radius_g = 0.01;
    m_rho_g = 2000;
    m_num_layers = 5;
    m_time_settling = 0.4;

    // Default proxy body properties
    m_fixed_proxies = false;
    m_mass_pF = 1;

    // Default terrain contact material
    m_material_terrain = std::make_shared<ChMaterialSurfaceSMC>();

    // ------------------------------------
    // Create the Chrono distributed system
    // ------------------------------------

    // Create system and set default method-specific solver settings
    m_system = new ChSystemDistributed(terrain_comm, 2 * m_radius_g, 10000);
    m_system->Set_G_acc(ChVector<>(0, 0, m_gacc));

    m_prefix = "[Terrain node]";
    if (OnMaster()) {
        cout << m_prefix << " num_threads = " << num_threads << endl;
    }

    // SCM contact settings
    m_system->GetSettings()->solver.contact_force_model = ChSystemSMC::Hertz;
    m_system->GetSettings()->solver.tangential_displ_mode = ChSystemSMC::TangentialDisplacementModel::OneStep;
    m_system->GetSettings()->solver.use_material_properties = true;

    // Solver settings
    m_system->GetSettings()->perform_thread_tuning = false;
    m_system->GetSettings()->solver.use_full_inertia_tensor = false;
    m_system->GetSettings()->solver.tolerance = 0.1;
    m_system->GetSettings()->solver.max_iteration_bilateral = 100;
    m_system->GetSettings()->collision.narrowphase_algorithm = NarrowPhaseType::NARROWPHASE_HYBRID_MPR;

    // Set number of threads
    m_system->SetParallelThreadNumber(num_threads);
    CHOMPfunctions::SetNumThreads(num_threads);

#pragma omp parallel
#pragma omp master
    {
        // Sanity check: print number of threads in a parallel region
        cout << m_prefix << " actual number of OpenMP threads: " << omp_get_num_threads() << endl;
    }

#ifdef CHRONO_OPENGL
    // -------------------------------
    // Create the visualization window
    // -------------------------------

    // Render only on the "master" rank
    m_render = m_render && OnMaster();

    if (m_render) {
        opengl::ChOpenGLWindow& gl_window = opengl::ChOpenGLWindow::getInstance();
        gl_window.Initialize(1280, 720, "Terrain Node", m_system);
        gl_window.SetCamera(ChVector<>(0, -4, 0), ChVector<>(0, 0, 0), ChVector<>(0, 0, 1), 0.05f);
        gl_window.SetRenderMode(opengl::WIREFRAME);
    }
#endif

    // Reserve space for tire information
    m_tire_data.resize(m_num_tires);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
TerrainNodeDistr::~TerrainNodeDistr() {
    delete m_system;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void TerrainNodeDistr::SetContainerDimensions(double length, double width, double height, int split_axis) {
    m_hdimX = length / 2;
    m_hdimY = width / 2;
    m_hdimZ = height / 2;

    // Set direction of splitting
    m_system->GetDomain()->SetSplitAxis(split_axis);

#ifdef CHRONO_OPENGL
    if (m_render) {
        opengl::ChOpenGLWindow& gl_window = opengl::ChOpenGLWindow::getInstance();
        gl_window.SetCamera(ChVector<>(0, -m_hdimY - 1, 0), ChVector<>(0, 0, 0), ChVector<>(0, 0, 1), 0.05f);
    }
#endif
}

void TerrainNodeDistr::SetPath(std::shared_ptr<ChBezierCurve> path) {
    m_path = path;
}

void TerrainNodeDistr::SetGranularMaterial(double radius, double density, int num_layers) {
    m_radius_g = radius;
    m_rho_g = density;
    m_num_layers = num_layers;
    m_system->GetSettings()->collision.collision_envelope = 0.1 * radius;
}

void TerrainNodeDistr::UseMaterialProperties(bool flag) {
    assert(m_system->GetContactMethod() == ChMaterialSurface::SMC);
    m_system->GetSettings()->solver.use_material_properties = flag;
}

void TerrainNodeDistr::SetContactForceModel(ChSystemSMC::ContactForceModel model) {
    assert(m_system->GetContactMethod() == ChMaterialSurface::SMC);
    m_system->GetSettings()->solver.contact_force_model = model;
}

void TerrainNodeDistr::SetMaterialSurface(const std::shared_ptr<ChMaterialSurfaceSMC>& mat) {
    assert(m_system->GetContactMethod() == ChMaterialSurface::SMC);
    m_material_terrain = mat;
}

void TerrainNodeDistr::SetProxyProperties(double mass, bool fixed) {
    m_mass_pF = mass;
    m_fixed_proxies = fixed;
}

// -----------------------------------------------------------------------------
// Complete construction of the mechanical system.
// This function is invoked automatically from Settle and Initialize.
// - adjust system settings
// - create the container body
// - if specified, create the granular material
// -----------------------------------------------------------------------------
void TerrainNodeDistr::Construct() {
    if (m_constructed)
        return;

    // Inflated particle radius
    double r = 1.01 * m_radius_g;

    // Domain decomposition
    double height = std::max((1 + m_num_layers) * 2 * r, 2 * m_hdimZ);
    m_system->GetDomain()->SetSimDomain(-m_hdimX, +m_hdimX, -m_hdimY, +m_hdimY, 0, height);

    // Estimates for number of bins for broad-phase.
    int factor = 2;
    ChVector<> sub_hi = m_system->GetDomain()->GetSubHi();
    ChVector<> sub_lo = m_system->GetDomain()->GetSubLo();
    ChVector<> sub_hdim = (sub_hi - sub_lo) / 2;
    int binsX = (int)std::ceil(sub_hdim.x() / m_radius_g) / factor;
    int binsY = (int)std::ceil(sub_hdim.y() / m_radius_g) / factor;
    int binsZ = 1;
    m_system->GetSettings()->collision.bins_per_axis = vec3(binsX, binsY, binsZ);
    if (OnMaster()) {
        cout << m_prefix << " broad-phase bins: " << binsX << " x " << binsY << " x " << binsZ << endl;
    }

    // ----------------------------------------------------
    // Create the container body and the collision boundary
    // ----------------------------------------------------

    auto container = std::shared_ptr<ChBody>(m_system->NewBody());
    container->SetIdentifier(-1);
    container->SetMass(1);
    container->SetBodyFixed(true);
    container->SetCollide(false);
    container->SetMaterialSurface(m_material_terrain);
    m_system->AddBodyAllRanks(container);

    m_boundary = std::make_shared<ChBoundary>(container);
    m_boundary->AddPlane(ChFrame<>(ChVector<>(0, 0, 0), QUNIT), ChVector2<>(2 * m_hdimX, 2 * m_hdimY));
    m_boundary->AddPlane(ChFrame<>(ChVector<>(+m_hdimX, 0, m_hdimZ), Q_from_AngY(-CH_C_PI_2)),
                         ChVector2<>(2 * m_hdimZ, 2 * m_hdimY));
    m_boundary->AddPlane(ChFrame<>(ChVector<>(-m_hdimX, 0, m_hdimZ), Q_from_AngY(+CH_C_PI_2)),
                         ChVector2<>(2 * m_hdimZ, 2 * m_hdimY));
    m_boundary->AddPlane(ChFrame<>(ChVector<>(0, +m_hdimY, m_hdimZ), Q_from_AngX(+CH_C_PI_2)),
                         ChVector2<>(2 * m_hdimX, 2 * m_hdimZ));
    m_boundary->AddPlane(ChFrame<>(ChVector<>(0, -m_hdimY, m_hdimZ), Q_from_AngX(-CH_C_PI_2)),
                         ChVector2<>(2 * m_hdimX, 2 * m_hdimZ));

    m_boundary->AddVisualization(0, 2 * m_radius_g);

    // Add path as visualization asset to the container body
    if (m_path) {
        auto path_asset = std::make_shared<ChLineShape>();
        path_asset->SetLineGeometry(std::make_shared<geometry::ChLineBezier>(m_path));
        path_asset->SetColor(ChColor(0.0f, 0.8f, 0.0f));
        path_asset->SetName("path");
        container->AddAsset(path_asset);
    }

    // Enable deactivation of bodies that exit a specified bounding box.
    // We set this bounding box to encapsulate the container with a conservative height.
    m_system->GetSettings()->collision.use_aabb_active = true;
    m_system->GetSettings()->collision.aabb_min = real3(-m_hdimX, -m_hdimY, 0);
    m_system->GetSettings()->collision.aabb_max = real3(+m_hdimX, +m_hdimY, 2 * m_hdimZ + 2);

    // --------------------------
    // Generate granular material
    // --------------------------

    // Granular material properties.
    m_Id_g = 100000;

    // Cache the number of bodies that have been added so far to the parallel system.
    // ATTENTION: This will be used to set the state of granular material particles if
    // initializing them from a checkpoint file.
    //// TODO:  global or local index?
    m_particles_start_index = m_system->data_manager->num_rigid_bodies;

    // Create particles
    // Create a particle generator and a mixture entirely made out of spheres
    utils::Generator gen(m_system);
    std::shared_ptr<utils::MixtureIngredient> m1 = gen.AddMixtureIngredient(utils::SPHERE, 1.0);
    m1->setDefaultMaterial(m_material_terrain);
    m1->setDefaultDensity(m_rho_g);
    m1->setDefaultSize(m_radius_g);

    // Set starting value for body identifiers
    gen.setBodyIdentifier(m_Id_g);

    //// TODO: remove this barrier
    MPI_Barrier(m_system->GetCommunicator());

    // Create particles in layers until reaching the desired number of particles
    ChVector<> hdims(m_hdimX - r, m_hdimY - r, 0);
    ChVector<> center(0, 0, 2 * r);

    for (int il = 0; il < m_num_layers; il++) {
        gen.createObjectsBox(utils::REGULAR_GRID, 2 * r, center, hdims);
        cout << m_terrain_rank << " level: " << il << " points: " << gen.getTotalNumBodies() << endl;
        center.z() += 2 * r;
    }

    // Get total number of particles (global)
    m_num_particles = gen.getTotalNumBodies();

    if (OnMaster()) {
        cout << m_prefix << " Generated particles:  " << m_num_particles << endl;
    }

    cout << m_prefix << " LocalRank: " << m_terrain_rank << " Local num. particles: " << m_system->GetNumBodies() << endl;

    //// TODO: remove this barrier
    MPI_Barrier(m_system->GetCommunicator());

    // --------------------------------------
    // Write file with terrain node settings
    // --------------------------------------

    if (OnMaster()) {
        std::ofstream outf;
        outf.open(m_node_out_dir + "/settings.dat", std::ios::out);
        outf << "System settings" << endl;
        outf << "   Integration step size = " << m_step_size << endl;
        outf << "   Contact method = SMC" << endl;
        outf << "   Use material properties? " << (m_system->GetSettings()->solver.use_material_properties ? "YES" : "NO")
            << endl;
        outf << "   Collision envelope = " << m_system->GetSettings()->collision.collision_envelope << endl;
        outf << "Container dimensions" << endl;
        outf << "   X = " << 2 * m_hdimX << "  Y = " << 2 * m_hdimY << "  Z = " << 2 * m_hdimZ << endl;
        outf << "Terrain material properties" << endl;
        auto mat = std::static_pointer_cast<ChMaterialSurfaceSMC>(m_material_terrain);
        outf << "   Coefficient of friction    = " << mat->GetKfriction() << endl;
        outf << "   Coefficient of restitution = " << mat->GetRestitution() << endl;
        outf << "   Young modulus              = " << mat->GetYoungModulus() << endl;
        outf << "   Poisson ratio              = " << mat->GetPoissonRatio() << endl;
        outf << "   Adhesion force             = " << mat->GetAdhesion() << endl;
        outf << "   Kn = " << mat->GetKn() << endl;
        outf << "   Gn = " << mat->GetGn() << endl;
        outf << "   Kt = " << mat->GetKt() << endl;
        outf << "   Gt = " << mat->GetGt() << endl;
        outf << "Granular material properties" << endl;
        outf << "   particle radius  = " << m_radius_g << endl;
        outf << "   particle density = " << m_rho_g << endl;
        outf << "   number layers    = " << m_num_layers << endl;
        outf << "   number particles = " << m_num_particles << endl;
        outf << "Proxy body properties" << endl;
        outf << "   proxies fixed? " << (m_fixed_proxies ? "YES" : "NO") << endl;
        outf << "   proxy mass = " << m_mass_pF << endl;
    }

    // Mark system as constructed.
    m_constructed = true;

    m_system->SetupInitial();
}

// -----------------------------------------------------------------------------
// Settling phase for the terrain node
// - if not already done, complete system construction
// - simulate granular material to settle or read from checkpoint
// - record height of terrain
// -----------------------------------------------------------------------------
void TerrainNodeDistr::Settle(bool use_checkpoint) {
    Construct();

    if (use_checkpoint) {

        ////
        //// TODO: what can we do about checkpointing w/ Chrono::Distributed?
        ////

        // ------------------------------------------------
        // Initialize granular terrain from checkpoint file
        // ------------------------------------------------

        // Open input file stream
        std::string checkpoint_filename = m_out_dir + "/" + m_checkpoint_filename;
        std::ifstream ifile(checkpoint_filename);
        std::string line;

        // Read and discard line with current time
        std::getline(ifile, line);

        // Read number of particles in checkpoint
        unsigned int num_particles;
        {
            std::getline(ifile, line);
            std::istringstream iss(line);
            iss >> num_particles;

            if (num_particles != m_num_particles && OnMaster()) {
                cout << "ERROR: inconsistent number of particles in checkpoint file!" << endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        // Read granular material state from checkpoint
        for (int ib = m_particles_start_index; ib < m_system->Get_bodylist().size(); ++ib) {
            std::getline(ifile, line);
            std::istringstream iss(line);
            int identifier;
            ChVector<> pos;
            ChQuaternion<> rot;
            ChVector<> pos_dt;
            ChQuaternion<> rot_dt;
            iss >> identifier >> pos.x() >> pos.y() >> pos.z() >> rot.e0() >> rot.e1() >> rot.e2() >> rot.e3() >>
                pos_dt.x() >> pos_dt.y() >> pos_dt.z() >> rot_dt.e0() >> rot_dt.e1() >> rot_dt.e2() >> rot_dt.e3();

            auto body = m_system->Get_bodylist()[ib];
            assert(body->GetIdentifier() == identifier);
            body->SetPos(ChVector<>(pos.x(), pos.y(), pos.z()));
            body->SetRot(ChQuaternion<>(rot.e0(), rot.e1(), rot.e2(), rot.e3()));
            body->SetPos_dt(ChVector<>(pos_dt.x(), pos_dt.y(), pos_dt.z()));
            body->SetRot_dt(ChQuaternion<>(rot_dt.e0(), rot_dt.e1(), rot_dt.e2(), rot_dt.e3()));
        }

        if (OnMaster()) {
            cout << m_prefix << " read checkpoint <=== " << checkpoint_filename
                 << "   num. particles = " << num_particles << endl;
        }

    } else {
        // -------------------------------------
        // Simulate settling of granular terrain
        // -------------------------------------
        double output_fps = 100;
        int sim_steps = (int)std::ceil(m_time_settling / m_step_size);
        int output_steps = (int)std::ceil(1 / (output_fps * m_step_size));
        int output_frame = 0;

        for (int is = 0; is < sim_steps; is++) {
            // Advance step
            m_timer.reset();
            m_timer.start();
            m_system->DoStepDynamics(m_step_size);
            m_timer.stop();
            m_cum_sim_time += m_timer();

            if (OnMaster()) {
                cout << '\r' << std::fixed << std::setprecision(6) << m_system->GetChTime() << "  ["
                    << m_timer.GetTimeSeconds() << "]" << std::flush;

                // Output (if enabled)
                if (m_settling_output && is % output_steps == 0) {
                    char filename[100];
                    sprintf(filename, "%s/settling_%04d.dat", m_node_out_dir.c_str(), output_frame + 1);
                    utils::CSV_writer csv(" ");
                    WriteParticleInformation(csv);
                    csv.write_to_file(filename);
                    output_frame++;
                }
            }

#ifdef CHRONO_OPENGL
            // OpenGL rendering
            if (m_render) {
                opengl::ChOpenGLWindow& gl_window = opengl::ChOpenGLWindow::getInstance();
                if (gl_window.Active()) {
                    gl_window.Render();
                } else {
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
            }
#endif
        }

        if (OnMaster()) {
            cout << m_prefix << " settling time = " << m_cum_sim_time << endl;
        }

        m_cum_sim_time = 0;
    }
}

// -----------------------------------------------------------------------------
// Initialization of the terrain node:
// - if not already done, complete system construction
// - send information on terrain height
// - receive information on tire mesh topology (number vertices and triangles)
// - receive tire contact material properties and create the "tire" material
// - create the appropriate proxy bodies (state not set yet)
// -----------------------------------------------------------------------------
void TerrainNodeDistr::Initialize() {
    Construct();

    // Find "height" of granular material

    ////
    //// TODO: clean up this ChsystemDistributed function
    ////       - here we do not need an AllReduce
    ////       - can we / should we be able to "filter" what bodies we consider in this operation?
    ////

    double init_height = m_system->GetHighestZ();
    init_height += m_radius_g;

    // Reset system time
    m_system->SetChTime(0);

    // Send information for initial vehicle position
    if (OnMaster()) {
        double init_dim[2] = {init_height, m_hdimX};
        MPI_Send(init_dim, 2, MPI_DOUBLE, VEHICLE_NODE_RANK, 0, MPI_COMM_WORLD);

        cout << m_prefix << " Sent initial terrain height = " << init_dim[0] << endl;
        cout << m_prefix << " Sent container half-length = " << init_dim[1] << endl;
    }

#ifdef CHRONO_OPENGL
    // Move OpenGL camera
    if (m_render) {
        opengl::ChOpenGLWindow& gl_window = opengl::ChOpenGLWindow::getInstance();
        gl_window.SetCamera(ChVector<>(0, -m_hdimY - 1, 1), ChVector<>(-m_hdimX, 0, 0), ChVector<>(0, 0, 1), 0.05f);
    }
#endif

    // Loop over all tires, receive information, create proxies.
    unsigned int start_tri_index = 0;

    for (int which = 0; which < m_num_tires; which++) {
        // Receive tire contact surface specification.
        unsigned int surf_props[2];

        if (OnMaster()) {
            MPI_Status status_p;
            MPI_Recv(surf_props, 2, MPI_UNSIGNED, TIRE_NODE_RANK(which), 0, MPI_COMM_WORLD, &status_p);
            cout << m_prefix << " Received vertices = " << surf_props[0] << " triangles = " << surf_props[1] << endl;
        }

        // Broadcast to intra-communicator
        MPI_Bcast(surf_props, 2, MPI_UNSIGNED, m_system->GetMasterRank(), m_system->GetCommunicator());

        m_tire_data[which].m_num_vert = surf_props[0];
        m_tire_data[which].m_num_tri = surf_props[1];

        m_tire_data[which].m_vertex_pos.resize(surf_props[0]);
        m_tire_data[which].m_vertex_vel.resize(surf_props[0]);
        m_tire_data[which].m_triangles.resize(surf_props[1]);
        m_tire_data[which].m_gids.resize(surf_props[1]);

        m_tire_data[which].m_start_tri = start_tri_index;
        start_tri_index += surf_props[1];

        // Receive tire contact material properties.
        // Create the "tire" contact material, but defer using it until the proxy bodies are created.
        float mat_props[8];

        if (OnMaster()) {
            MPI_Status status_m;
            MPI_Recv(mat_props, 8, MPI_FLOAT, TIRE_NODE_RANK(which), 0, MPI_COMM_WORLD, &status_m);
            cout << m_prefix << " received tire material:  friction = " << mat_props[0] << endl;
        }

        // Broadcast to intra-communicator
        MPI_Bcast(mat_props, 8, MPI_FLOAT, m_system->GetMasterRank(), m_system->GetCommunicator());

        // Properties for tire
        auto mat_tire = std::make_shared<ChMaterialSurfaceSMC>();
        mat_tire->SetFriction(mat_props[0]);
        mat_tire->SetRestitution(mat_props[1]);
        mat_tire->SetYoungModulus(mat_props[2]);
        mat_tire->SetPoissonRatio(mat_props[3]);
        mat_tire->SetKn(mat_props[4]);
        mat_tire->SetGn(mat_props[5]);
        mat_tire->SetKt(mat_props[6]);
        mat_tire->SetGt(mat_props[7]);

        // Create proxy bodies. Represent the tire as triangles associated with mesh faces.
        CreateFaceProxies(which, mat_tire);
    }
}

// Create bodies with triangular contact geometry as proxies for the tire mesh faces.
// Assign to each body an identifier equal to the index of its corresponding mesh face.
// Maintain a list of all bodies associated with the tire.
// Add all proxy bodies to the same collision family and disable collision between any
// two members of this family.
void TerrainNodeDistr::CreateFaceProxies(int which, std::shared_ptr<ChMaterialSurfaceSMC> material) {
    //// TODO:  better approximation of mass / inertia?
    ChVector<> inertia_pF = 1e-3 * m_mass_pF * ChVector<>(0.1, 0.1, 0.1);

    for (unsigned int it = 0; it < m_tire_data[which].m_num_tri; it++) {
        auto body = std::shared_ptr<ChBody>(m_system->NewBody());
        body->SetIdentifier(m_tire_data[which].m_start_tri + it);
        body->SetMass(m_mass_pF);
        body->SetInertiaXX(inertia_pF);
        body->SetBodyFixed(m_fixed_proxies);
        body->SetMaterialSurface(material);

        // Create contact shape.
        // Note that the vertex locations will be updated at every synchronization time.
        std::string name = "tri_" + std::to_string(m_tire_data[which].m_start_tri + it);
        double len = 0.1;

        body->GetCollisionModel()->ClearModel();
        utils::AddTriangle(body.get(), ChVector<>(len, 0, 0), ChVector<>(0, len, 0), ChVector<>(0, 0, len), name);
        body->GetCollisionModel()->SetFamily(1);
        body->GetCollisionModel()->SetFamilyMaskNoCollisionWithFamily(1);
        body->GetCollisionModel()->BuildModel();

        // For Chrono::Parallel this must be done after setting family collisions
        // (in case collision is being disabled)
        body->SetCollide(true);

        m_system->AddBody(body);
    }
}

// -----------------------------------------------------------------------------
// Synchronization of the terrain node:
// - receive tire mesh vertex states and set states of proxy bodies
// - calculate current cumulative contact forces on all system bodies
// - extract and send forces at each vertex
// -----------------------------------------------------------------------------
void TerrainNodeDistr::Synchronize(int step_number, double time) {
    // --------------------------------------------------------------
    // Loop over all tires, receive mesh vertex state, update proxies
    // --------------------------------------------------------------

    for (int which = 0; which < m_num_tires; which++) {
        // Receive tire mesh vertex locations and velocities.
        MPI_Status status;
        unsigned int num_vert = m_tire_data[which].m_num_vert;
        unsigned int num_tri = m_tire_data[which].m_num_tri;
        double* vert_data = new double[2 * 3 * num_vert];
        int* tri_data = new int[3 * num_tri];

        if (OnMaster()) {
            MPI_Recv(vert_data, 2 * 3 * num_vert, MPI_DOUBLE, TIRE_NODE_RANK(which), step_number, MPI_COMM_WORLD, &status);
            MPI_Recv(tri_data, 3 * num_tri, MPI_INT, TIRE_NODE_RANK(which), step_number, MPI_COMM_WORLD, &status);
        }

        // Brodcast to intra-communicator
        MPI_Bcast(vert_data, 2 * 3 * num_vert, MPI_DOUBLE, m_system->GetMasterRank(), m_system->GetCommunicator());
        MPI_Bcast(tri_data, 3 * num_tri, MPI_INT, m_system->GetMasterRank(), m_system->GetCommunicator());

        for (unsigned int iv = 0; iv < num_vert; iv++) {
            unsigned int offset = 3 * iv;
            m_tire_data[which].m_vertex_pos[iv] =
                ChVector<>(vert_data[offset + 0], vert_data[offset + 1], vert_data[offset + 2]);
            offset += 3 * num_vert;
            m_tire_data[which].m_vertex_vel[iv] =
                ChVector<>(vert_data[offset + 0], vert_data[offset + 1], vert_data[offset + 2]);
        }

        for (unsigned int it = 0; it < num_tri; it++) {
            m_tire_data[which].m_triangles[it][0] = tri_data[3 * it + 0];
            m_tire_data[which].m_triangles[it][1] = tri_data[3 * it + 1];
            m_tire_data[which].m_triangles[it][2] = tri_data[3 * it + 2];
        }

        delete[] vert_data;
        delete[] tri_data;

        // Set position, rotation, and velocity of proxy bodies.
        UpdateFaceProxies(which);
        PrintFaceProxiesUpdateData(which);
    }

    // ------------------------------------------------------------
    // Calculate cumulative contact forces for all bodies in system
    // ------------------------------------------------------------

    m_system->CalculateContactForces();

    std::string msg =
        " step number: " + std::to_string(step_number) + "  num contacts: " + std::to_string(m_system->GetNcontacts());

    // -----------------------------------------------------------------
    // Loop over all tires, calculate vertex contact forces, send forces
    // -----------------------------------------------------------------

    msg += "  [  ";

    for (int which = 0; which < m_num_tires; which++) {
        // Collect contact forces on subset of mesh vertices.
        // Note that no forces are collected at the first step.
        std::vector<double> vert_forces;
        std::vector<int> vert_indices;

        if (step_number > 0) {
            ForcesFaceProxies(which, vert_forces, vert_indices);
        }

        //// TODO: This should only happen from the master rank
        // Send vertex indices and forces.
        int num_vert = (int)vert_indices.size();
        MPI_Send(vert_indices.data(), num_vert, MPI_INT, TIRE_NODE_RANK(which), step_number, MPI_COMM_WORLD);
        MPI_Send(vert_forces.data(), 3 * num_vert, MPI_DOUBLE, TIRE_NODE_RANK(which), step_number, MPI_COMM_WORLD);

        msg += std::to_string(num_vert) + "  ";
    }

    msg += "]";
    cout << m_prefix << msg << endl;
}

// Set position, orientation, and velocity of proxy bodies based on tire mesh faces.
// The proxy body is effectively reconstructed at each synchronization time:
//    - position at the center of mass of the three vertices
//    - orientation: identity
//    - linear and angular velocity: consistent with vertex velocities
//    - contact shape: redefined to match vertex locations
void TerrainNodeDistr::UpdateFaceProxies(int which) {
    // Traverse the information for the current tire and collect updated information.
    const TireData& tire_data = m_tire_data[which];
    std::vector<ChSystemDistributed::BodyState> states(tire_data.m_num_tri);
    std::vector<ChSystemDistributed::TriData> shapes(tire_data.m_num_tri);
    std::vector<int> shape_idx(tire_data.m_num_tri, 0);

    for (unsigned int it = 0; it < tire_data.m_num_tri; it++) {
        const ChVector<int>& tri = tire_data.m_triangles[it];

        // Vertex locations and velocities (expressed in global frame)
        const ChVector<>& pA = m_tire_data[which].m_vertex_pos[tri[0]];
        const ChVector<>& pB = m_tire_data[which].m_vertex_pos[tri[1]];
        const ChVector<>& pC = m_tire_data[which].m_vertex_pos[tri[2]];

        const ChVector<>& vA = m_tire_data[which].m_vertex_vel[tri[0]];
        const ChVector<>& vB = m_tire_data[which].m_vertex_vel[tri[1]];
        const ChVector<>& vC = m_tire_data[which].m_vertex_vel[tri[2]];

        // Position and orientation of proxy body (at triangle barycenter)
        ChVector<> pos = (pA + pB + pC) / 3;
        states[it].pos = pos;
        states[it].rot = QUNIT;

        // Linear velocity (absolute) and angular velocity (local)
        // These are the solution of an over-determined 9x6 linear system. However, for a centroidal
        // body reference frame, the linear velocity is the average of the 3 vertex velocities.
        // This leaves a 9x3 linear system for the angular velocity which should be solved in a
        // least-square sense:   Ax = b   =>  (A'A)x = A'b
        states[it].pos_dt = (vA + vB + vC) / 3;
        states[it].rot_dt = ChQuaternion<>(0, 0, 0, 0);  //// TODO: angular velocity

        // Triangle contact shape (expressed in local frame).
        shapes[it].v1 = pA - pos;
        shapes[it].v2 = pB - pos;
        shapes[it].v3 = pC - pos;
    }

    // Update body states
    m_system->SetBodyStates(tire_data.m_gids, states);

    // Update collision shapes (one triangle per collision model)
    m_system->SetTriangleShapes(tire_data.m_gids, shape_idx, shapes);
}

// Calculate barycentric coordinates (a1, a2, a3) for a given point P
// with respect to the triangle with vertices {v1, v2, v3}
ChVector<> TerrainNodeDistr::CalcBarycentricCoords(const ChVector<>& v1,
                                                   const ChVector<>& v2,
                                                   const ChVector<>& v3,
                                                   const ChVector<>& vP) {
    ChVector<> v12 = v2 - v1;
    ChVector<> v13 = v3 - v1;
    ChVector<> v1P = vP - v1;

    double d_12_12 = Vdot(v12, v12);
    double d_12_13 = Vdot(v12, v13);
    double d_13_13 = Vdot(v13, v13);
    double d_1P_12 = Vdot(v1P, v12);
    double d_1P_13 = Vdot(v1P, v13);

    double denom = d_12_12 * d_13_13 - d_12_13 * d_12_13;

    double a2 = (d_13_13 * d_1P_12 - d_12_13 * d_1P_13) / denom;
    double a3 = (d_12_12 * d_1P_13 - d_12_13 * d_1P_12) / denom;
    double a1 = 1 - a2 - a3;

    return ChVector<>(a1, a2, a3);
}

// Collect contact forces on the (face) proxy bodies that are in contact.
// Load mesh vertex forces and corresponding indices.
void TerrainNodeDistr::ForcesFaceProxies(int which, std::vector<double>& vert_forces, std::vector<int>& vert_indices) {
    // Gather contact forces on proxy bodies on the terrain master rank.
    auto force_pairs = m_system->GetBodyContactForces(m_tire_data[which].m_gids);

    /*


    std::vector<ChVector<double>> forces = m_system->GetBodyContactForces(m_tire_data[which].gids);


    struct ContactForce_Internal {
        uint gid;
        double force[3];
    };



    std::vector<std::pair<uint, ChVector<>>> frc = m_system->GetBodyContactForces(m_tire_data[which].gids);

    if (OnMaster()) {
        // Maintain an unordered map of vertex indices and associated contact forces.
        std::unordered_map<int, ChVector<>> my_map;

        // TODO: Add count to m_tire_data[which]
        // TODO: Add gid list in m_tire_data[which]
        // TODO: Track which trianlges are active and only pass those...

        // NOTE: it -- the index of a triangle in this tire in m_tire_data[which].m_triangles
        for (unsigned int it = 0; it < m_tire_data[which].m_num_tri; it++) {
            Triangle tri = m_tire_data[which].m_triangles[it];

            // Get cumulative contact force at triangle centroid.
            // TODO Better search
            real3 rforce;
            for (uint i = 0; i < count; i++) {
                if (forces[i].gid == m_tire_data[which].m_proxies[it].gid) {
                    rforce = real3(forces[i].force[0], forces[i].force[1], forces[i].force[2]);
                    break;
                }
            }

            // Do nothing if zero force.
            if (IsZero(rforce))
                continue;

            // Centroid has barycentric coordinates {1/3, 1/3, 1/3}, so force is
            // distributed equally to the three vertices.
            ChVector<> force(rforce.x / 3, rforce.y / 3, rforce.z / 3);

            // For each vertex of the triangle, if it appears in the map, increment
            // the total contact force. Otherwise, insert a new entry in the map.
            auto v1 = my_map.find(tri.v1);
            if (v1 != my_map.end()) {
                v1->second += force;
            } else {
                my_map[tri.v1] = force;
            }

            auto v2 = my_map.find(tri.v2);
            if (v2 != my_map.end()) {
                v2->second += force;
            } else {
                my_map[tri.v2] = force;
            }

            auto v3 = my_map.find(tri.v3);
            if (v3 != my_map.end()) {
                v3->second += force;
            } else {
                my_map[tri.v3] = force;
            }
        }

        // Extract map keys (indices of vertices in contact) and map values
        // (corresponding contact forces) and load output vectors.
        // Note: could improve efficiency by reserving space for vectors.
        for (auto kv : my_map) {
            vert_indices.push_back(kv.first);
            vert_forces.push_back(kv.second.x());
            vert_forces.push_back(kv.second.y());
            vert_forces.push_back(kv.second.z());
        }
    }  // End of only MASTER rank

    delete[] forces;

    
*/    
    
}

// -----------------------------------------------------------------------------
// Advance simulation of the terrain node by the specified duration
// -----------------------------------------------------------------------------
void TerrainNodeDistr::Advance(double step_size) {
    m_timer.reset();
    m_timer.start();
    double t = 0;
    while (t < step_size) {
        double h = std::min<>(m_step_size, step_size - t);
        m_system->DoStepDynamics(h);
        t += h;
    }
    m_timer.stop();
    m_cum_sim_time += m_timer();

#ifdef CHRONO_OPENGL
    if (m_render) {
        opengl::ChOpenGLWindow& gl_window = opengl::ChOpenGLWindow::getInstance();
        if (gl_window.Active()) {
            gl_window.Render();
        } else {
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
#endif

    for (int which = 0; which < m_num_tires; which++) {
        PrintFaceProxiesContactData(which);
    }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void TerrainNodeDistr::OutputData(int frame) {
    // Append to results output file
    if (m_outf.is_open()) {
        //// TODO
    }

    // Create and write frame output file.
    char filename[100];
    sprintf(filename, "%s/data_%04d.dat", m_node_out_dir.c_str(), frame + 1);

    utils::CSV_writer csv(" ");
    WriteParticleInformation(csv);
    csv.write_to_file(filename);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void TerrainNodeDistr::WriteParticleInformation(utils::CSV_writer& csv) {
    // Write current time, number of granular particles and their radius
    csv << m_system->GetChTime() << endl;
    csv << m_num_particles << m_radius_g << endl;

    // Write particle positions and linear velocities
    for (auto body : m_system->Get_bodylist()) {
        if (body->GetIdentifier() < m_Id_g)
            continue;
        csv << body->GetIdentifier() << body->GetPos() << body->GetPos_dt() << endl;
    }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void TerrainNodeDistr::WriteCheckpoint() {
    utils::CSV_writer csv(" ");

    // Write current time and number of granular material bodies.
    csv << m_system->GetChTime() << endl;
    csv << m_num_particles << endl;

    // Loop over all bodies in the system and write state for granular material bodies.
    // Filter granular material using the body identifier.
    for (auto body : m_system->Get_bodylist()) {
        if (body->GetIdentifier() < m_Id_g)
            continue;
        csv << body->GetIdentifier() << body->GetPos() << body->GetRot() << body->GetPos_dt() << body->GetRot_dt()
            << endl;
    }

    std::string checkpoint_filename = m_out_dir + "/" + m_checkpoint_filename;
    csv.write_to_file(checkpoint_filename);
    cout << m_prefix << " write checkpoint ===> " << checkpoint_filename << endl;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void TerrainNodeDistr::PrintFaceProxiesContactData(int which) {
    //// TODO: implement this
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
void TerrainNodeDistr::PrintFaceProxiesUpdateData(int which) {
    //// TODO: implement this
}
