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
// Calculate a maximum radius over all triangles in a tire mesh
//
// =============================================================================

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include "chrono/physics/ChSystemSMC.h"

#include "chrono_vehicle/ChVehicleModelData.h"

#include "chrono_thirdparty/rapidjson/document.h"
#include "chrono_thirdparty/rapidjson/filereadstream.h"

#include "TireNode.h"

using std::cout;
using std::endl;

using namespace chrono;
using namespace rapidjson;

std::string tire_filename("hmmwv/tire/HMMWV_ANCFTire.json");
////std::string tire_filename("hmmwv/tire/HMMWV_ANCFTire_Lumped.json");
////std::string tire_filename("hmmwv/tire/HMMWV_RigidMeshTire.json");
////std::string tire_filename("hmmwv/tire/HMMWV_RigidMeshTire_Coarse.json");
////std::string tire_filename("hmmwv/tire/HMMWV_RigidMeshTire_Rough.json");

double calc_radius(const ChVector<>& P1, const ChVector<>& P2, const ChVector<>& P3) {
    
    // Radius of circumscribed circle
    ////ChVector<> a = P1 - P2;
    ////ChVector<> b = P2 - P3;
    ////ChVector<> c = P3 - P1;
    ////ChVector<> x = Vcross(a, b);    
    ////double r = (a.Length() * b.Length() * c.Length()) / (2 * x.Length());

    // Maximum distance from othocenter to a vertex
    ChVector<> G = (P1 + P2 + P3) / 3;
    double d1 = (P1 - G).Length();
    double d2 = (P2 - G).Length();
    double d3 = (P3 - G).Length();
    double r = std::max(std::max(d1, d2), d3);

    return r;
}

int main(int argc, char* argv[]) {
    std::string json_filename = vehicle::GetDataFile(tire_filename);

    // Peek in JSON file and infer tire type
    enum Type { ANCF, FEA, RIGID };
    Type type;

    FILE* fp = fopen(json_filename.c_str(), "r");

    char readBuffer[65536];
    FileReadStream is(fp, readBuffer, sizeof(readBuffer));

    fclose(fp);

    Document d;
    d.ParseStream<ParseFlag::kParseCommentsFlag>(is);

    assert(d.HasMember("Type"));
    assert(d.HasMember("Template"));
    std::string template_type = d["Type"].GetString();
    std::string template_subtype = d["Template"].GetString();
    assert(template_type.compare("Tire") == 0);

    if (template_subtype.compare("ANCFTire") == 0) {
        type = ANCF;
    } else if (template_subtype.compare("FEATire") == 0) {
        type = FEA;
    } else if (template_subtype.compare("RigidTire") == 0) {
        type = RIGID;
    }

    // Create dummy system and rim body
    ChSystemSMC sys;
    auto rim = std::shared_ptr<ChBody>(sys.NewBody());
    sys.AddBody(rim);

    // Create the tire wrapper
    TireBase* tire_wrapper;

    switch (type) {
        case ANCF:
            tire_wrapper = new TireANCF(json_filename, false);
            break;
        ////case FEA:
        ////    tire_wrapper = new TireFEA(json_filename, false);
        ////    break;
        case RIGID:
            tire_wrapper = new TireRigid(json_filename);
            break;
    }

    // Initialize the tire and obtain contact surface properties.
    std::array<int, 2> surf_props;
    std::array<float, 8> mat_props;
    tire_wrapper->Initialize(rim, vehicle::LEFT, surf_props, mat_props);

    cout << " vertices = " << surf_props[0] << "  triangles = " << surf_props[1] << endl;

    // Extract mesh connectivity
    std::vector<ChVector<>> vert_pos;
    std::vector<ChVector<>> vert_vel;  // ignored here
    std::vector<ChVector<int>> triangles;
    tire_wrapper->GetMeshState(vert_pos, vert_vel, triangles);

    // Find maximum circumscribed radius
    double r_max = 0;
    ChVector<int> t_max;
    for (auto t : triangles) {
        double r = calc_radius(vert_pos[t.x()], vert_pos[t.y()], vert_pos[t.z()]);
        if (r > r_max) {
            r_max = r;
            t_max = t;
        }
    }

    auto P1 = vert_pos[t_max.x()];
    auto P2 = vert_pos[t_max.y()];
    auto P3 = vert_pos[t_max.z()];

    cout << "Maximum radius: " << r_max << endl;
    cout << "Triangle vertices: " << endl;
    cout << "  " << P1.x() << "  " << P1.y() << "  " << P1.z() << endl; 
    cout << "  " << P2.x() << "  " << P2.y() << "  " << P2.z() << endl; 
    cout << "  " << P3.x() << "  " << P3.y() << "  " << P3.z() << endl; 

    return 0;
}
