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
// Check particle generation utilities
//
// =============================================================================

#include <cmath>
#include <cstdio>
#include <vector>

#include "chrono/ChConfig.h"
#include "chrono/utils/ChUtilsCreators.h"
#include "chrono/utils/ChUtilsGeometry.h"
#include "chrono/utils/ChUtilsInputOutput.h"
#include "chrono/utils/ChUtilsSamplers.h"

using namespace chrono;

using std::cout;
using std::endl;

int main(int argc, char* argv[]) {
    double length = 110;
    double width = 6;

    double radius_g = 0.0125;
    int num_layers = 10;

    double r = 1.01 * radius_g;
    double hdimX = length / 2;
    double hdimY = width / 2;

    ChVector<> hdims(hdimX - r, hdimY - r, 0);
    ChVector<> center(0, 0, 2 * r);

    size_t num_particles = 0;
    for (int il = 0; il < num_layers; il++) {
        utils::PDSampler<> sampler(2 * r);
        auto points = sampler.SampleBox(center, hdims);
        num_particles += points.size();
        cout << " level: " << il << " points: " << points.size() << endl;
        center.z() += 2 * r;
    }

    cout << "Domain:      " << length << " x " << width << endl;
    cout << "Radius:      " << radius_g << endl;
    cout << "Num. layers: " << num_layers << endl;
    cout << "Total number particles: " << num_particles << endl;

    return 0;
}
