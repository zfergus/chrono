#include "chrono/physics/ChBody.h"
#include "chrono/utils/ChUtilsSamplers.h"

using namespace chrono;

// Adds two spheres tangent to each other at the body center
void AddTangentBiSphere(ChBody* body, double radius) {
    utils::AddBiSphereGeometry(body, radius / 2.0, radius);
}

// Adds two spheres whose boundaries coincide with each other's centers
void AddBiSphere(ChBody* body, double radius) {
    utils::AddBiSphereGeometry(body, 2.0 * radius / 3.0, 2.0 * radius / 3.0);
}

// Adds a central large sphere flanked by two smaller spheres
void AddTrisphere(ChBody* body, double radius) {
    utils::AddSphereGeometry(body, 2.0 * radius / 3.0);
    utils::AddSphereGeometry(body, radius / 3.0, ChVector<>(2.0 * radius / 3.0, 0, 0));
    utils::AddSphereGeometry(body, radius / 3.0, ChVector<>(-2.0 * radius / 3.0, 0, 0));
}

// Adds a large sphere on one side of the model and a small sphere on the other
void AddAsymmetricBisphere(ChBody* body, double radius) {
    utils::AddSphereGeometry(body, 4.0 * radius / 5, ChVector<>(-radius / 5.0, 0, 0));
    utils::AddSphereGeometry(body, 2.0 * radius / 5.0, ChVector<>(3.0 * radius / 5.0, 0, 0));
}

// Adds three colinear spheres in order decreading order of size
void AddSnowman(ChBody* body, double radius) {
    utils::AddSphereGeometry(body, 4.0 * radius / 7.0, ChVector<>(-3.0 * radius / 7.0, 0, 0));
    utils::AddSphereGeometry(body, 2.0 * radius / 7.0, ChVector<>(3.0 * radius / 7.0, 0, 0));
    utils::AddSphereGeometry(body, radius / 7.0, ChVector<>(6.0 * radius / 7.0, 0, 0));
}