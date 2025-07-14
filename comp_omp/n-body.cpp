#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include "util.h"
#include <omp.h>  
using namespace std::chrono;
using namespace std;

#define time_steps 100000

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <input_file> <output_file>" << endl;
        return 1;
    }

    string in_file = argv[1];
    fstream fin;
    fin.open(in_file, ios::binary | ios::in);

    int N;
    fin.read(reinterpret_cast<char *>(&N), sizeof(N));

    body *bodies = new body[N];

    for (int i = 0; i < N; ++i) {
        double m;
        fin.read(reinterpret_cast<char *>(&m), sizeof(m));
        double px, py, pz, vx, vy, vz;
        fin.read(reinterpret_cast<char *>(&px), sizeof(px));
        fin.read(reinterpret_cast<char *>(&py), sizeof(py));
        fin.read(reinterpret_cast<char *>(&pz), sizeof(pz));
        fin.read(reinterpret_cast<char *>(&vx), sizeof(vx));
        fin.read(reinterpret_cast<char *>(&vy), sizeof(vy));
        fin.read(reinterpret_cast<char *>(&vz), sizeof(vz));
        bodies[i] = body(vector3D(px, py, pz), vector3D(vx, vy, vz), m);
    }
    fin.close();

    auto start = high_resolution_clock::now();

    cout << "Initial Info " << endl;
    for (int i = 0; i < N; ++i) {
        cout << "body" << i << ": mass=" << bodies[i].mass << ", pos=("
                 << bodies[i].pos.x << ", " << bodies[i].pos.y << ", "
                 << bodies[i].pos.z << "); ";
        cout << "vel=(" << bodies[i].v.x << ", " << bodies[i].v.y << ", "
                 << bodies[i].v.z << ")\n";
    }

    // **********************************
    // *  DONT'T MODIFY THE CODE ABOVE  *
    // **********************************
 for (int t = 0; t < time_steps; ++t) {
        vector3D *a = new vector3D[N]();

        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            vector3D ai = vector3D();  // ai 是一個臨時變量
            for (int j = 0; j < N; ++j) {
                if (j != i) {
                    body other = bodies[j];
                    vector3D d = bodies[i].getDist(other);
                    double r = sqrt(d.x * d.x + d.y * d.y + d.z * d.z + SOFTENING);
                    double r3 = r * r * r;
                    ai += d * (other.mass) * G / r3;
                }
            }
            a[i] = ai;  // 最後將計算結果存入 a[i]
        }

        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            bodies[i].update(a[i]);
        }

        delete[] a;
    }
    // **********************************
    // *  DONT'T MODIFY THE CODE BELOW  *
    // **********************************

    cout << "=========================================" << endl;
    cout << "results after " << time_steps * dt << " sec: " << endl;
    for (int i = 0; i < N; ++i) {
        cout << "body" << i << ": pos=(" << bodies[i].pos.x << ", "
                 << bodies[i].pos.y << ", " << bodies[i].pos.z << "); ";
        cout << "vel=(" << bodies[i].v.x << ", " << bodies[i].v.y << ", "
                 << bodies[i].v.z << ")\n";
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "================" << "time elapsed: " << duration.count() / 10e6
             << " sec=========================" << endl;

    if (argc == 3) {
        string out_file = argv[2];
        fstream fout;

        fout.open(out_file, ios::binary | ios::out);
        fout.write(reinterpret_cast<const char *>(&N), sizeof(N));
        for (int i = 0; i < N; ++i) {
            fout.write(reinterpret_cast<const char *>(&bodies[i]), sizeof(bodies[i]));
        }

        fout.close();
    }

    delete[] bodies;
    return 0;
}
