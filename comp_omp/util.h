#include<iostream>

using namespace std;

#define G 0.66743//6.67430e-11
#define SOFTENING 1e-9f
#define dt 0.001

class vector3D{
public:

    double x, y, z;

    vector3D() : x(0.0), y(0.0), z(0.0) {}  // Default constructor
    vector3D(double x, double y, double z) : x(x), y(y), z(z) {}

    void print(){
        cout<<"("<<x<<", "<<y<<", "<<z<<")";
    }

    vector3D& operator+=(const vector3D& other){
        if (this == &other) return *this;
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    vector3D operator*=(const double c){
        x *= c;
        y *= c;
        z *= c;
        return *this;
    }

    vector3D operator*(const double c){
        return vector3D(x*c, y*c, z*c);
    }

    vector3D operator/(const double c){
        return vector3D(x/c, y/c, z/c);
    }

};

class body{
public:

    vector3D pos;
    vector3D v;
    double mass;

    body(){}
    body(const vector3D& p, const vector3D& v, double m){
        this->pos = p;
        this->v = v;
        this->mass = m;
    }

    vector3D getDist(const body& other) const {
        double dx = other.pos.x - pos.x;
        double dy = other.pos.y - pos.y;
        double dz = other.pos.z - pos.z;

        return vector3D{dx, dy, dz};
    }

    void update(const vector3D a){

        pos.x += v.x * dt;
        pos.y += v.y * dt;
        pos.z += v.z * dt;

        v.x += a.x * dt;
        v.y += a.y * dt;
        v.z += a.z * dt;
    }

    void updateV(const vector3D a){
        v.x += a.x * dt;
        v.y += a.y * dt;
        v.z += a.z * dt;
    }

    void updateP(){

        pos.x += v.x * dt;
        pos.y += v.y * dt;
        pos.z += v.z * dt;
    }

    string getString(){
        string strpos = "(" + to_string(pos.x) + "," + to_string(pos.y) + "," + to_string(pos.z) + ")";
        string strvel = "(" + to_string(v.x) + "," + to_string(v.y) + "," + to_string(v.z) + ")";
        return strpos + strvel;
    }
    
};