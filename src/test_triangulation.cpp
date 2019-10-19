#include <iostream>
#include <vector>
#include <fstream>
#include <map>
using namespace std;

// opencv2
#include <opencv2/opencv.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>


// Eigen3
#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace Eigen;
#include <opencv2/core/eigen.hpp>

// #include "utils/TermColor.h"
// #include "utils/ElapsedTime.h"
// #include "utils/MiscUtils.h"


#include "Triangulation.h"


int main()
{
    cout << "Hello Triangulation\n";

    Matrix4d P = Matrix4d::Identity();
    Matrix4d P1 = Matrix4d::Identity();
    P1(0,3) = 0.1;

    auto u = Vector3d::Random();
    auto u1 = Vector3d::Random();
    Vector4d X;
    Triangulation::LinearLSTriangulation( u, P, u1, P1, X );
    cout << "Result:\n" << X << endl;
}
