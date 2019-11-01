#include <iostream>
#include <vector>
#include <fstream>
#include <map>
using namespace std;

// opencv2

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/imgproc/imgproc.hpp>

// Eigen3
#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace Eigen;
#include <opencv2/core/eigen.hpp>

#include "LocalBundle.h"

int main()
{
    cout << "Hello\n";

    #if 0
    Matrix4d tmp = Matrix4d::Identity();
    json obj = RawFileIO::write_eigen_matrix_tojson( tmp );
    cout << obj << endl;


    Matrix4d res;
    RawFileIO::read_eigen_matrix4d_fromjson( obj, res );
    cout <<res << endl;
    #endif


    LocalBundle bundle;
    // bundle.odomSeqJ_fromJSON("/app/catkin_ws/src/gmm_pointcloud_align/resources/local_bundle/", 0);
    // bundle.odomSeqJ_fromJSON("/app/catkin_ws/src/gmm_pointcloud_align/resources/local_bundle/", 1);
    // bundle.matches_SeqPair_fromJSON("/app/catkin_ws/src/gmm_pointcloud_align/resources/local_bundle/", 0, 1);
    bundle.fromJSON("/app/catkin_ws/src/gmm_pointcloud_align/resources/local_bundle/" );
    bundle.print_inputs_info();
}
