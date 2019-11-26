#include <iostream>
#include <vector>
#include <fstream>
#include <map>
using namespace std;

// opencv2
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// Eigen3
#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace Eigen;
#include <opencv2/core/eigen.hpp>

#include "utils/PointFeatureMatching.h"
#include "LocalBundle.h"
#include "EdgeAlignment.h"

#include "utils/TermColor.h"
#include "utils/ElapsedTime.h"

#include <Eigen/Sparse>

// Camodocal
#include "camodocal/camera_models/Camera.h"
#include "camodocal/camera_models/CameraFactory.h"

int main()
{
    cout << "Hello Edge Alignment\n";


    #if 1
    // load camodocam camera
    camodocal::CameraPtr cam = camodocal::CameraFactory::instance()->generateCameraFromYamlFile("/app/catkin_ws/src/gmm_pointcloud_align/resources/local_bundle/camera.yaml");
    if( !cam ) { cout << "Cannot load camera\n"; exit(2); }
    cout << cam->parametersToString() << endl;
    #endif

    LocalBundle bundle;
    bundle.fromJSON("/app/catkin_ws/src/gmm_pointcloud_align/resources/local_bundle/" );
    bundle.print_inputs_info();
    bundle.solve();


    const cv::Mat im_ref = bundle.get_image(0, 0 ); cout << "im_ref : " << MiscUtils::cvmat_info( im_ref ) << endl;
    const cv::Mat im_curr = bundle.get_image(1, 0); cout << "im_curr: " << MiscUtils::cvmat_info( im_curr ) << endl;
    const cv::Mat depth_curr = bundle.get_depthmap( 1, 0); cout << "depth_curr: " << MiscUtils::cvmat_info( depth_curr ) << endl;

    cv::imshow( "im_ref", im_ref );
    cv::imshow( "im_curr", im_curr );


    EdgeAlignment ealign( cam, im_ref, im_curr, depth_curr );

    Matrix4d initial_guess____ref_T_curr =     bundle.retrive_optimized_pose( 0, 0, 1, 0 ); // Matrix4d::Identity();
    ealign.solve( initial_guess____ref_T_curr );

    cv::waitKey(0);

    return 0;
}
