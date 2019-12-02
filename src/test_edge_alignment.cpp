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

int main( int argc, char ** argv )
{
    cout << "Hello Edge Alignment\n";

    if( argc != 3 ) {
        cout << "You only supplied argc=" << argc << endl;
        cout << "Usage: " << argv[0] << " <a_idx> <b_idx>\n";

            int a_idx = std::stoi( argv[1] );
            int b_idx = std::stoi( argv[2] );
        exit(1);
    }


    #if 1
    // load camodocam camera
    camodocal::CameraPtr cam = camodocal::CameraFactory::instance()->generateCameraFromYamlFile("/app/catkin_ws/src/gmm_pointcloud_align/resources/local_bundle/camera.yaml");
    if( !cam ) { cout << "Cannot load camera\n"; exit(2); }
    cout << cam->parametersToString() << endl;
    #endif

    LocalBundle bundle;
    bundle.fromJSON("/app/catkin_ws/src/gmm_pointcloud_align/resources/local_bundle/" );
    bundle.print_inputs_info();
    // bundle.solve();


    int a_idx = std::stoi( argv[1] );
    int b_idx = std::stoi( argv[2] );
    cout << "a_idx=" << a_idx << "\tb_idx=" << b_idx << endl;
    const cv::Mat im_ref = bundle.get_image(0, a_idx ); cout << "im_ref : " << MiscUtils::cvmat_info( im_ref ) << endl;
    const cv::Mat im_curr = bundle.get_image(1, b_idx); cout << "im_curr: " << MiscUtils::cvmat_info( im_curr ) << endl;
    const cv::Mat depth_curr = bundle.get_depthmap( 1, b_idx); cout << "depth_curr: " << MiscUtils::cvmat_info( depth_curr ) << endl;

    cv::imshow( "im_ref", im_ref );
    cv::imshow( "im_curr", im_curr );


    ElapsedTime t_eaalign("Edge Alignment");

    // Matrix4d initial_guess____ref_T_curr =     bundle.retrive_optimized_pose( 0, a_idx, 1, b_idx ); // Matrix4d::Identity();
    Matrix4d initial_guess____ref_T_curr =     bundle.retrive_initial_guess_pose( 0, a_idx, 1, b_idx ); // Matrix4d::Identity();
    // Matrix4d initial_guess____ref_T_curr =     bundle.retrive_odometry_pose( 0, a_idx, 1, b_idx ); // Matrix4d::Identity();


    Matrix4d ref_T_curr_optvar;

    EdgeAlignment ealign( cam, im_ref, im_curr, depth_curr );
    ealign.set_make_representation_image();
    bool ea_status = ealign.solve( initial_guess____ref_T_curr, ref_T_curr_optvar );
    cv::imshow( "debug_image_ealign", ealign.get_representation_image() );

    cout << "[main]" << t_eaalign.toc() << endl;
    cout << "[main] ea_status = " << ea_status << endl;
    cout << "[main] initial_guess____ref_T_curr = " << PoseManipUtils::prettyprintMatrix4d( initial_guess____ref_T_curr );
    cout << "[main] ref_T_curr_optvar = " << PoseManipUtils::prettyprintMatrix4d( ref_T_curr_optvar );
    cv::waitKey(0);

    return 0;
}
