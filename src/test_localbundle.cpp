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
#include "utils/ElapsedTime.h"

// Camodocal
#include "camodocal/camera_models/Camera.h"
#include "camodocal/camera_models/CameraFactory.h"

int main()
{
    cout << "Hello\n";

    #if 1
    // load camodocam camera
    camodocal::CameraPtr cam = camodocal::CameraFactory::instance()->generateCameraFromYamlFile("/app/catkin_ws/src/gmm_pointcloud_align/resources/local_bundle/camera.yaml");
    if( !cam ) { cout << "Cannot load camera\n"; exit(2); }
    cout << cam->parametersToString() << endl;
    #endif

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

    ElapsedTime tp("Bundle Solver");
    // bundle.solve();
    bundle.solve_ea(cam, false);
    cout << "[main]" <<  tp.toc() << endl;

    bundle.retrive_optimized_pose( 0, 0, 1, 0 );

    #if 0
    bundle.reprojection_test(cam);
    bundle.reprojection_error( cam );
    #else
    int n_im_pairs = bundle.get_image_pairs_len(0,1);
    for(int h=0 ; h<n_im_pairs; h++ )
    {
        cv::Mat _dst_observed_correspondence_, _dst_image_a, _dst_image_b;
        bundle.reprojection_debug_images_for_this_image_pair( cam, h,
            _dst_observed_correspondence_, _dst_image_a, _dst_image_b );

        MiscUtils::imshow( "_dst_observed_correspondence_", _dst_observed_correspondence_ , 0.5 );
        MiscUtils::imshow( "_dst_image_a", _dst_image_a , 0.5 );
        MiscUtils::imshow( "_dst_image_b", _dst_image_b , 0.5 );
        cv::waitKey(0);
    }
    #endif

    bundle.edgealignment_debug_images_to_disk( "/app/tmp/hyp_x_");
    bundle.debug_print_initial_and_final_poses( "/app/tmp/hyp_x_" );


}
