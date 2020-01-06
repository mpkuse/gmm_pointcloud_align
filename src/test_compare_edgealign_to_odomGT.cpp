// Give as input two adjacent keyframe-images I try to compare the 6DOF with odometry ground truth.
// Also compare with pointbased method.


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


// Camodocal
#include "camodocal/camera_models/Camera.h"
#include "camodocal/camera_models/CameraFactory.h"


// JSON
#include "utils/nlohmann/json.hpp"
using json = nlohmann::json;

// KuseUtils
//#include "SurfelMap.h"
//#include "SlicClustering.h"
#include "utils/CameraGeometry.h"
#include "utils/PointFeatureMatching.h"


#include "PoseComputation.h"
#include "Triangulation.h"
#include "LocalBundle.h"

//
#include "utils/RosMarkerUtils.h"
#include "utils/RawFileIO.h"
#include "utils/TermColor.h"
#include "utils/ElapsedTime.h"

#include "XLoader.h"

void next_available( const json& STATE, int& a )
{
    while( true ) {
        json data_node = STATE["DataNodes"][a];
        if( data_node["isPoseAvailable"] == false || data_node["isKeyFrame"] == false ) {
            a++;
            continue;
        }
        break;
    }
}



int main( int argc, char ** argv )
{
    cout << "Hello test compare.\n";
    cout << "argc = " << argc << endl;
    if( argc != 4 ) {
        cout << "INVALID USAGE:\n";
        cout << "\t" << argv[0] << " <int:a> <int:b> <int:use_n_3dpts>\n";
        exit(1);
    }

    //
    // Load Camera (camodocal)
    //
    XLoader xloader;
    xloader.load_left_camera();
    xloader.load_right_camera();
    xloader.load_stereo_extrinsics();
    xloader.make_stereogeometry();
    auto cam = xloader.left_camera;

    //
    // Load JSON
    //
    json STATE = xloader.load_json();

    Matrix4d imu_T_cam;
    bool is_imuTcam_available = false;
    if( STATE["MiscVariables"]["isIMUCamExtrinsicAvailable"] == true )
    {
        bool status = RawFileIO::read_eigen_matrix4d_fromjson(  STATE["MiscVariables"]["imu_T_cam"] , imu_T_cam );
        cout << "imu_T_cam:\n" << imu_T_cam << endl;
        assert( status );
        is_imuTcam_available = true;
    }

    cout << "--------------------------------\n";
    cout << "STATE[\"DataNodes\"].size="<< STATE["DataNodes"].size() << endl;
    cout << "--------------------------------\n";


    // INPUT
    int a = atoi(argv[1] );// 40;
    int b = atoi(argv[2] );// 43;
    cout << TermColor::GREEN() << "a=" << a << "\tb=" << b << endl << TermColor::RESET();
    next_available( STATE, a );
    next_available( STATE, b );
    cout << TermColor::GREEN() << "using, a=" << a << "\tb=" << b << endl << TermColor::RESET();


    //
    // Load Images
    //
    json data_node_a = STATE["DataNodes"][a];
    cv::Mat image_a, depth_a;
    bool status = xloader.retrive_image_data_from_json_datanode( data_node_a, image_a, depth_a );
    const Matrix4d w_T_a = xloader.retrive_pose_from_json_datanode( data_node_a );

    json data_node_b = STATE["DataNodes"][b];
    cv::Mat image_b, depth_b;
    status = xloader.retrive_image_data_from_json_datanode( data_node_b, image_b, depth_b );
    const Matrix4d w_T_b = xloader.retrive_pose_from_json_datanode( data_node_b );

    cout << "image_a: " <<  MiscUtils::cvmat_info( image_a ) << "\t" << MiscUtils::cvmat_minmax_info( image_a ) << endl;
    cout << "depth_a: " <<  MiscUtils::cvmat_info( depth_a ) << "\t" << MiscUtils::cvmat_minmax_info( depth_a ) << endl;

    cout << "image_b: " <<  MiscUtils::cvmat_info( image_b ) << "\t" << MiscUtils::cvmat_minmax_info( image_b ) << endl;
    cout << "depth_b: " <<  MiscUtils::cvmat_info( depth_b ) << "\t" << MiscUtils::cvmat_minmax_info( depth_b ) << endl;

    cout << "w_T_a: " << PoseManipUtils::prettyprintMatrix4d( w_T_a ) << endl;
    cout << "w_T_b: " << PoseManipUtils::prettyprintMatrix4d( w_T_b ) << endl;
    Matrix4d odom_a_T_b =  w_T_a.inverse() * w_T_b;
    cout << "odom_a_T_b: " << PoseManipUtils::prettyprintMatrix4d( odom_a_T_b ) << endl;

    cv::imshow( "image_a", image_a );
    cv::imshow( "image_b", image_b );



    #if 0
    //
    // 6DOF pose with EdgeAlignment
    cv::Mat im_ref = image_a;
    cv::Mat im_curr = image_b;
    cv::Mat depth_curr = depth_b;
    EdgeAlignment ealign( cam, im_ref, im_curr, depth_curr );
    ealign.set_make_representation_image();
    Matrix4d initial_guess____ref_T_curr = Matrix4d::Identity();
    Matrix4d ref_T_curr_optvar;

    ElapsedTime t_main_ea( "ealign.solve()");
    bool ea_status = ealign.solve( initial_guess____ref_T_curr, ref_T_curr_optvar );
    cout << TermColor::uGREEN() << t_main_ea.toc() << TermColor::RESET() << endl;

    cout << "initial_guess____ref_T_curr :" << PoseManipUtils::prettyprintMatrix4d( initial_guess____ref_T_curr ) << endl;
    cout << "ref_T_curr_optvar           :" << PoseManipUtils::prettyprintMatrix4d( ref_T_curr_optvar ) << endl;
    cv::imshow( "debug_image_ealign", ealign.get_representation_image() );
    #endif


    #if 1
    //
    // 4DOF pose with EdgeAlignment
    int use_n_3dpts = std::atoi( argv[3] ) ; 200; //INPUT

    cv::Mat im_ref = image_a;
    cv::Mat im_curr = image_b;
    cv::Mat depth_curr = depth_b;

    Matrix4d initial_guess____ref_T_curr = Matrix4d::Identity();
    Matrix4d initial_guess____refimu_T_currimu =  imu_T_cam * initial_guess____ref_T_curr * imu_T_cam.inverse();

    EdgeAlignment ealign( cam, im_ref, im_curr, depth_curr, use_n_3dpts  );
    ealign.set_make_representation_image();

    ElapsedTime t_main_ea( "ealign.solve()");
    Matrix4d ref_T_curr_optvar;
    bool ea_4dof_status = ealign.solve4DOF( initial_guess____ref_T_curr,
        imu_T_cam, w_T_a, w_T_b,
        ref_T_curr_optvar );
    cout << TermColor::uGREEN() << t_main_ea.toc() << TermColor::RESET() << endl;
    cout << "[main] status = " << ea_4dof_status << ", ref_T_curr_optvar = " << PoseManipUtils::prettyprintMatrix4d(ref_T_curr_optvar) << endl;

    cv::imshow( "debug_image_ealign", ealign.get_representation_image() );



        cout << "----------------\n";
        cout << "use_n_3dpts=" << use_n_3dpts << endl;
        cout << "odom_a_T_b        :" << PoseManipUtils::prettyprintMatrix4d( odom_a_T_b ) << endl;
        cout << "ref_T_curr_optvar :" << PoseManipUtils::prettyprintMatrix4d( ref_T_curr_optvar ) << endl;
        cout << "delta             :" << PoseManipUtils::prettyprintMatrix4d( ref_T_curr_optvar * odom_a_T_b.inverse() ) << endl;
        cout << "----------------\n";

    #endif



    //
    // 6DOF pose with PointFeatureMatching+PNP


    cv::waitKey(0);

    return 0;
}
