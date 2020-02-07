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

#include <numeric>
#include "DlsPnpWithRansac.h"

#define XNORM( X ) ( sqrt( X[0]*X[0] + X[1]*X[1] + X[2]*X[2] ) )

void prettyPrint_stats( const string name, const std::vector<double>& v )
{
    cout << TermColor::bWHITE() << name << TermColor::RESET() << endl;
    for( int j=0 ; j<v.size() ; j++ )
        cout << j << '\t' << v[j] << endl;;
    // Eigen::Map<Eigen::VectorXd> e_v(v.data(), v.size());
    // cout << e_v.mean() << endl;

    double sum = std::accumulate(v.begin(), v.end(), 0.0);
    double mean = sum / v.size();
    double sq_sum = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / v.size() - mean * mean);
    cout << "mean=" << mean << "\tstdev=" << stdev << endl;

}

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



// outputs a_T_b
Matrix4d getPnPPose( const camodocal::CameraPtr cam,
    const cv::Mat& image_a,  const cv::Mat& depth_a,
    const cv::Mat& image_b,  const cv::Mat& depth_b
)
{
    // Point feature matrix
    // >> uv_a, uv_b
    MatrixXd uv_a, uv_b;
    #if 0 // 1 for gms, 0 for ORB
    StaticPointFeatureMatching::gms_point_feature_matches( image_a, image_b, uv_a, uv_b, 10000 );
    #else
    PointFeatureMatchingSummary tmp;
    StaticPointFeatureMatching::point_feature_matches( image_a, image_b, uv_a, uv_b, tmp );
    #endif



    // depths at these points
    // >> d_a, d_b
    VectorXd d_a = StaticPointFeatureMatching::depth_at_image_coordinates( uv_a, depth_a );
    VectorXd d_b = StaticPointFeatureMatching::depth_at_image_coordinates( uv_b, depth_b );

    // 3d points
    // >> aX, bX
    // //--- normalized image cords
    MatrixXd normed_uv_a = StaticPointFeatureMatching::image_coordinates_to_normalized_image_coordinates( cam, uv_a );
    MatrixXd normed_uv_b = StaticPointFeatureMatching::image_coordinates_to_normalized_image_coordinates( cam, uv_b );
    //
    // // 3d points
    MatrixXd aX = StaticPointFeatureMatching::normalized_image_coordinates_and_depth_to_3dpoints( normed_uv_a, d_a, true );
    MatrixXd bX = StaticPointFeatureMatching::normalized_image_coordinates_and_depth_to_3dpoints( normed_uv_b, d_b, true );


    // filter 3d points based on far and near
    double near = 0.5, far = 5.0;
    vector<bool> valids_a = MiscUtils::filter_near_far( d_a, near, far );
    vector<bool> valids_b = MiscUtils::filter_near_far( d_b, near, far );
    int nvalids_a = MiscUtils::total_true( valids_a );
    int nvalids_b = MiscUtils::total_true( valids_b );
    vector<bool> valids = MiscUtils::vector_of_bool_AND( valids_a, valids_b );
    int nvalids = MiscUtils::total_true( valids );


    bool PLOT = false;
    if( PLOT ) {
    cv::Mat dst_matcher;
    string msg_str = "#matches="+to_string( uv_a.cols() )+"  nvalids="+to_string(nvalids);
    MiscUtils::plot_point_pair( image_a, uv_a, 0,
                                image_b, uv_b, 0,
                                dst_matcher,
                                #if 0 // make this to 1 to mark matches by spatial color codes (gms style). set this to 0 to mark the matches with lines
                                3, msg_str
                                #else
                                cv::Scalar( 0,0,255 ), cv::Scalar( 0,255,0 ), false, msg_str
                                #endif
                            );

    cv::imshow( "dst_matcher", dst_matcher );
    }


    // a_T_b = PNP( uv_a, bX )
    //          or
    // b_T_a = PNP( uv_b, aX )
    std::vector<Vector3d> w_X;
    std::vector<Vector2d> c_uv_normalized;
    for( int i=0 ; i<uv_a.cols() ; i++ )
    {
        if( valids[i] == false )
            continue;

        Vector3d _X;
        Vector2d _uv;
        _X << bX(0,i), bX(1,i), bX(2,i);
        _uv << normed_uv_a(0,i) , normed_uv_a(1,i);

        w_X.push_back( _X );
        c_uv_normalized.push_back( _uv );

    }
    Matrix4d c_T_w;
    string pnp_msg;
    StaticTheiaPoseCompute::PNP(w_X, c_uv_normalized, c_T_w, pnp_msg );

    cout << "c_T_w (StaticTheiaPoseCompute::PNP): " << PoseManipUtils::prettyprintMatrix4d( c_T_w ) << endl;

    if( PLOT ) {
    cv::waitKey(0);
    }

    return c_T_w;
    cout << "EXIT....\n";
    exit(1);
}



int main( int argc, char ** argv )
{
    cout << "Hello test compare.\n";
    cout << "argc = " << argc << endl;
    if( argc != 4 ) {
        cout << "INVALID USAGE:\n";
        cout << "\t" << argv[0] << " <int:a_start> <int:a_end> <int:delta> \n";
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
    int a_start = atoi( argv[1] );
    int a_end = atoi( argv[2] );
    int delta = atoi( argv[3] );


    vector<double> all_initial_costs;
    vector<double> all_final_costs;

    vector<double> all_delta_ypr;
    vector<double> all_delta_t;

    for( int _a=a_start ; _a<a_end ; _a+=2 ) {


    int a = _a; //atoi(argv[1] );// 40;
    int b = _a+delta; //atoi(argv[2] );// 43;
    cout << TermColor::GREEN() << "a=" << a << "\tb=" << b << endl << TermColor::RESET();
    next_available( STATE, a );
    next_available( STATE, b );
    cout << TermColor::iGREEN() << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>using, a=" << a << "\tb=" << b << endl << TermColor::RESET();


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

    #if 1
    // Pose with Sparse PointFeatureMatching & PNP
    Matrix4d pnp_a_T_b = getPnPPose( cam, image_a, depth_a, image_b, depth_b );

    #endif


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
    int use_n_3dpts = 2000; //INPUT

    cv::Mat im_ref = image_a;
    cv::Mat im_curr = image_b;
    cv::Mat depth_curr = depth_b;

    Matrix4d initial_guess____ref_T_curr = Matrix4d::Identity();
    // Matrix4d initial_guess____ref_T_curr = pnp_a_T_b;
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

        auto summary = ealign.getCeresSummary();
        cout << "initial_cost=" << summary.initial_cost << endl;
        cout << "final_cost=" << summary.final_cost << endl;
        all_initial_costs.push_back( summary.initial_cost );
        all_final_costs.push_back( summary.final_cost );


        // Matrix4d delta_pose = ref_T_curr_optvar * odom_a_T_b.inverse();
        Matrix4d delta_pose = pnp_a_T_b * odom_a_T_b.inverse();
        double delta_ypr[5], delta_t[5];
        PoseManipUtils::eigenmat_to_rawyprt( delta_pose, delta_ypr, delta_t );
        all_delta_ypr.push_back( XNORM(delta_ypr) );
        all_delta_t.push_back( XNORM(delta_t) );

    #endif



    //
    // 6DOF pose with PointFeatureMatching+PNP


    char key = cv::waitKey(10);
    if( key == 'q' ) {
        cout << "BREAK\n";
        break;
    }
    } // for( _a )


    // prettyPrint_stats( "all_initial_costs", all_initial_costs );
    // prettyPrint_stats( "all_final_costs", all_final_costs );
    prettyPrint_stats( "all_delta_ypr", all_delta_ypr );
    prettyPrint_stats( "all_delta_t", all_delta_t );


    return 0;
}
