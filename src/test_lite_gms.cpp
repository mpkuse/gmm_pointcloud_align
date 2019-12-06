// Analyse the GMS matching at each loop candidate





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


const string CAM_PREFIX = "/app/catkin_ws/src/gmm_pointcloud_align/resources/local_bundle/";
const string SAVE_LOCALBUNDLE_REPROJECTION_DEBUG_IMAGES_PREFIX = "/app/tmp/cerebro/reprojections/";

int main( int argc, char ** argv )
{
    int hyp_idx = 11;
    cout << "argc=" << argc << endl;
    if( argc != 2 ) {
        cout << "Usage: " << argv[0] << " <hyp_idx:int> \n";
        exit(1);
    }
    hyp_idx = std::stoi( argv[1] );

    cout << TermColor::bWHITE() <<  "-------------------------------------------------------\n";
    cout << "---- Hello GMS lite : hyp_idx="<< hyp_idx << "-----\n";
    cout << "-------------------------------------------------------\n" << TermColor::RESET();

    //   Load camodocal
    #if 0
    // load camodocam camera
    camodocal::CameraPtr cam = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(CAM_PREFIX+"/camera.yaml");
    if( !cam ) { cout << "Cannot load camera\n"; exit(2); }
    cout << cam->parametersToString() << endl;
    #endif



    string fname_a = SAVE_LOCALBUNDLE_REPROJECTION_DEBUG_IMAGES_PREFIX+"../live_system/hyp_" + to_string(hyp_idx) + "_left_image_a" + ".jpg";
    cv::Mat left_image_a = cv::imread( fname_a, 0 );
    string fname_b = SAVE_LOCALBUNDLE_REPROJECTION_DEBUG_IMAGES_PREFIX+"../live_system/hyp_" + to_string(hyp_idx) + "_left_image_b" + ".jpg";
    cv::Mat left_image_b = cv::imread( fname_b, 0 );
    assert( !left_image_a.empty() && !left_image_a.empty() );


    // load depth_a, depth_b
    cv::Mat depth_a, depth_b;
    string fname_storage = SAVE_LOCALBUNDLE_REPROJECTION_DEBUG_IMAGES_PREFIX+"../live_system/hyp_" + to_string(hyp_idx) + "_depths.yaml";
    cv::FileStorage storage(fname_storage, cv::FileStorage::READ);
    assert( storage.isOpened() );
    storage["depth_a"] >> depth_a;
    storage["depth_b"] >> depth_b;
    storage.release();

    cout << "Files Read:";
    cout << "\t" << fname_a << endl;
    cout << "\t" << fname_b << endl;
    cout << "\t" << fname_storage << endl;
    cout << "\n";
    cout << "left_image_a: " << MiscUtils::cvmat_info( left_image_a ) << endl;
    cout << "left_image_b: " << MiscUtils::cvmat_info( left_image_b ) << endl;
    cout << "depth_a: " << MiscUtils::cvmat_info( depth_a ) << endl;
    cout << "depth_b: " << MiscUtils::cvmat_info( depth_b ) << endl;


    cout << TermColor::bWHITE() <<  "-------------------------------------------------------\n";
    cout << "---- Data Loading Done  : hyp_idx="<< hyp_idx <<  "\n";
    cout << "-------------------------------------------------------\n" << TermColor::RESET();




    // try point feature matching
    MatrixXd uv_a, uv_b;
    ElapsedTime t_gms( "Gms Matcher");
    // StaticPointFeatureMatching::gms_point_feature_matches( left_image_a, left_image_b, uv_a, uv_b, 10000 );
    StaticPointFeatureMatching::gms_point_feature_matches_scaled( left_image_a, left_image_b, uv_a, uv_b, .5, 10000 );
    cout << "uv_a:" << uv_a.cols() << "\t" << t_gms.toc() << endl;

    // depths at correspondences
    VectorXd d_a = StaticPointFeatureMatching::depth_at_image_coordinates( uv_a, depth_a );
    VectorXd d_b = StaticPointFeatureMatching::depth_at_image_coordinates( uv_b, depth_b );

    // //--- normalized image cords
    // MatrixXd normed_uv_a = StaticPointFeatureMatching::image_coordinates_to_normalized_image_coordinates( dataManager->getAbstractCameraRef(), uv_a );
    // MatrixXd normed_uv_b = StaticPointFeatureMatching::image_coordinates_to_normalized_image_coordinates( dataManager->getAbstractCameraRef(), uv_b );
    //
    // // 3d points
    // MatrixXd aX = StaticPointFeatureMatching::normalized_image_coordinates_and_depth_to_3dpoints( normed_uv_a, d_a, true );
    // MatrixXd bX = StaticPointFeatureMatching::normalized_image_coordinates_and_depth_to_3dpoints( normed_uv_b, d_b, true );


    double near = 0.5, far = 5.0;
    vector<bool> valids_a = MiscUtils::filter_near_far( d_a, near, far );
    vector<bool> valids_b = MiscUtils::filter_near_far( d_b, near, far );
    int nvalids_a = MiscUtils::total_true( valids_a );
    int nvalids_b = MiscUtils::total_true( valids_b );
    vector<bool> valids = MiscUtils::vector_of_bool_AND( valids_a, valids_b );
    int nvalids = MiscUtils::total_true( valids );


    bool PLOT = true;
    if( PLOT ) {
        cv::Mat dst_matcher;
        string msg_str = "#matches="+to_string( uv_a.cols() )+" ";
        msg_str+= ";nvalids_a,nvalids_b,nvalids=(" + to_string(nvalids_a) + "," + to_string(nvalids_b) + "," + to_string(nvalids) + ")";
        MiscUtils::plot_point_pair( left_image_a, uv_a,
                                    left_image_b, uv_b,
                                    dst_matcher,
                                    #if 1 // make this to 1 to mark matches by spatial color codes (gms style). set this to 0 to mark the matches with lines
                                    3, msg_str
                                    #else
                                    cv::Scalar( 0,0,255 ), cv::Scalar( 0,255,0 ), false, msg_str
                                    #endif
                                );

    cv::imshow( "dst_matcher", dst_matcher );
    cv::waitKey(0);

    }
}
