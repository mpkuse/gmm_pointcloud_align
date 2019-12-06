// Analyse each of the edge alignments
//   Load camodocal
//   Load im_ref, im_curr
//   Load depth_curr
//   Load initial guess
//   EA
//   view debug image



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
const string ED_DEBUG_DATA_PREFIX = "/app/tmp/cerebro/reprojections/";

int main( int argc, char ** argv )
{
    int hyp_idx = 11;
    // int ea_idx  = 0;

    cout << "argc=" << argc << endl;

    if( argc != 2 ) {
        cout << "Usage: " << argv[0] << " <hyp_idx:int> \n";
        exit(1);
    }

    hyp_idx = std::stoi( argv[1] );
    // ea_idx =  std::stoi( argv[2] );

    cout << TermColor::bWHITE() <<  "-------------------------------------------------------\n";
    cout << "---- Hello Edge Alignment lite : hyp_idx="<< hyp_idx << "\n";
    cout << "-------------------------------------------------------\n" << TermColor::RESET();

    //   Load camodocal
    #if 1
    // load camodocam camera
    camodocal::CameraPtr cam = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(CAM_PREFIX+"/camera.yaml");
    if( !cam ) { cout << "Cannot load camera\n"; exit(2); }
    cout << cam->parametersToString() << endl;
    #endif



    //   Load im_ref, im_curr
    string im_ref_fname = ED_DEBUG_DATA_PREFIX+"hyp_"+to_string(hyp_idx)+"_ea_im_ref.jpg";
    string im_curr_fname = ED_DEBUG_DATA_PREFIX+"hyp_"+to_string(hyp_idx)+"_ea_im_curr.jpg";
    cv::Mat im_ref = cv::imread( im_ref_fname, 0  );
    cv::Mat im_curr = cv::imread( im_curr_fname, 0  );
    assert( !im_ref.empty() && !im_curr.empty() );

    //   Load depth_curr
    //   Load initial guess
    string depth_curr_fname = ED_DEBUG_DATA_PREFIX+"hyp_"+to_string(hyp_idx)+"_ea_depth_curr.yaml";
    cv::FileStorage storage( depth_curr_fname, cv::FileStorage::READ);
    assert( storage.isOpened() );
    cv::Mat depth_curr, initial_guess____ref_T_curr_opencv;
    storage["depth_curr"] >> depth_curr;
    storage["initial_guess____ref_T_curr"] >> initial_guess____ref_T_curr_opencv;
    Matrix4d initial_guess____ref_T_curr;
    cv::cv2eigen( initial_guess____ref_T_curr_opencv, initial_guess____ref_T_curr );
    cout << "initial_guess____ref_T_curr:\n" << initial_guess____ref_T_curr << endl;
    storage.release();


    // test if input are valid?
    cout << "im_ref: " << MiscUtils::cvmat_info( im_ref ) << endl;
    cout << "\t" << im_ref_fname << endl;
    cout << "im_curr: " << MiscUtils::cvmat_info( im_curr ) << endl;
    cout << "\t" << im_curr_fname << endl;
    cout << "depth_curr: " << MiscUtils::cvmat_info( depth_curr ) << endl;
    cout << "\t" << depth_curr_fname << endl;


    // Load additional info
    //  w_T_a, w_T_b (the vio poses of camera); imu_T_cam
    Matrix4d w_T_a, w_T_b, imu_T_cam;
    cv::Mat mat_w_T_a, mat_w_T_b, mat_imu_T_cam;

    string fname_additional_info =  ED_DEBUG_DATA_PREFIX+"hyp_"+to_string(hyp_idx)+"_ea_additional_info.yaml";
    cv::FileStorage storage2(fname_additional_info, cv::FileStorage::READ);
    assert( storage2.isOpened() );
    storage2["w_T_a"] >> mat_w_T_a;
    storage2["w_T_b"] >> mat_w_T_b;
    storage2["imu_T_cam"] >> mat_imu_T_cam;
    cv::cv2eigen(mat_w_T_a, w_T_a );
    cv::cv2eigen(mat_w_T_b, w_T_b );
    cv::cv2eigen(mat_imu_T_cam, imu_T_cam );
    storage2.release();

    cout << endl;
    cout << "w_T_a     =" << PoseManipUtils::prettyprintMatrix4d( w_T_a ) << endl;
    cout << "w_T_b     =" << PoseManipUtils::prettyprintMatrix4d( w_T_b ) << endl;
    cout << "imu_T_cam =" << PoseManipUtils::prettyprintMatrix4d( imu_T_cam ) << endl;
    cout << endl;

    cout << "w_T_imua  =" << PoseManipUtils::prettyprintMatrix4d( w_T_a *imu_T_cam.inverse() ) << endl;
    cout << "w_T_imub  =" << PoseManipUtils::prettyprintMatrix4d( w_T_b *imu_T_cam.inverse() ) << endl;

    cout << TermColor::bWHITE() <<  "-------------------------------------------------------\n";
    cout << "---- Data Loading Done  : hyp_idx="<< hyp_idx <<  "\n";
    cout << "-------------------------------------------------------\n" << TermColor::RESET();


    #if 0
    //   standard EA
    EdgeAlignment ealign( cam, im_ref, im_curr, depth_curr );
    ealign.set_make_representation_image();
    Matrix4d ref_T_curr_optvar;

    ElapsedTime t_main_ea( "ealign.solve()");
    bool ea_status = ealign.solve( initial_guess____ref_T_curr, ref_T_curr_optvar );
    cout << TermColor::uGREEN() << t_main_ea.toc() << TermColor::RESET() << endl;



    // some anaylsis
    // according to initial guess the pitch and roll between ref and curr imu
    cout << TermColor::bWHITE() << "Analyse\n" << TermColor::RESET() << endl;
    Matrix4d initial_guess____refimu_T_currimu =  imu_T_cam * initial_guess____ref_T_curr * imu_T_cam.inverse();
    Matrix4d refimu_T_currimu_optvar =  imu_T_cam * ref_T_curr_optvar * imu_T_cam.inverse();
    Matrix4d aimu_T_bimu = imu_T_cam * w_T_a.inverse() * w_T_b * imu_T_cam.inverse();

    cout << "initial_guess____refimu_T_currimu :" << PoseManipUtils::prettyprintMatrix4d( initial_guess____refimu_T_currimu ) << endl;
    cout << "refimu_T_currimu_optvar           :" << PoseManipUtils::prettyprintMatrix4d( refimu_T_currimu_optvar ) << endl;
    cout << "aimu_T_bimu                       :" << PoseManipUtils::prettyprintMatrix4d( aimu_T_bimu ) << endl;
    cout << TermColor::bWHITE() << "END Analyse\n" << TermColor::RESET() << endl;



    //   view debug image
    cv::imshow( "debug_image_ealign", ealign.get_representation_image() );
    cv::waitKey(0);

    #endif // EA


    #if 1
    // some anaylsis
    // according to initial guess the pitch and roll between ref and curr imu
    cout << TermColor::bWHITE() << "Analyse\n" << TermColor::RESET() << endl;
    Matrix4d initial_guess____refimu_T_currimu =  imu_T_cam * initial_guess____ref_T_curr * imu_T_cam.inverse();
    // Matrix4d refimu_T_currimu_optvar =  imu_T_cam * ref_T_curr_optvar * imu_T_cam.inverse();
    Matrix4d aimu_T_bimu = imu_T_cam * w_T_a.inverse() * w_T_b * imu_T_cam.inverse();

    cout << "initial_guess____ref_T_curr       :" << PoseManipUtils::prettyprintMatrix4d( initial_guess____ref_T_curr ) << endl;
    cout << "initial_guess____refimu_T_currimu :" << PoseManipUtils::prettyprintMatrix4d( initial_guess____refimu_T_currimu ) << endl;
    // cout << "refimu_T_currimu_optvar           :" << PoseManipUtils::prettyprintMatrix4d( refimu_T_currimu_optvar ) << endl;
    cout << "aimu_T_bimu                       :" << PoseManipUtils::prettyprintMatrix4d( aimu_T_bimu ) << endl;
    cout << TermColor::bWHITE() << "END Analyse\n" << TermColor::RESET() << endl;





    #if 1
    // 4DOF EA
    EdgeAlignment ealign( cam, im_ref, im_curr, depth_curr );
    ealign.set_make_representation_image();

    ElapsedTime t_main_ea( "ealign.solve()");
    Matrix4d ref_T_curr_optvar;
    bool ea_4dof_status = ealign.solve4DOF( initial_guess____ref_T_curr,
        imu_T_cam, w_T_a, w_T_b,
        ref_T_curr_optvar );
    cout << TermColor::uGREEN() << t_main_ea.toc() << TermColor::RESET() << endl;
    cout << "[main] status = " << ea_4dof_status << ", ref_T_curr_optvar = " << ref_T_curr_optvar << endl;

    cv::imshow( "debug_image_ealign", ealign.get_representation_image() );
    cv::waitKey(0);


    #endif

    #endif

}
