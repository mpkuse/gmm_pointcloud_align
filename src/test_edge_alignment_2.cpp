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

const int hyp_idx = 11;
const int ea_idx  = 0;
int main( )
{
    cout << "Hello Edge Alignment 2: hyp_idx="<< hyp_idx << "\tea_idx=" << ea_idx << "\n";

    //   Load camodocal
    #if 1
    // load camodocam camera
    camodocal::CameraPtr cam = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(CAM_PREFIX+"/camera.yaml");
    if( !cam ) { cout << "Cannot load camera\n"; exit(2); }
    cout << cam->parametersToString() << endl;
    #endif



    //   Load im_ref, im_curr
    string im_ref_fname = ED_DEBUG_DATA_PREFIX+"hyp_"+to_string(hyp_idx)+"_ea_"+to_string(ea_idx)+"_im_ref.jpg";
    string im_curr_fname = ED_DEBUG_DATA_PREFIX+"hyp_"+to_string(hyp_idx)+"_ea_"+to_string(ea_idx)+"_im_curr.jpg";
    cv::Mat im_ref = cv::imread( im_ref_fname, 0  );
    cv::Mat im_curr = cv::imread( im_curr_fname, 0  );
    assert( !im_ref.empty() && !im_curr.empty() );

    //   Load depth_curr
    //   Load initial guess
    string depth_curr_fname = ED_DEBUG_DATA_PREFIX+"hyp_"+to_string(hyp_idx)+"_ea_"+to_string(ea_idx)+"_depth_curr.yaml";
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

    //   EA
    EdgeAlignment ealign( cam, im_ref, im_curr, depth_curr );
    ealign.set_make_representation_image();
    Matrix4d ref_T_curr_optvar;
    bool ea_status = ealign.solve( initial_guess____ref_T_curr, ref_T_curr_optvar );


    //   view debug image
    cv::imshow( "debug_image_ealign", ealign.get_representation_image() );
    cv::waitKey(0);

}
