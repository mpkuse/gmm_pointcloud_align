// This creates a local pointcloud. The implementation is based on
// Wang Kaixin's DenseSurfel Mapping


#include <iostream>
#include <vector>
#include <fstream>
using namespace std;

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
#include "SlicClustering.h"
#include "utils/CameraGeometry.h"
#include "utils/RawFileIO.h"


#include "utils/TermColor.h"

const string base_path = "/Bulk_Data/chkpts_cerebro/";
const string pkg_path = "/app/catkin_ws/src/gmm_pointcloud_align/";



camodocal::CameraPtr left_camera, right_camera;
Matrix4d right_T_left;
std::shared_ptr<StereoGeometry> stereogeom;

bool process_this_datanode( json data_node, Matrix4d toret__wTc, MatrixXd toret__cX )
{
    cout << TermColor::GREEN() << "[process_this_datanode]t=" << data_node["stampNSec"] << TermColor::RESET() << endl;
    int64_t t_sec = data_node["stampNSec"];
    ros::Time stamp = ros::Time().fromNSec( t_sec );


    // if wTc and image do not exist then return
    if( data_node["isPoseAvailable"] == false || data_node["isKeyFrame"] == false ) {
        cout << "\tno pose or image data...return false\n";
        return false;
    }


    // odom pose
    Matrix4d w_T_c;
    // string _tmp = data_node["w_T_c"];
    bool status = RawFileIO::read_eigen_matrix4d_fromjson(  data_node["w_T_c"], w_T_c  );


    string imleft_fname = base_path+"/cerebro_stash/left_image__" + std::to_string(stamp.toNSec()) + ".jpg";
    cout << "\tload imleft_fname: " << imleft_fname << endl;
    cv::Mat left_image = cv::imread( imleft_fname, 0 );
    if( !left_image.data ) {
        cout << TermColor::RED() << "[process_this_datanode]ERROR cannot load image...return false\n" << TermColor::RESET();
        return false;
    }
    cout << "left_image " << MiscUtils::cvmat_info( left_image ) << endl;

    string imright_fname = base_path+"/cerebro_stash/right_image__" + std::to_string(stamp.toNSec()) + ".jpg";
    cout << "\tload imright_fname: " << imright_fname << endl;
    cv::Mat right_image = cv::imread( imright_fname, 0 );
    if( !right_image.data ) {
        cout << TermColor::RED() << "[process_this_datanode]ERROR cannot load image...return false\n" << TermColor::RESET();
        return false;
    }
    cout << "right_image " << MiscUtils::cvmat_info( right_image ) << endl;


    // make a depth map
    #if 1
    ElapsedTime t_stereogeom;
    cv::Mat disparity, disparity_for_visualization_gray;
    t_stereogeom.tic();
    stereogeom->do_stereoblockmatching_of_srectified_images( left_image, right_image, disparity );
    cout << TermColor::BLUE() << "do_stereoblockmatching_of_srectified_images took (ms): " << t_stereogeom.toc_milli() << TermColor::RESET() << endl;


    stereogeom->disparity_to_falsecolormap( disparity, disparity_for_visualization_gray );

    #if 0
    cv::Mat out3D;     MatrixXd _3dpts;
    stereogeom->disparity_to_3DPoints( disparity, out3D, _3dpts, false );
    cout << "out3D " << MiscUtils::cvmat_info( out3D ) << endl; //32FC3
    cout << "disparity " << MiscUtils::cvmat_info( disparity ) << endl;
    cout << "disparity_for_visualization_gray " << MiscUtils::cvmat_info( disparity_for_visualization_gray ) << endl;
    #endif

    cv::Mat depth_map;

    #if 1
    t_stereogeom.tic();
    stereogeom->disparity_to_depth( disparity, depth_map );
    cout << TermColor::BLUE() << "disparity_to_depth took (ms): " << t_stereogeom.toc_milli() << TermColor::RESET() << endl;
    cout << "depth_map " << MiscUtils::cvmat_info( depth_map ) << endl;
    #else

    cv::Mat __XYZ[3];
    cv::split( out3D, __XYZ );
    depth_map = __XYZ[2];

    #endif


    // cv::split( )
    cv::imshow( "left_image", left_image );
    cv::imshow( "disparity" , disparity_for_visualization_gray );
    // cv::waitKey(0);


    #endif


    //-------------- now we have all the needed data ---------------------//
    // USE:
    // stamp, w_T_c, left_image, right_image, out3D, depth_map
    //--------------------------------------------------------------------//
    ElapsedTime t_slic;

    // SLIC
    // SlicClustering slic_obj;
    // slic_obj.generate( left_image, out3D, ddd, dddd );
    int w = left_image.cols, h = left_image.rows;
    int nr_superpixels = 400;
    int nc = 40;
    double step = sqrt((w * h) / ( (double) nr_superpixels ) ); ///< step size per cluster
    cout << "SLIC Params:\n";
    cout << "step size per cluster: " << step << endl;
    cout << "Weight: " << nc << endl;
    cout << "Number of superpixel: "<< nr_superpixels << endl;
    cout << "===\n";


    SlicClustering slic_obj;
    t_slic.tic();
    slic_obj.generate_superpixels( left_image, depth_map, step, nc );
    cout << TermColor::BLUE() << "t_slic (ms): " << t_slic.toc_milli() << TermColor::RESET() << endl;

    //viz
    cv::Mat imA;
    if( left_image.channels() == 3 )
        imA = left_image.clone();
    else
        cv::cvtColor(left_image, imA, cv::COLOR_GRAY2BGR);

    // slic_obj.display_center_grid( imA, cv::Scalar(0,0,255) );
    // slic_obj.colour_with_cluster_means( imA );
    slic_obj.display_contours( imA,  cv::Scalar(0,0,255) );
    // slic_obj.display_center_grid();

    MatrixXd sp_3dpts___cX = slic_obj.retrive_superpixel_XYZ( true );

    cv::imshow( "slic" , imA );
    cv::waitKey(0);

    //
    //--- Return
    toret__wTc = w_T_c;
    toret__cX  = sp_3dpts___cX;

    cout << TermColor::GREEN() << "[process_this_datanode]Finished " << data_node["stampNSec"] << TermColor::RESET() << endl;

    return true;

}

int main( int argc, char ** argv )
{
    //
    // specify start and end image index idx for ptcld construction
    //
    int startImIdx = 80;
    int endImIndx  = 100;


    //
    // Ros INIT
    //


    //
    // Load Camera (camodocal)
    //

    {
    std::string calib_file = pkg_path+"/resources/realsense_d435i_left.yaml";
    cout << TermColor::YELLOW() << "Camodocal load: " << calib_file << TermColor::RESET() << endl;
    left_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
    std::cout << ((left_camera)?"left cam is initialized":"cam is not initiazed") << std::endl; //this should print 'initialized'
    }

    {
    std::string calib_file = pkg_path+"/resources/realsense_d435i_right.yaml";
    cout << TermColor::YELLOW() << "Camodocal load: " << calib_file << TermColor::RESET() << endl;
    right_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
    std::cout << ((right_camera)?"right cam is initialized":"cam is not initiazed") << std::endl; //this should print 'initialized'
    }


    {
        string ___extrinsic_1_T_0_path = pkg_path+"/resources/realsense_d435i_extrinsics.yaml";
        cout << "opencv yaml reading: open file: " << ___extrinsic_1_T_0_path << endl;
        cv::FileStorage fs(___extrinsic_1_T_0_path, cv::FileStorage::READ);

        if (!fs.isOpened())
        {
            cout <<   "config_file asked to open extrinsicbasline file but it cannot be opened.\nTHIS IS FATAL, QUITING" ;
            exit(1);
        }

        cout << TermColor::GREEN() << "successfully opened file "<< ___extrinsic_1_T_0_path << TermColor::RESET() << endl;
        cv::FileNode n = fs["transform"];
        if( n.empty() ) {
            cout << TermColor::RED() << "I was looking for the key `transform` in the file but it doesnt seem to exist. FATAL ERROR" << TermColor::RESET() << endl;
            exit(1);
        }
        Vector4d q_xyzw;
        q_xyzw << (double)n["q_x"] , (double)n["q_y"] ,(double) n["q_z"] ,(double) n["q_w"];

        Vector3d tr_xyz;
        tr_xyz << (double)n["t_x"] , (double)n["t_y"] ,(double) n["t_z"];

        cout << "--values from file--\n" << TermColor::iGREEN();
        cout << "q_xyzw:\n" << q_xyzw << endl;
        cout << "tr_xyz:\n" << tr_xyz << endl;
        cout << TermColor::RESET() << endl;

        Matrix4d _1_T_0;
        PoseManipUtils::raw_xyzw_to_eigenmat( q_xyzw, tr_xyz/1000., _1_T_0 ); cout << "translation divided by 1000 to convert from mm (in file) to meters (as needed)\n";
        // cout << TermColor::iBLUE() << "_1_T_0:\n" <<  _1_T_0  << TermColor::RESET() << endl;
        cout << TermColor::iBLUE() << "_1_T_0: " << PoseManipUtils::prettyprintMatrix4d( _1_T_0 ) << TermColor::RESET() << endl;
        cout << "_1_T_0:\n" << _1_T_0 << endl;


        right_T_left = _1_T_0;


    }

    stereogeom = std::make_shared<StereoGeometry>( left_camera,right_camera,     right_T_left  );


    // Load JSON (state.json)
    std:string json_fname = base_path + "/state.json";
    cout << TermColor::YELLOW() << "JSON load: " << json_fname << TermColor::RESET() << endl;
    std::ifstream json_stream(json_fname);
    if(json_stream.fail()){
      cout << TermColor::RED() << "Cannot load json file...exit" << TermColor::RESET() << endl;
      exit(1);
    }
    cout << "JSON Loaded successfully..." << endl;
    json STATE;
    json_stream >> STATE;
    json_stream.close();



    for( int i=0 ; i<STATE["DataNodes"].size() ; i++ )
    {
        if( STATE["DataNodes"][i]["seq"] >= startImIdx && STATE["DataNodes"][i]["seq"] < endImIndx )
        {
            Matrix4d w_T_c;
            MatrixXd sp_cX;
            bool status = process_this_datanode( STATE["DataNodes"][i],  w_T_c, sp_cX );

            if( status ) {

            }
        }
    }


}
