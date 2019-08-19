// This creates a local pointcloud. The implementation is based on
// Wang Kaixin's DenseSurfel Mapping


#include <iostream>
#include <vector>
#include <fstream>
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

// ROS
#include <ros/ros.h>
#include <visualization_msgs/Marker.h>

// Camodocal
#include "camodocal/camera_models/Camera.h"
#include "camodocal/camera_models/CameraFactory.h"


// JSON
#include "utils/nlohmann/json.hpp"
using json = nlohmann::json;

// KuseUtils
#include "SlicClustering.h"
#include "utils/CameraGeometry.h"

//
#include "utils/RosMarkerUtils.h"
#include "utils/RawFileIO.h"
#include "utils/TermColor.h"
#include "utils/ElapsedTime.h"



const string base_path = "/Bulk_Data/chkpts_cerebro/";
const string pkg_path = "/app/catkin_ws/src/gmm_pointcloud_align/";



camodocal::CameraPtr left_camera, right_camera;
Matrix4d right_T_left;
std::shared_ptr<StereoGeometry> stereogeom;



bool load_left_camera()
{
    std::string calib_file = pkg_path+"/resources/realsense_d435i_left.yaml";
    cout << TermColor::YELLOW() << "Camodocal load: " << calib_file << TermColor::RESET() << endl;
    left_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
    std::cout << ((left_camera)?"left cam is initialized":"cam is not initiazed") << std::endl; //this should print 'initialized'
}

bool load_right_camera()
{
    std::string calib_file = pkg_path+"/resources/realsense_d435i_right.yaml";
    cout << TermColor::YELLOW() << "Camodocal load: " << calib_file << TermColor::RESET() << endl;
    right_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
    std::cout << ((right_camera)?"right cam is initialized":"cam is not initiazed") << std::endl; //this should print 'initialized'
}

bool load_stereo_extrinsics()
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


json load_json()
{
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
    return STATE;
}

bool process_this_datanode( json data_node, Matrix4d& toret__wTc, MatrixXd& toret__cX )
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
    // cv::imshow( "left_image", left_image );
    cv::imshow( "disparity" , disparity_for_visualization_gray );
    // cv::waitKey(0);



    #if 0 // save to file
    cout << TermColor::iRED() << "..........SAVE TO FILE...............\n" << TermColor::RESET() << endl;
    string opencv_file_name = pkg_path+"/resources/rgbd_samples/"+to_string(t_sec)+".xml";
    cout << "Write: " << opencv_file_name << endl;
    cv::FileStorage opencv_file(opencv_file_name, cv::FileStorage::WRITE);
    if( opencv_file.isOpened() == false ) {
        cout << "ERROR....><><> Cannot open opencv_file: " << opencv_file_name << endl;
        exit(1);
    }
    opencv_file << "left_image" << left_image;
    opencv_file << "right_image" << right_image;
    opencv_file << "depth_map" << depth_map;
    opencv_file << "disparity_for_visualization_gray" << disparity_for_visualization_gray;

    opencv_file.release();
    cout << TermColor::iRED() << "..........SAVE TO FILE DONE...............\n" << TermColor::RESET() << endl;
    #endif


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
    // slic_obj.generate_superpixels( left_image, depth_map, step, nc );
    slic_obj.generate_superpixels( left_image, depth_map, step );
    cout << TermColor::BLUE() << "t_slic (ms): " << t_slic.toc_milli() << TermColor::RESET() << endl;

    //viz
    cv::Mat imA;
    if( left_image.channels() == 3 )
        imA = left_image.clone();
    else
        cv::cvtColor(left_image, imA, cv::COLOR_GRAY2BGR);

    // slic_obj.display_center_grid( imA, cv::Scalar(0,0,255) );
    // slic_obj.colour_with_cluster_means( imA );
    slic_obj.display_contours( left_image,  cv::Scalar(0,0,255), imA );
    // slic_obj.display_center_grid();

    MatrixXd sp_3dpts___cX = slic_obj.retrive_superpixel_XYZ( true );

    cv::imshow( "slic" , imA );
    cv::waitKey(10);

    //
    //--- Return
    toret__wTc = w_T_c;
    toret__cX  = sp_3dpts___cX;

    cout << TermColor::GREEN() << "[process_this_datanode]Finished " << data_node["stampNSec"] << TermColor::RESET() << endl;

    return true;

}


void _print_(
    vector<   vector<int>         > all_i,
    vector<   vector<Matrix4d>    > all_odom_poses,
    vector<   vector<MatrixXd>    > all_sp_cX,

    vector<int> curr_i,
    vector<Matrix4d> curr_odom_poses,
    vector<MatrixXd> curr_sp_cX,

    ros::Publisher& marker_pub,
    bool publish_all = false
  )
  {
      cout << TermColor::YELLOW() << "_print_" << TermColor::RESET() << endl;
      ros::spinOnce();


      //
      //
      visualization_msgs::Marker local_3dpt;
      RosMarkerUtils::init_points_marker( local_3dpt );
      local_3dpt.ns = "curr"+to_string(all_i.size());
      local_3dpt.id = 0;
      local_3dpt.scale.x = 0.02;
      local_3dpt.scale.y = 0.02;
      vector< visualization_msgs::Marker > cam_viz;
      cv::Scalar color = FalseColors::randomColor(all_i.size());


      cout << "===current";
      cout << " #items = " << curr_i.size() << endl;
      int total_3dpts = 0;
      for( int i=0 ; i<curr_i.size() ; i++ ) {
          // print
          cout << "\t" << i << "  im#"<< curr_i[i] << " has " << curr_sp_cX[i].cols() << " 3d points\n";
          total_3dpts += curr_sp_cX[i].cols();


          // 3dpoints
          MatrixXd wX = curr_odom_poses[0].inverse() * curr_odom_poses[i] * curr_sp_cX[i]; // wTc * cX
          RosMarkerUtils::add_points_to_marker( wX, local_3dpt, (  (i==0)?true:false  ) );
          RosMarkerUtils::setcolor_to_marker( color[0]/255., color[1]/255., color[2]/255., .3, local_3dpt );

          // camera visual
          visualization_msgs::Marker cam_viz_i;
          cam_viz_i.ns = local_3dpt.ns+"__cam_viz";
          cam_viz_i.id = i;
          RosMarkerUtils::init_camera_marker( cam_viz_i, 4.0 );
          Matrix4d c0_T_ci =  curr_odom_poses[0].inverse() * curr_odom_poses[i];
          RosMarkerUtils::setpose_to_marker( c0_T_ci, cam_viz_i  );
          RosMarkerUtils::setcolor_to_marker(color[0]/255., color[1]/255., color[2]/255., 1.0, cam_viz_i );
          cam_viz.push_back( cam_viz_i );
      }
      cout << "total_3dpts=" << total_3dpts << endl;
      cout << endl;


      //
      //
      marker_pub.publish( local_3dpt );
      for( int i=0 ; i<cam_viz.size() ; i++ )
          marker_pub.publish( cam_viz[i] );



      //
      //
      cout << "===all size= " << all_i.size() <<  "\n";

            if( publish_all == false ) {
                cout << "not publishing all\n";
                return;
            }

      for( int j=0 ; j<all_i.size() ; j++ )  //loop over all the hypothesis
      {
          cout << "\t======loop hypothesis#" << j << "\t";
          cout << " #items = " << all_i[j].size() << endl;
          int total_3dpts = 0;

          visualization_msgs::Marker local_3dpt;
          RosMarkerUtils::init_points_marker( local_3dpt );
          local_3dpt.ns = "curr"+to_string(j);
          local_3dpt.id = 0;
          local_3dpt.scale.x = 0.02;
          local_3dpt.scale.y = 0.02;
          vector< visualization_msgs::Marker > cam_viz;
          cv::Scalar color = FalseColors::randomColor(j);


          for( int i=0 ; i<all_i[j].size() ; i++ )
          {
              cout << "\t\t" << i << "  im#"<<all_i[j][i] << " has " << all_sp_cX[j][i].cols() << " 3d points\n";
              total_3dpts += all_sp_cX[j][i].cols();


              // 3dpoints
              MatrixXd wX = all_odom_poses[j][0].inverse() * all_odom_poses[j][i] * all_sp_cX[j][i]; // wTc * cX
              RosMarkerUtils::add_points_to_marker( wX, local_3dpt, (  (i==0)?true:false  ) );
              RosMarkerUtils::setcolor_to_marker( color[0]/255., color[1]/255., color[2]/255., .3, local_3dpt );


              // camera visual
              visualization_msgs::Marker cam_viz_i;
              cam_viz_i.ns = local_3dpt.ns+"__cam_viz";
              cam_viz_i.id = i;
              RosMarkerUtils::init_camera_marker( cam_viz_i, 4.0 );
              Matrix4d c0_T_ci =  all_odom_poses[j][0].inverse() * all_odom_poses[j][i];
              RosMarkerUtils::setpose_to_marker( c0_T_ci, cam_viz_i  );
              RosMarkerUtils::setcolor_to_marker(color[0]/255., color[1]/255., color[2]/255., 1.0, cam_viz_i );
              cam_viz.push_back( cam_viz_i );

          }
          cout << "\ttotal_3dpts=" << total_3dpts << endl;

          marker_pub.publish( local_3dpt );
          for( int i=0 ; i<cam_viz.size() ; i++ )
              marker_pub.publish( cam_viz[i] );


      }
  }

int main( int argc, char ** argv ) {
    //
    // Ros INIT
    //
    ros::init(argc, argv, "local_ptcloud");
    ros::NodeHandle nh;
    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("marker", 1000);

    //
    // Load Camera (camodocal)
    //
    load_left_camera();
    load_right_camera();
    load_stereo_extrinsics();
    stereogeom = std::make_shared<StereoGeometry>( left_camera,right_camera,     right_T_left  );

    //
    // Load JSON
    //
    json STATE = load_json();



    //
    // LOOP Over data and show image.
    //
    vector<   vector<int>         > all_i; //other vector to hold multiple sequences
    vector<   vector<Matrix4d>    > all_odom_poses;
    vector<   vector<MatrixXd>    > all_sp_cX;

    vector<int> curr_i;
    vector<Matrix4d> curr_odom_poses;
    vector<MatrixXd> curr_sp_cX;

    for( int i=0 ; i< (int) STATE["DataNodes"].size() ; i++ )
    {
        json data_node = STATE["DataNodes"][i];
        ros::Time stamp = ros::Time().fromNSec( data_node["stampNSec"] );


        if( data_node["isPoseAvailable"] == false || data_node["isKeyFrame"] == false ) {
            continue;
        }


        // show image
        string imleft_fname = base_path+"/cerebro_stash/left_image__" + std::to_string(stamp.toNSec()) + ".jpg";
        cv::Mat left_image = cv::imread( imleft_fname, 0 );
        if( !left_image.data ) {
            cout << TermColor::RED() << "[main]ERROR cannot load image...return false\n" << TermColor::RESET();
            exit(1);
        }
        cout << "seq=" << data_node["seq"] << "\t";
        cout << "t=" << stamp << "\t";
        cout << "left_image"  << MiscUtils::cvmat_info( left_image ) << endl;

        cv::imshow( "left_image", left_image );
        char ch = cv::waitKey(0);
        cout << "\tyou pressed: ``" << ch << "``" << endl;

        if( ch == 'q' ) {
            cout << "QUIT....\n";
            exit(0);
        }

        if( ch == 'b' ) {
            cout << "BREAK....\n";
            break;
        }

        if( ch == 'p' ) {
            Matrix4d wTc;
            MatrixXd sp_cX;
            process_this_datanode( data_node, wTc, sp_cX );
            curr_i.push_back(i);
            curr_odom_poses.push_back( wTc );
            curr_sp_cX.push_back( sp_cX ); //TODO: instead of simply accumulating 3d points look at how Wang Kaixuan fuse it.


            _print_( all_i, all_odom_poses, all_sp_cX,
                     curr_i, curr_odom_poses, curr_sp_cX, marker_pub  );
        }

        if( ch == 'n' ) {
            cout << TermColor::iGREEN() << "=======starting a new seq======" << TermColor::RESET() << endl;
            cout << "Stash current sequence\n";
            all_i.push_back( curr_i );
            all_odom_poses.push_back( curr_odom_poses );
            all_sp_cX.push_back( curr_sp_cX );

            cout << "Clear current sequence data\n";
            curr_i.clear();
            curr_odom_poses.clear();
            curr_sp_cX.clear();
        }

        if( ch == 'l' ) {
            _print_( all_i, all_odom_poses, all_sp_cX,
                     curr_i, curr_odom_poses, curr_sp_cX,
                     marker_pub, true  );
        }

    }






}

#if 0
int main( int argc, char ** argv )
{
    if( argc != 6 ) {
        cout << "argc need to be 5 : startImIdx, endImIndx, red, green, blue\n";
        cout << "now it is argc=" << argc << endl;
        for( int i=0 ; i<argc ; i++ )
            cout << "argv["<< i << "] = " << argv[i] << endl;
        exit(1);
    }

    //
    // specify start and end image index idx for ptcld construction
    //
    /*
    #if 1
    int startImIdx = 2420;
    int endImIndx  = 2520;
    #else
    int startImIdx = 3400;
    int endImIndx  = 3500;
    #endif
    */
    int startImIdx = atoi( argv[1] );
    int endImIndx = atoi( argv[2] );


    //
    // Ros INIT
    //
    ros::init(argc, argv, "local_ptcloud");
    ros::NodeHandle nh;
    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("marker", 1000);



    //
    // Load Camera (camodocal)
    //
    load_left_camera();
    load_right_camera();
    load_stereo_extrinsics();
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



    vector<int> all_i;
    vector<Matrix4d> all_odom_poses;
    vector<MatrixXd> all_sp_cX;
    for( int i=0 ; i< (int) STATE["DataNodes"].size() ; i++ )
    {
        if( STATE["DataNodes"][i]["seq"] >= startImIdx && STATE["DataNodes"][i]["seq"] < endImIndx )
        {
            Matrix4d w_T_c;
            MatrixXd sp_cX;
            bool status = process_this_datanode( STATE["DataNodes"][i],  w_T_c, sp_cX );

            if( status ) {
                all_i.push_back( i );
                all_odom_poses.push_back( w_T_c );
                all_sp_cX.push_back( sp_cX );
            }
        }
    }



    // make ros-marker
    visualization_msgs::Marker local_3dpt;
    RosMarkerUtils::init_points_marker( local_3dpt );
    local_3dpt.ns = "local_ptcloud_"+to_string( *(all_i.begin() ) )+string("--->")+to_string( *(all_i.rbegin() ) );
    local_3dpt.id = 0;
    local_3dpt.scale.x = 0.02;
    local_3dpt.scale.y = 0.02;

    vector< visualization_msgs::Marker > cam_viz;
    for( int i=0 ; i<(int)all_i.size(); i++ )
    {
        float color__[3];
        color__[0] = atof( argv[3] );
        color__[1] = atof( argv[4] );
        color__[2] = atof( argv[5] );

        cout << i << "] seq=" << all_i[i] << " npts=" << all_sp_cX[i].cols() << endl;

        // 3dpoints
        MatrixXd wX = all_odom_poses[0].inverse() * all_odom_poses[i] * all_sp_cX[i]; // wTc * cX
        RosMarkerUtils::add_points_to_marker( wX, local_3dpt, (  (i==0)?true:false  ) );
        RosMarkerUtils::setcolor_to_marker( color__[0], color__[1], color__[2], .3, local_3dpt );

        // camera visual
        visualization_msgs::Marker cam_viz_i;
        cam_viz_i.ns = local_3dpt.ns+"__cam_viz";
        cam_viz_i.id = i;
        RosMarkerUtils::init_camera_marker( cam_viz_i, 4.0 );
        Matrix4d c0_T_ci =  all_odom_poses[0].inverse() * all_odom_poses[i];
        RosMarkerUtils::setpose_to_marker( c0_T_ci, cam_viz_i  );
        RosMarkerUtils::setcolor_to_marker(color__[0], color__[1], color__[2], 1.0, cam_viz_i );
        cam_viz.push_back( cam_viz_i );
    }


    ros::Rate rate(10);
    while( ros::ok() ) {
        rate.sleep();
        ros::spinOnce();

        marker_pub.publish( local_3dpt );
        for( int i=0 ; i<cam_viz.size() ; i++ )
            marker_pub.publish( cam_viz[i] );
    }

}

#endif
