#pragma once
// Some functions to load camera calib and other related data for
// stand alone testing



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
#include "SurfelMap.h"
#include "SlicClustering.h"
#include "utils/CameraGeometry.h"

//
#include "utils/RosMarkerUtils.h"
#include "utils/RawFileIO.h"
#include "utils/TermColor.h"
#include "utils/ElapsedTime.h"

class XLoader
{
public:
    XLoader() {}

    const string base_path = "/Bulk_Data/chkpts_cerebro/";
    const string pkg_path = "/app/catkin_ws/src/gmm_pointcloud_align/";

    camodocal::CameraPtr left_camera, right_camera;
    Matrix4d right_T_left;
    std::shared_ptr<StereoGeometry> stereogeom;

    bool is_imuTcam_available = false;
    Matrix4d imu_T_cam;


    bool load_left_camera()
    {
        std::string calib_file = pkg_path+"/resources/realsense_d435i_left.yaml";
        cout << TermColor::YELLOW() << "Camodocal load: " << calib_file << TermColor::RESET() << endl;
        left_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
        std::cout << ((left_camera)?"left cam is initialized":"cam is not initiazed") << std::endl; //this should print 'initialized'
        return ((left_camera)?true:false);
    }

    bool load_right_camera()
    {
        std::string calib_file = pkg_path+"/resources/realsense_d435i_right.yaml";
        cout << TermColor::YELLOW() << "Camodocal load: " << calib_file << TermColor::RESET() << endl;
        right_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
        std::cout << ((right_camera)?"right cam is initialized":"cam is not initiazed") << std::endl; //this should print 'initialized'
        return ((right_camera)?true:false);
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
        return true;

    }


    bool make_stereogeometry()
    {
        assert( left_camera && right_camera );
        assert( right_T_left.rows() == 4 && right_T_left.cols() == 4 );
        stereogeom = std::make_shared<StereoGeometry>( left_camera,right_camera, right_T_left  );
        return true;
    }

    json load_json()
    {
        std::string json_fname = base_path + "/state.json";
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



        // if json contains imu_T_cam
        try{
            json m_vars = STATE.at("MiscVariables");
            json imu_t_cam_json = m_vars.at("imu_T_cam");
            cout << imu_t_cam_json << endl;
            // Matrix4d imu_T_cam;
            RawFileIO::read_eigen_matrix4d_fromjson( imu_t_cam_json, imu_T_cam );
            cout << "(json) imu_T_cam:\n";
            cout << imu_T_cam << endl;
            is_imuTcam_available = true;
        }
        catch (json::type_error& e)
        {
            is_imuTcam_available = false;
            cout << e.what() << endl;
        }


        return STATE;
    }


#define _retrive_info_( msg ) msg;
// #define _retrive_info_( msg ) ;

// #define _retrive_debug_( msg ) msg;
#define _retrive_debug_( msg ) ;

ros::Time retrive_timestamp_from_json_datanode( json data_node )
{
    int64_t t_sec = data_node["stampNSec"];
    ros::Time stamp = ros::Time().fromNSec( t_sec );
    return stamp;
}

bool retrive_image_data_from_json_datanode( json data_node,
        // ros::Time& stamp, Matrix4d& w_T_c,
        cv::Mat& left_image
    )
    {
        _retrive_info_( cout << TermColor::GREEN() << "[retrive_image_data_from_json_datanode]t=" << data_node["stampNSec"] << TermColor::RESET() << endl;)

            int64_t t_sec = data_node["stampNSec"];
            ros::Time stamp = ros::Time().fromNSec( t_sec );


            // if wTc and image do not exist then return
            if( data_node["isPoseAvailable"] == false || data_node["isKeyFrame"] == false ) {
                cout << "\tno pose or image data...return false\n";
                return false;
            }


            // odom pose
            // Matrix4d w_T_c;
            // string _tmp = data_node["w_T_c"];
            // bool status = RawFileIO::read_eigen_matrix4d_fromjson(  data_node["w_T_c"], w_T_c  );
            // assert( status );


            string imleft_fname = base_path+"/cerebro_stash/left_image__" + std::to_string(stamp.toNSec()) + ".jpg";
            _retrive_debug_( cout << "\tload imleft_fname: " << imleft_fname << endl; )
            left_image = cv::imread( imleft_fname, 0 );
            if( !left_image.data ) {
                cout << TermColor::RED() << "[retrive_data_from_json_datanode]ERROR cannot load image...return false\n" << TermColor::RESET();
                return false;
            }
            _retrive_debug_( cout << "\tleft_image " << MiscUtils::cvmat_info( left_image ) << endl; )
            return true;

    }


    // return true if OK to retrive or returns false
    bool is_data_available( json data_node  )
    {
        _retrive_info_( cout << TermColor::GREEN() << "[is_data_available]t=" << data_node["stampNSec"] << TermColor::RESET() << endl; )

            int64_t t_sec = data_node["stampNSec"];
            ros::Time stamp = ros::Time().fromNSec( t_sec );


            // if wTc and image do not exist then return
            if( data_node["isPoseAvailable"] == false || data_node["isKeyFrame"] == false ) {
                // cout << "\t[retrive_image_data_from_json_datanode]no pose or image data...return false\n";
                return false;
            }
            return true;
    }

    bool retrive_image_data_from_json_datanode( json data_node,
        // ros::Time& stamp, Matrix4d& w_T_c,
        cv::Mat& left_image,
        cv::Mat & depth_map
    )
    {
        _retrive_info_( cout << TermColor::GREEN() << "[retrive_image_data_from_json_datanode]t=" << data_node["stampNSec"] << TermColor::RESET() << endl; )

            int64_t t_sec = data_node["stampNSec"];
            ros::Time stamp = ros::Time().fromNSec( t_sec );


            // if wTc and image do not exist then return
            if( data_node["isPoseAvailable"] == false || data_node["isKeyFrame"] == false ) {
                cout << "\t[retrive_image_data_from_json_datanode]no pose or image data...return false\n";
                return false;
            }


            // odom pose
            // Matrix4d w_T_c;
            // string _tmp = data_node["w_T_c"];
            // bool status = RawFileIO::read_eigen_matrix4d_fromjson(  data_node["w_T_c"], w_T_c  );
            // assert( status );


            string imleft_fname = base_path+"/cerebro_stash/left_image__" + std::to_string(stamp.toNSec()) + ".jpg";
            _retrive_debug_( cout << "\tload imleft_fname: " << imleft_fname << endl; )
            left_image = cv::imread( imleft_fname, 0 );
            if( !left_image.data ) {
                cout << TermColor::RED() << "[retrive_data_from_json_datanode]ERROR cannot load image...return false\n" << TermColor::RESET();
                return false;
            }
            _retrive_debug_( cout << "\tleft_image " << MiscUtils::cvmat_info( left_image ) << endl; )



            // Direct Load depth
            string depth_image_fname = base_path+"/cerebro_stash/depth_image__" + std::to_string(stamp.toNSec()) + ".jpg.png";
            _retrive_debug_( cout << "\tload depth_image_fname: " << depth_image_fname << endl; )
            cv::Mat depth_image = cv::imread( depth_image_fname, -1 );
            if( !depth_image.data ) {
                cout << TermColor::RED() << "[retrive_data_from_json_datanode]ERROR cannot load depth-image...return false\n" << TermColor::RESET();
                return false;
            }
            _retrive_debug_( cout << "\tdepth_image " << MiscUtils::cvmat_info( depth_image ) << endl; )
            depth_map = depth_image;
            return true;

    }

    bool retrive_pose_from_json_datanode( json data_node, Matrix4d& w_T_c )
    {
        _retrive_info_( cout << TermColor::GREEN() << "[retrive_pose_from_json_datanode]t=" << data_node["stampNSec"] << TermColor::RESET() << endl; )

            int64_t t_sec = data_node["stampNSec"];
            ros::Time stamp = ros::Time().fromNSec( t_sec );


            // if wTc and image do not exist then return
            if( data_node["isPoseAvailable"] == false || data_node["isKeyFrame"] == false ) {
                cout << "\t[retrive_pose_from_json_datanode]no pose or image data...return false\n";
                return false;
            }

            // odom pose
            // Matrix4d w_T_c;
            // string _tmp = data_node["w_T_c"];
            bool status = RawFileIO::read_eigen_matrix4d_fromjson(  data_node["w_T_c"], w_T_c  );
            assert( status );
            return status;

    }

    const Matrix4d retrive_pose_from_json_datanode( json data_node  )
    {
        _retrive_info_( cout << TermColor::GREEN() << "[retrive_pose_from_json_datanode]t=" << data_node["stampNSec"] << TermColor::RESET() << endl; )

            int64_t t_sec = data_node["stampNSec"];
            ros::Time stamp = ros::Time().fromNSec( t_sec );


            // if wTc and image do not exist then return
            if( data_node["isPoseAvailable"] == false || data_node["isKeyFrame"] == false ) {
                cout << "\t[retrive_pose_from_json_datanode]no pose or image data...return false\n";
                throw 3;
            }

            // odom pose
            // Matrix4d w_T_c;
            // string _tmp = data_node["w_T_c"];
            Matrix4d w_T_c;
            bool status = RawFileIO::read_eigen_matrix4d_fromjson(  data_node["w_T_c"], w_T_c  );
            if( status == false ) {
                cout <<__FILE__ << ":" << __LINE__ <<  ": status was false\n";
                throw 4;
            }

            return w_T_c;

    }

    bool retrive_data_from_json_datanode( json data_node,
        ros::Time& stamp, Matrix4d& w_T_c,
        cv::Mat& left_image, cv::Mat& right_image,
        cv::Mat& depth_map, cv::Mat& disparity_for_visualization_gray
    )
    {
        _retrive_info_( cout << TermColor::GREEN() << "[retrive_data_from_json_datanode]t=" << data_node["stampNSec"] << TermColor::RESET() << endl; )

            int64_t t_sec = data_node["stampNSec"];
            stamp = ros::Time().fromNSec( t_sec );


            // if wTc and image do not exist then return
            if( data_node["isPoseAvailable"] == false || data_node["isKeyFrame"] == false ) {
                cout << "\tno pose or image data...return false\n";
                return false;
            }


            // odom pose
            // Matrix4d w_T_c;
            // string _tmp = data_node["w_T_c"];
            bool status = RawFileIO::read_eigen_matrix4d_fromjson(  data_node["w_T_c"], w_T_c  );
            assert( status );


            string imleft_fname = base_path+"/cerebro_stash/left_image__" + std::to_string(stamp.toNSec()) + ".jpg";
            _retrive_debug_( cout << "\tload imleft_fname: " << imleft_fname << endl; )
            left_image = cv::imread( imleft_fname, 0 );
            if( !left_image.data ) {
                cout << TermColor::RED() << "[retrive_data_from_json_datanode]ERROR cannot load image...return false\n" << TermColor::RESET();
                return false;
            }
            _retrive_debug_( cout << "\tleft_image " << MiscUtils::cvmat_info( left_image ) << endl; )

            string imright_fname = base_path+"/cerebro_stash/right_image__" + std::to_string(stamp.toNSec()) + ".jpg";
            cout << "\tload imright_fname: " << imright_fname << endl;
            right_image = cv::imread( imright_fname, 0 );
            if( !right_image.data ) {
                cout << TermColor::RED() << "[retrive_data_from_json_datanode]ERROR cannot load image...return false\n" << TermColor::RESET();
                return false;
            }
            _retrive_debug_( cout << "\tright_image " << MiscUtils::cvmat_info( right_image ) << endl; )


            // make a depth map

            #if 0 //set this to 1 to use stereogeometry, set this to zero to use depth image from disk.
            // Using StereoGeometry
            ElapsedTime t_stereogeom;
            cv::Mat disparity;
            // cv::Mat disparity_for_visualization_gray;
            t_stereogeom.tic();
            stereogeom->do_stereoblockmatching_of_srectified_images( left_image, right_image, disparity );
            cout << TermColor::BLUE() << "\t[retrive_data_from_json_datanode] do_stereoblockmatching_of_srectified_images took (ms): " << t_stereogeom.toc_milli() << TermColor::RESET() << endl;


            stereogeom->disparity_to_falsecolormap( disparity, disparity_for_visualization_gray );
            // cv::Mat depth_map;

            t_stereogeom.tic();
            stereogeom->disparity_to_depth( disparity, depth_map );
            cout << TermColor::BLUE() << "\t[retrive_data_from_json_datanode] disparity_to_depth took (ms): " << t_stereogeom.toc_milli() << TermColor::RESET() << endl;
            cout << "\tdepth_map " << MiscUtils::cvmat_info( depth_map ) << endl;


            // cv::imshow( "disparity" , disparity_for_visualization_gray );
            #else
            // Direct Load depth
            string depth_image_fname = base_path+"/cerebro_stash/depth_image__" + std::to_string(stamp.toNSec()) + ".jpg.png";
            _retrive_debug_( cout << "\tload depth_image_fname: " << depth_image_fname << endl; )
            cv::Mat depth_image = cv::imread( depth_image_fname, -1 );
            if( !depth_image.data ) {
                cout << TermColor::RED() << "[retrive_data_from_json_datanode]ERROR cannot load depth-image...return false\n" << TermColor::RESET();
                return false;
            }
            _retrive_debug_( cout << "\tdepth_image " << MiscUtils::cvmat_info( depth_image ) << endl; )
            depth_map = depth_image;
            #if 0
            depth_image.convertTo( depth_map, CV_32FC1 , 1.0/1000. );


            disparity_for_visualization_gray = cv::Mat::zeros( depth_image.rows, depth_image.cols, CV_8UC3 );
            FalseColors false_c;
            for( int r=0 ; r<depth_map.rows ; r++ ) {
                for( int c=0 ; c<depth_map.cols ; c++ ) {
                    float val = depth_map.at<float>( r,c );
                    // cout << val << ", ";

                    cv::Scalar _c = false_c.getFalseColor( val/3.0 );
                    disparity_for_visualization_gray.at<cv::Vec3b>(r,c)[0] = _c[0];
                    disparity_for_visualization_gray.at<cv::Vec3b>(r,c)[1] = _c[1];
                    disparity_for_visualization_gray.at<cv::Vec3b>(r,c)[2] = _c[2];
                }
                // cout << endl;
            }
            // cv::applyColorMap(depth_map, disparity_for_visualization_gray, cv::COLORMAP_JET);
            // cv::imshow( "left_image" , left_image );
            // cv::imshow( "depth COLORMAP_JET" , disparity_for_visualization_gray );
            // cv::waitKey(0);
            #endif


            #endif


            #if 0 // save to file (opencv yaml)
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

        _retrive_info_( cout << TermColor::GREEN() << "[retrive_data_from_json_datanode]DONE t=" << data_node["stampNSec"] << TermColor::RESET() << endl; )
        return true;
    }


};
