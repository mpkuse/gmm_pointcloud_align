


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

#include "XLoader.h"



bool process_this_datanode( XLoader& xloader, json data_node,
    Matrix4d& toret__wTc, MatrixXd& toret__cX, MatrixXd& toret__uv,
    cv::Mat& toret__left_image, cv::Mat& toret__depth_image,
    cv::Mat& toret_viz_slic
    )
{
    ros::Time stamp;
    Matrix4d w_T_c;
    cv::Mat left_image, right_image;
    cv::Mat depth_map, disparity_for_visualization_gray;

    bool status = xloader.retrive_data_from_json_datanode( data_node,
                    stamp, w_T_c,
                    left_image, right_image, depth_map, disparity_for_visualization_gray );
    if( status == false )  {
        cout << "[process_this_datanode] not processing this node because `retrive_data_from_json_datanode` returned false\n";
        return false;
    }
    // imshow here if need be


    //-------------- now we have all the needed data ---------------------//
    // USE:
    // stamp, w_T_c, left_image, right_image, out3D, depth_map
    //--------------------------------------------------------------------//
    ElapsedTime t_slic;

    // --------SLIC
    // SlicClustering slic_obj;
    // slic_obj.generate( left_image, out3D, ddd, dddd );
    int w = left_image.cols, h = left_image.rows;
    int nr_superpixels = 400;
    int nc = 40;
    double step = sqrt((w * h) / ( (double) nr_superpixels ) ); ///< step size per cluster
    cout << "===\n";
    cout << "\tSLIC Params:\n";
    cout << "\tstep size per cluster: " << step << endl;
    cout << "\tWeight: " << nc << endl;
    cout << "\tNumber of superpixel: "<< nr_superpixels << endl;


    SlicClustering slic_obj;
    t_slic.tic();
    // slic_obj.generate_superpixels( left_image, depth_map, step, nc );
    slic_obj.generate_superpixels( left_image, depth_map, step );
    cout << TermColor::BLUE() << "\tgenerate_superpixels() t_slic (ms): " << t_slic.toc_milli()
         << " and resulted in " << slic_obj.retrive_nclusters() << " superpixels " << TermColor::RESET() << endl;
     cout << "===\n";

    //------------ SLIC viz
    #if 1
    cv::Mat imA;
    if( left_image.channels() == 3 )
        imA = left_image.clone();
    else
        cv::cvtColor(left_image, imA, cv::COLOR_GRAY2BGR);

    // slic_obj.colour_with_cluster_means( imA );
    slic_obj.display_contours( left_image,  cv::Scalar(0,0,255), imA );
    slic_obj.display_center_grid( imA, cv::Scalar(0,255,0) );
    // slic_obj.display_center_grid();

    //----------SLIC Retrive
    // retrive useful data
    MatrixXd sp_uv = slic_obj.retrive_superpixel_uv(true, false);
    MatrixXd sp_3dpts___cX = slic_obj.retrive_superpixel_XYZ( true );
    int nclusters = slic_obj.retrive_nclusters();
    // MatrixXd sp_normals = slic_obj.retrive_superpixel_localnormals(); //return estimated exponential surface normals at cluster centers

    // cv::imshow( "slic" , imA );
    // toret_viz_slic = imA.clone();
    toret_viz_slic = imA;

    #endif

    //
    //--- Return
    toret__wTc = w_T_c;
    toret__cX  = sp_3dpts___cX;
    toret__uv  = sp_uv;

    toret__left_image = left_image;
    toret__depth_image = depth_map;

    return true;

}


bool process_key_press( char ch, vector<int>& idx_ptr )
{
    // cout << "[process_key_press] ch = "<< ch << endl;
    if( ch == 'a' ) {
        idx_ptr[0]+=1;
        return true;
    }

    if( ch == 'z' ) {
        idx_ptr[0]+=10;
        return true;
    }

    //-----
    if( ch == 's' ) {
        idx_ptr[1]+=1;
        return true;
    }

    if( ch == 'x' ) {
        idx_ptr[1]+=10;
        return true;
    }


    //-----
    if( ch == 'd' ) {
        idx_ptr[2]+=1;
        return true;
    }

    if( ch == 'c' ) {
        idx_ptr[2]+=10;
        return true;
    }


    //-----
    if( ch == 'f' ) {
        idx_ptr[3]+=1;
        return true;
    }

    if( ch == 'v' ) {
        idx_ptr[3]+=10;
        return true;
    }



    if( ch == 'm' ) {
        // new sequence
        idx_ptr.push_back(0);
        return true;
    }

    if( ch == 'n' ) {
        int last_ptr_idx = *(idx_ptr.rbegin());
        idx_ptr.push_back( last_ptr_idx );
        return true;
    }


    return true; //TODO: return false and this indicates to idx_ptr is unchanged

}

void print_idx_ptr_status( const vector<int>& idx_ptr, const json& STATE, cv::Mat& status )
{
    assert( status.data );

    #if 0
    cout << "#idx_ptr=" << idx_ptr.size() << "\t:";
    for( int i=0 ; i<(int)idx_ptr.size() ; i++ )
    {
        cout << idx_ptr[i] << ", ";
    }
    cout << endl;
    #endif

    string msg = "#IDX_PTR=" + to_string( idx_ptr.size() ) + ";";
    for( int i=0 ; i<(int)idx_ptr.size() ; i++ ) {
        msg+= "idx_ptr"+to_string( i ) + "--->" + to_string( idx_ptr[i] ) + "    ";

        ros::Time stamp = ros::Time().fromNSec( STATE["DataNodes"][  idx_ptr[i]  ]["stampNSec"] );
        msg+= "t="+  to_string( stamp.toNSec() );

        msg += ";";
    }

    MiscUtils::append_status_image( status, msg, .6, cv::Scalar(0,0,0), cv::Scalar(255,255,255) );
    // cv::imshow( "status", status );
    // cout << "[print_idx_ptr_status]msg=" << msg <<endl;

}


void print_processingseries_status( const map<  int,  vector<int>   > & all_i, cv::Mat& status )
{
    string msg = "#PROCESSED_SERIES=" + to_string( all_i.size() ) + ";";
    for( auto it=all_i.begin() ; it!= all_i.end() ; it++ )
    {
        msg += "series#" + to_string( it->first ) + ": ";
        msg += to_string(   *(it->second.begin()) ) + "---->";
        msg += to_string(   *(it->second.rbegin()) );
        msg += "   nitems=" + to_string( it->second.size() );
        msg += ";";
    }
    MiscUtils::append_status_image( status, msg, .6, cv::Scalar(0,0,0), cv::Scalar(0,255,0) );


}


void print_surfelmaps_status( const map< int, SurfelMap* > vec_map, cv::Mat& status )
{
    string msg = "#SurfelMaps=" + to_string( vec_map.size() ) + ";";
    for( auto it=vec_map.begin() ; it!= vec_map.end() ; it++ )
    {
        msg += "surfelmap#" + to_string( it->first ) + ": ";
        // msg += to_string(   *(it->second.begin()) ) + "---->";
        // msg += to_string(   *(it->second.rbegin()) );
        msg += "number of 3dpts = "+to_string( it->second->surfelSize() );
        msg += ";";
    }
    MiscUtils::append_status_image( status, msg, .6, cv::Scalar(0,0,0), cv::Scalar(0,255,0) );

}


void print_usage( cv::Mat& status )
{
    string msg = "# Usage;";
    msg += "a,s,d,f....: step by 1,  the the idx_ptr;";
    msg += "a,s,d,f....: step by 10, the the idx_ptr;";
    msg += "q,w,e,r.....: Process the series;";
    msg += "n: fork new idx_ptr, m: new idx_ptr from 0;";
    msg += "ESC: quit;";
    MiscUtils::append_status_image( status, msg, .45, cv::Scalar(0,0,0), cv::Scalar(255,255,255) );

}

int main( int argc, char ** argv )
{
    //
    // Ros INIT
    //
    ros::init(argc, argv, "local_ptcloud");
    ros::NodeHandle nh;
    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("marker", 1000);

    //
    // Load Camera (camodocal)
    //
    XLoader xloader;
    xloader.load_left_camera();
    xloader.load_right_camera();
    xloader.load_stereo_extrinsics();
    xloader.make_stereogeometry();


    //
    // Load JSON
    //
    json STATE = xloader.load_json();



    //
    //
    //
    vector<int> idx_ptr;
    idx_ptr.push_back(0);

    map<  int,  vector<int>         > all_i; //other vector to hold multiple sequences
    map<  int,  vector<Matrix4d>    > all_odom_poses;
    map<  int,  vector<MatrixXd>    > all_sp_cX;

    map< int, SurfelMap* > vec_map;

    // inf loop
    while( ros::ok() )
    {
        ros::spinOnce();


        for( int i=0 ; i<(int)idx_ptr.size() ; i++ ) // I can have multiple idx
        {
            assert( idx_ptr[i] >= 0 && idx_ptr[i]<(int)STATE["DataNodes"].size() );
            json data_node;
            ros::Time stamp;

            while( true ) {
                // step until next `loadable` is found
                data_node = STATE["DataNodes"][idx_ptr[i]];
                stamp = ros::Time().fromNSec( data_node["stampNSec"] );

                if( data_node["isPoseAvailable"] == false || data_node["isKeyFrame"] == false ) {
                    idx_ptr[i]++;
                    continue;
                }
                break;
            }


            // show image
            string imleft_fname = xloader.base_path+"/cerebro_stash/left_image__" + std::to_string(stamp.toNSec()) + ".jpg";
            // cout << "Load Image: " << imleft_fname << endl;
            cv::Mat left_image = cv::imread( imleft_fname, 0 );
            if( !left_image.data ) {
                cout << TermColor::RED() << "[main]ERROR cannot load image...return false\n" << TermColor::RESET();
                exit(1);
            }
            #if 0
            cout << TermColor::iYELLOW() << "idx_ptr[" << i << "]" << idx_ptr[i] << TermColor::RESET() << "\t";
            cout << "seq=" << data_node["seq"] << "\t";
            cout << "t=" << stamp << "\t";
            cout << "left_image"  << MiscUtils::cvmat_info( left_image ) << endl;
            #endif

            string win_name = "left_image_series#"+to_string( i );
            cv::imshow( win_name.c_str(), left_image );
        }



        // Statuis image
        cv::Mat status_im = cv::Mat::zeros( 10, 500, CV_8UC3 );
        print_idx_ptr_status( idx_ptr, STATE, status_im );
        print_processingseries_status( all_i, status_im );
        print_surfelmaps_status( vec_map, status_im );
        print_usage( status_im );
        cv::imshow( "status", status_im );


        // char ch = (char) cv::waitKey(0);
        char ch = cv::waitKey(0) & 0xEFFFFF;
        // cout << "main: you pressed: " << ch << endl;
        bool status = process_key_press( ch, idx_ptr );
        assert( status );
        if( ch == 27 ) {
            cout << "BREAK.....\n";
            break;
        }


        // process the idx_ptr
        if( ch == 'q' || ch == 'w' || ch == 'e' || ch == 'r' ) {
            int seriesI = 0;
            switch (ch) {
                case 'q':
                seriesI = 0; break;
                case 'w':
                seriesI = 1; break;
                case 'e':
                seriesI = 2; break;
                case 'r':
                seriesI = 3; break;
                default: assert( false );
            }
            assert( seriesI >= 0 && seriesI<idx_ptr.size() && "You choose to process a series for which the pointer doesnopt exist. The correct way is to press n to start a new pointer and then go by processing it\n" );

            cout << TermColor::BLUE() << "Q PRESSED, PROCESS\n";
            json data_node = STATE["DataNodes"][idx_ptr[seriesI]];


            //--- SLIC
            Matrix4d wTc;
            MatrixXd sp_cX;
            MatrixXd sp_uv;
            cv::Mat left_image_i, depth_map_i;
            cv::Mat viz_slic;

            process_this_datanode( xloader, data_node, wTc, sp_cX, sp_uv, left_image_i, depth_map_i, viz_slic );

            //--- Note this data (ie. output of slic)
            if( all_i.count( seriesI ) == 0 ) {
                // not found, so first in the sequence.
                cout << ">>>This is the first element to be processed in seriesI=" <<seriesI << endl;

                all_i[ seriesI ] = vector<int>();
                all_odom_poses[ seriesI ] = vector<Matrix4d>();
                all_sp_cX[ seriesI ] = vector<MatrixXd>();

                vec_map[ seriesI ] = new SurfelMap( xloader.left_camera );
            } else {
                cout << ">>>Number of elements processed in this seriesI=" << seriesI << " is : " << all_i[seriesI].size() << endl;
            }
            all_i[ seriesI ].push_back( idx_ptr[seriesI] );
            all_odom_poses[ seriesI ].push_back( wTc );
            all_sp_cX[ seriesI ].push_back( sp_cX ); //TODO: not needed


            //--- viz
            string win_name = "left_image_series#"+to_string( seriesI );
            cv::imshow( win_name.c_str(), viz_slic );

            cv::Scalar color = FalseColors::randomColor(seriesI);

            MatrixXd sp_wX = all_odom_poses[ seriesI ][0].inverse() * wTc * sp_cX; //< w_T_c0 * w_T_ci * cX
            RosPublishUtils::publish_3d( marker_pub, sp_wX,
                "raw_pts"+to_string(seriesI), all_i[ seriesI ].size(),
                float(color[2]),float(color[1]),float(color[0]), 0.4  );


            visualization_msgs::Marker cam_viz_i;
            cam_viz_i.ns = "cam_viz"+to_string(seriesI);
            cam_viz_i.id = all_i[ seriesI ].size();
            RosMarkerUtils::init_camera_marker( cam_viz_i, 4.0 );
            Matrix4d c0_T_ci =  all_odom_poses[ seriesI ][0].inverse() * wTc;
            RosMarkerUtils::setpose_to_marker( c0_T_ci, cam_viz_i  );
            RosMarkerUtils::setcolor_to_marker( color[2]/255., color[1]/255., color[0]/255., 1.0, cam_viz_i );
            marker_pub.publish( cam_viz_i );



            //---- SurfelFuse & Retrive
            ElapsedTime t_fusesurfels; t_fusesurfels.tic();
            vec_map[seriesI]->fuse_with( idx_ptr[seriesI],  wTc, sp_cX, sp_uv, left_image_i, depth_map_i );
            cout << "fuse returned in (ms) " << t_fusesurfels.toc_milli() << endl;

            MatrixXd __p = all_odom_poses[ seriesI ][0].inverse() * vec_map[seriesI]->surfelWorldPosition();


            //---- viz : publish __p to rviz
            #if 1 // fixed color regime
            RosPublishUtils::publish_3d( marker_pub, __p,
                "surfels"+to_string(vec_map.size()), 0,
                float(color[2]), float(color[1]), float(color[0]), float(1.0), 2.0 );
            #else
            RosPublishUtils::publish_3d( marker_pub, __p,
                "surfels"+to_string(vec_map.size()), 0,
                1, -1, 5,
                2.0 );
            #endif


            //--- waitkey (to show results)
            cout << TermColor::iGREEN() << "Showing results, press <space> to continue\n" << TermColor::RESET();
            ros::spinOnce();
            cv::waitKey(0);
            idx_ptr[seriesI]++; //move the pointer ahead by 1


            cout << TermColor::RESET() << endl;
        }


        // various alignment methods
        if( ch == '1' ) {
            // ICP, on map0, map1
            cout << "ch == 1, not implemented\n";
            exit(2);
        }

    } //while(ros::ok())


}