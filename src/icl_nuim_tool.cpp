// This will let you go over the ICL-NUIM dataset


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

#include "surfel_fusion/surfel_map.h"

// Camodocal
#include "camodocal/camera_models/Camera.h"
#include "camodocal/camera_models/PinholeCamera.h"
#include "camodocal/camera_models/CameraFactory.h"

#include "utils/PointFeatureMatching.h"
#include "ICLNUIMLoader.h"

#include "utils/RosMarkerUtils.h"
#include "utils/RawFileIO.h"
#include "utils/TermColor.h"
#include "utils/ElapsedTime.h"

void print_pointer_status( const vector<ICLNUIMLoader>& all_loaders, vector<int>& curr_ptr,
    cv::Mat& status )
{
    assert( status.data &&  all_loaders.size() == curr_ptr.size()  );
    string msg = "";
    msg += "# curr_ptr;";
    for( int i=0 ; i<all_loaders.size() ; i++ )
    {
        msg += all_loaders[i].DB_NAME + "_" + to_string( all_loaders[i].DB_IDX ) + ": ";
        msg += to_string( curr_ptr[i] ) + " of " + to_string( all_loaders[i].len() ) + ";";
    }
    MiscUtils::append_status_image( status, msg, .6, cv::Scalar(0,0,0), cv::Scalar(255,255,255) );
}


void print_surfelmaps_status( const map< int, SurfelMap* > vec_map, cv::Mat& status )
{
    string msg = "#SurfelMaps=" + to_string( vec_map.size() ) + ";";
    for( auto it=vec_map.begin() ; it!= vec_map.end() ; it++ )
    {
        msg += "surfelmap#" + to_string( it->first ) + ": ";
        // msg += to_string(   *(it->second.begin()) ) + "---->";
        // msg += to_string(   *(it->second.rbegin()) );
        msg += "number of 3dpts = "+to_string( it->second->n_surfels() );
        msg += ";";
    }
    MiscUtils::append_status_image( status, msg, .6, cv::Scalar(0,0,0), cv::Scalar(0,255,0) );

}

void print_usage( cv::Mat& status )
{
    string msg = "# Usage;";
    msg += "a,s,d,f....: step by 1,  the the idx_ptr;";
    msg += "z,x,c,v....: step by 10, the the idx_ptr;";
    msg += "q,w,e,r.....: Process the series;";
    // msg += "1:Draw random image pair, map point feature to surfels;";
    msg += "2:viz 3d model normals;";
    msg += "ESC: quit;";
    MiscUtils::append_status_image( status, msg, .45, cv::Scalar(0,0,0), cv::Scalar(255,255,255) );

}



int random_in_range( int start, int end )
{
    return ( rand() % (end-start) ) + start;
}



// Given the json state, and the index of 2 images , gives the image correspondences
// and the 3d points of those correspondences
//      icl_nuim_camera: camera, this is needed for backprojection
//      xloader_a, xloader_b: loaders of the 2 sequences.
//      idx0, idx1 : Index0, Index1. These indices will be looked up from STATE.
//      uv_a, uv_b [Output] : 2d point. xy (ie. col, row) in image plane. 3xN
//      aX, bX     [Output] : 3d points of uv_a, uv_b (respectively) expressed in co-ordinates of the camera. 4xN
bool image_correspondences(
    camodocal::CameraPtr icl_nuim_camera,
    ICLNUIMLoader& xloader_a,  ICLNUIMLoader& xloader_b,
    int idx_a, int idx_b,
    MatrixXd& uv_a, MatrixXd& uv_b,
    MatrixXd& aX, MatrixXd& bX, vector<bool>& valids )
{
    // -- Retrive Image Data
    cv::Mat image_a, depth_a, depth_a_falsecolor;
    xloader_a.retrive_im_depth( idx_a, image_a, depth_a, depth_a_falsecolor );
    xloader_a.print_info();

    cv::Mat image_b, depth_b, depth_b_falsecolor;
    xloader_b.retrive_im_depth( idx_b, image_b, depth_b, depth_b_falsecolor );
    xloader_b.print_info();


    #if 1
    cout << TermColor::YELLOW() << "+++++++++++++++++++image_correspondences+++++++++++++++++++++++" << TermColor::RESET() << endl;
    // cout << TermColor::GREEN() << "=== Showing the 1st image of both seq. Press any key to continue\n"<< TermColor::RESET();
    cv::imshow( "image_a", image_a );
    cv::imshow( "image_b", image_b );
    cv::imshow( "depth_a_falsecolor", depth_a_falsecolor );
    cv::imshow( "depth_b_falsecolor", depth_b_falsecolor );
    double d_min, d_max;
    cv::minMaxLoc(depth_a, &d_min, &d_max);
    cout << "depth_a: minmax=" << d_min << "," << d_max << "\t";
    cv::minMaxLoc(depth_b, &d_min, &d_max);
    cout << "depth_b: minmax=" << d_min << "," << d_max << "\n";
    #endif

    // -- GMS Matcher
    cout << TermColor::GREEN() << "=== GMS Matcher for idx_a="<< idx_a << ", idx_b=" << idx_b << TermColor::RESET() << endl;
    StaticPointFeatureMatching::gms_point_feature_matches( image_a, image_b, uv_a, uv_b );
    cout << "uv_a: " << uv_a.rows() << "x" << uv_a.cols() << "\t";
    cout << "uv_b: " << uv_b.rows() << "x" << uv_b.cols() << "\t";

    if( uv_a.cols() < 50 ) {
        cout << TermColor::YELLOW() << "\nGMSMatcher produced fewer than 50 point matches, return false\n" << TermColor::RESET();
        return false;
    }

    cv::Mat dst_gmsmatcher;
    MiscUtils::plot_point_pair( image_a, uv_a, idx_a,
                                image_b, uv_b, idx_b, dst_gmsmatcher, 3, "gms plot (resize 0.5)" );
    cv::resize(dst_gmsmatcher, dst_gmsmatcher, cv::Size(), 0.5, 0.5);
    cv::imshow( "GMSMatcher", dst_gmsmatcher );

    #if 1
    // -- 3D Points from depth image at the correspondences
    // vector<bool> valids;
    StaticPointFeatureMatching::make_3d_3d_collection__using__pfmatches_and_depthimage(
        icl_nuim_camera,
        uv_a, depth_a, uv_b, depth_b,
        aX, bX, valids
    );

    int nvalids = 0;
    for( int i=0 ; i<valids.size() ; i++ )
        if( valids[i]  == true )
            nvalids++;
    cout << "nvalids=" << nvalids << " of total=" << valids.size() << "\t";

    cout << "aX: " << aX.rows() << "x" << aX.cols() << "\t";
    cout << "bX: " << bX.rows() << "x" << bX.cols() << endl;

    if( nvalids < 50 ) {
        cout << TermColor::YELLOW() << "GMSMatcher produced fewer than 50 valid (points where good depth value) point matches, return false\n" << TermColor::RESET();
        return false;
    }
    #endif
    return true;


}

#if 0
int main()
{
    camodocal::CameraPtr icl_nuim_camera;
    icl_nuim_camera = camodocal::CameraFactory::instance()->generateCamera( camodocal::Camera::PINHOLE, "xxx", cv::Size(480,640) );
    vector<double>parameterVec;
    icl_nuim_camera->writeParameters( parameterVec );
    parameterVec[4] = 481.20; //fx
    parameterVec[5] = 480.; //fy
    parameterVec[6] = 319.50; //cx
    parameterVec[7] = 239.50; //cy
    icl_nuim_camera->readParameters( parameterVec );
    cout << "Cam:\n" << icl_nuim_camera->parametersToString() << endl;

    // icl_nuim_camera = camodocal::CameraFactory::instance()->generateCameraFromYamlFile("fff");
    // icl_nuim_camera = new camodocal::PinholeCamera( "icl_nuim_camera", 640, 480, 0,0,0,0, 481.20, 480., 319.50, 239.50 );
}
#endif


#if 0
int main()
{
    ros::Time::init();

    string DB_BASE = "/Bulk_Data/ICL_NUIM_RGBD/";
    //living_room, office_room
    string DB_NAME = "living_room";
    // string DB_NAME = "office_room";
    int DB_IDX = 0; //0,1,2,3,


    ICLNUIMLoader xloader( DB_BASE, DB_NAME, DB_IDX );
    xloader.load_gt_pose();

}
#endif

#if 0
int main()
{
    ros::Time::init();

    //-- Camera - got this info from Handa's paper (ICRA2014): https://www.doc.ic.ac.uk/~ahanda/VaFRIC/icra2014.pdf
    camodocal::CameraPtr icl_nuim_camera;
    icl_nuim_camera = camodocal::CameraFactory::instance()->generateCamera( camodocal::Camera::PINHOLE, "icl_nuim_camera", cv::Size(640, 480) );
    vector<double>parameterVec;
    icl_nuim_camera->writeParameters( parameterVec );
    parameterVec[4] = 481.20; //fx
    parameterVec[5] = 480.; //fy
    parameterVec[6] = 319.50; //cx
    parameterVec[7] = 239.50; //cy
    icl_nuim_camera->readParameters( parameterVec );
    cout << "icl_nuim_camera:\n" << icl_nuim_camera->parametersToString() << endl;


    //-- Loader
    string DB_BASE = "/Bulk_Data/ICL_NUIM_RGBD/";
    //living_room, office_room
    string DB_NAME = "living_room";
    // string DB_NAME = "office_room";
    int DB_IDX = 0; //0,1,2,3,

    ICLNUIMLoader xloader( DB_BASE, DB_NAME, DB_IDX );
    xloader.set_camera_intrinsics( parameterVec[4], parameterVec[5], parameterVec[6], parameterVec[7] );


    cv::Mat im, depth, depth_falsecolor;
    xloader.retrive_im_depth( 0, im, depth, depth_falsecolor );
    cv::imshow( "im", im );
    cv::imshow( "depth", depth_falsecolor );

    cv::waitKey(0);
}
#endif

#if 1
int main(int argc, char ** argv )
{
    cout << TermColor::GREEN()
         << "============================\n";
    cout << "===== ICL_NUIM_Dataset =====\n";
    cout << "============================\n" << TermColor::RESET() ;

    //--- ROS INIT
    ros::init(argc, argv, "ICL_NUIM_Dataset");
    ros::NodeHandle nh;
    ros::Publisher marker_pub = nh.advertise<visualization_msgs::Marker>("marker", 1000);


    //-- Camera - got this info from Handa's paper (ICRA2014): https://www.doc.ic.ac.uk/~ahanda/VaFRIC/icra2014.pdf
    camodocal::CameraPtr icl_nuim_camera;
    icl_nuim_camera = camodocal::CameraFactory::instance()->generateCamera( camodocal::Camera::PINHOLE, "icl_nuim_camera", cv::Size(640, 480) );
    vector<double>parameterVec;
    icl_nuim_camera->writeParameters( parameterVec );
    parameterVec[4] = 481.20; //fx
    parameterVec[5] = 480.; //fy
    parameterVec[6] = 319.50; //cx
    parameterVec[7] = 239.50; //cy
    icl_nuim_camera->readParameters( parameterVec );
    cout << "icl_nuim_camera:\n" << icl_nuim_camera->parametersToString() << endl;


    //--- Loader
    string DB_BASE = "/Bulk_Data/ICL_NUIM_RGBD/";
    //living_room, office_room
    // string DB_NAME = "living_room";
    string DB_NAME = "office_room";
    // int DB_IDX = 0; //0,1,2,3,

    vector<ICLNUIMLoader> all_loaders;
    vector<int> curr_ptr;
    vector<bool> changed;
    for( int seriesI=0 ; seriesI<4 ; seriesI++ )
    {
        ICLNUIMLoader xloader( DB_BASE, DB_NAME, seriesI );
        xloader.set_camera_intrinsics( parameterVec[4], parameterVec[5], parameterVec[6], parameterVec[7] );
        // xloader.print_info();
        all_loaders.push_back( xloader );

        curr_ptr.push_back( 0 );
        changed.push_back( true );
    }




    //---
    map<  int,  vector<int>         > all_i; //other vector to hold multiple sequences
    map<  int,  vector<Matrix4d>    > all_odom_poses;
    map< int, SurfelMap* > vec_surf_map;


    //--- control loop
    while( ros::ok() )
    {
        ros::spinOnce();


        cv::Mat status_im = cv::Mat::zeros( 10, 500, CV_8UC3 );
        print_pointer_status( all_loaders, curr_ptr, status_im );
        print_surfelmaps_status( vec_surf_map, status_im );
        print_usage( status_im );
        cv::imshow( "status", status_im );

        // imshow
        for( int seriesI=0 ; seriesI<4 ; seriesI++ )
        {
            cv::Mat im, depth, depth_falsecolor;
            if( changed[seriesI] ) {
                changed[seriesI] = false;

                all_loaders[seriesI].retrive_im( curr_ptr[seriesI], im );
                // all_loaders[seriesI].retrive_im_depth( curr_ptr[seriesI], im, depth );
                // all_loaders[seriesI].retrive_im_depth( curr_ptr[seriesI], im, depth, depth_falsecolor );

                string winname = all_loaders[seriesI].DB_NAME+"_"+to_string( seriesI );
                cv::imshow( winname.c_str(), im );

                // cv::imshow( (winname+"(depth_map)").c_str(), depth_falsecolor );


            }
        }

        char ch = cv::waitKey(0) & 0xEFFFFF;
        if( ch == 27 ) {
            cout << "ESC (break)...\n";
            break;
        }
        if( ch == 'a' || ch == 's' || ch == 'd' || ch == 'f' )
        {
            int seriesI = 0;
            switch( ch )
            {
                case 'a':
                    seriesI = 0;
                    break;
                case 's':
                    seriesI = 1;
                    break;
                case 'd':
                    seriesI = 2;
                    break;
                case 'f':
                    seriesI = 3;
                    break;

            }

            if( curr_ptr[seriesI]+1 < all_loaders[seriesI].len() ) {
                changed[seriesI] = true;
                curr_ptr[seriesI]++;
            }
            continue;

        }

        if( ch == 'z' || ch == 'x' || ch == 'c' || ch == 'v' )
        {
            int seriesI = 0;
            switch( ch )
            {
                case 'z':
                    seriesI = 0;
                    break;
                case 'x':
                    seriesI = 1;
                    break;
                case 'c':
                    seriesI = 2;
                    break;
                case 'v':
                    seriesI = 3;
                    break;

            }
            if( curr_ptr[seriesI]+10 < all_loaders[seriesI].len() ) {
                changed[seriesI] = true;
                curr_ptr[seriesI]+=10;
            }
            continue;

        }

        if( ch == 'q' || ch == 'w' || ch == 'e' || ch == 'r' )
        {
            int seriesI = 0;
            switch( ch )
            {
                case 'q':
                    seriesI = 0;
                    break;
                case 'w':
                    seriesI = 1;
                    break;
                case 'e':
                    seriesI = 2;
                    break;
                case 'r':
                    seriesI = 3;
                    break;
            }
            cout << TermColor::BLUE() << "``" << ch << "`` PRESSED, PROCESS seriesI=" << seriesI << "\n" << TermColor::RESET();

            //-- Retrive Data
            cv::Mat im, depth;
            Matrix4d wx_T_cx;
            all_loaders[seriesI].retrive_im_depth( curr_ptr[seriesI], im, depth );
            all_loaders[seriesI].retrive_pose( curr_ptr[seriesI], wx_T_cx );
            ros::Time dxc__stamp = all_loaders[seriesI].idx_to_stamp(curr_ptr[seriesI]);

            //-- Note Pose Info in processed
            if( all_i.count( seriesI ) == 0 ) {
                // not found, so first in the sequence.
                cout << ">>>This is the first element to be processed in seriesI=" <<seriesI << endl;

                all_i[ seriesI ] = vector<int>();
                all_odom_poses[ seriesI ] = vector<Matrix4d>();

                #if 0
                vec_surf_map[ seriesI ] = new SurfelMap( true ); //TODO pass fx,fy,cx,cy, near,far as params
                #else
                // width, height, (camera intrinsics) fx, fy, cx, cy, far, near

                vec_surf_map[ seriesI ] = new SurfelMap( icl_nuim_camera->imageWidth(), icl_nuim_camera->imageHeight(),
                        parameterVec[4],
                        parameterVec[5],
                        parameterVec[6],
                        parameterVec[7],
                        6, .5 );
                #endif

            }
            else {
                cout << ">>>Number of elements processed in this seriesI=" << seriesI << " is : " << all_i[seriesI].size() << endl;
            }
            all_i[ seriesI ].push_back( curr_ptr[seriesI] );
            all_odom_poses[ seriesI ].push_back( wx_T_cx );



            //------------ Wang Kaixuan's Dense Surfel Mapping -------------------//
            vec_surf_map[ seriesI ]->image_input( dxc__stamp, im );
            vec_surf_map[ seriesI ]->depth_input( dxc__stamp, depth );
            vec_surf_map[ seriesI ]->camera_pose__w_T_ci__input( dxc__stamp, wx_T_cx );
            cout << "Input pose for Kaixuan : " << PoseManipUtils::prettyprintMatrix4d( wx_T_cx ) << endl;

            // : 3d point retrive
            cout << "n_active_surfels = " << vec_surf_map[ seriesI ]->n_active_surfels() << "\t";
            cout << "n_fused_surfels = " << vec_surf_map[ seriesI ]->n_fused_surfels() << "\t";
            cout << "n_surfels = " << vec_surf_map[ seriesI ]->n_surfels() << "\n";

            MatrixXd _w_X = vec_surf_map[ seriesI ]->get_surfel_positions();
            cout << "_w_X : " << _w_X.rows() << "x" << _w_X.cols() << endl;

            cout << "all_odom_poses[ seriesI ][0].inverse() : " << PoseManipUtils::prettyprintMatrix4d( all_odom_poses[ seriesI ][0].inverse() ) << endl;
            MatrixXd __p = all_odom_poses[ seriesI ][0].inverse() * _w_X; // 3d points in this series's 1st camera ref frame
            // cout << "__p\n" << __p << endl;

            //  vec_surf_map[ seriesI ]->print_persurfel_info( 1 );

            //------------ END Wang Kaixuan's Dense Surfel Mapping -------------------//


            cv::Scalar color = FalseColors::randomColor(seriesI);
            //-- Viz pose
            visualization_msgs::Marker cam_viz_i;
            cam_viz_i.ns = "cam_viz"+to_string(seriesI);
            cam_viz_i.id = all_i[ seriesI ].size();
            RosMarkerUtils::init_camera_marker( cam_viz_i, 4.0 );
            Matrix4d c0_T_ci;
            c0_T_ci =  (all_odom_poses[ seriesI ][0]).inverse() * wx_T_cx;
            RosMarkerUtils::setpose_to_marker( c0_T_ci, cam_viz_i  );
            RosMarkerUtils::setcolor_to_marker( color[2]/255., color[1]/255., color[0]/255., 1.0, cam_viz_i );
            marker_pub.publish( cam_viz_i );

            //-- PLOT 3D points
            RosPublishUtils::publish_3d( marker_pub, __p,
                "surfels"+to_string(seriesI), 0,
                float(color[2]), float(color[1]), float(color[0]), float(1.0), 1.5 );

            if( curr_ptr[seriesI]+5 < all_loaders[seriesI].len() ) {
                changed[seriesI] = true;
                curr_ptr[seriesI]+=5;
            }

            // changed[seriesI] = true;

        }

        if( ch == '0' ) //print info on all_i, all_odom_poses
        {
            for( auto it = all_i.begin() ; it!= all_i.end() ; it++ )
            {
                cout << "all_i[" << it->first << "] (len=" << it->second.size() << "): ";
                for( int i=0 ; i<it->second.size() ; i++ )
                {
                    cout << it->second[i] << ", ";
                }
                cout << endl;
            }

            for( auto it = all_odom_poses.begin() ; it!= all_odom_poses.end() ; it++ )
            {
                cout << "all_odom_poses[" << it->first << "] (len=" << it->second.size() << "):\n";
                for( int i=0 ; i<it->second.size() ; i++ )
                {
                    cout << "\ti=" << i << "  " << PoseManipUtils::prettyprintMatrix4d( it->second[i] ) << "\n";
                }
                cout << endl;
            }
        }

        if( ch == '1' ) // basic ICP on surfelmap[0] and surfelmap[1]
        {
            cout << TermColor::BLUE() << "1 pressed PROCESS" << TermColor::RESET() << endl;
            int idx_of_series_a = 0;
            int idx_of_series_b = 1;
            assert( int( all_i.size() )  > max(idx_of_series_b,idx_of_series_a) );

            cout << "series#" << idx_of_series_a << ": " << *(all_i[idx_of_series_a].begin()) << ", " << *(all_i[idx_of_series_a].begin()+1) << " --> " << *(all_i[idx_of_series_a].rbegin()) << endl;
            cout << "series#" << idx_of_series_b << ": " << *(all_i[idx_of_series_b].begin()) << ", " <<  *(all_i[idx_of_series_b].begin()+1) << " --> " << *(all_i[idx_of_series_b].rbegin()) << endl;


            vector< MatrixXd > all_sa_c0_X, all_sb_c0_X;
            vector< vector<bool> > all_valids;
            // for( int k=0 ; k< 5 ; k++ )
            while( true )
            {
                int _a = random_in_range( 0, (int)all_i[idx_of_series_a].size() );
                int _b = random_in_range( 0, (int)all_i[idx_of_series_b].size() );

                int idx_a = all_i[idx_of_series_a][_a];
                int idx_b = all_i[idx_of_series_b][_b];


                // --- Image correspondences and 3d points from depth images
                MatrixXd uv_a, uv_b, aX, bX;
                vector<bool> valids;
                bool status = image_correspondences( icl_nuim_camera,
                    all_loaders[idx_of_series_a], all_loaders[idx_of_series_b],
                    idx_a, idx_b, uv_a, uv_b, aX, bX, valids  );


                if( status == false ) {
                    cout << TermColor::RED() << "image_correspondences returned false, skip this sample\n" << TermColor::RESET();
                    continue;
                }

                #if 1

                // --- Change co-ordinate system of aX and bX

                Matrix4d wTa, wTb;
                wTa = all_odom_poses[idx_of_series_a][_a];
                wTb = all_odom_poses[idx_of_series_b][_b];

                cout << "_a = " << _a << "\t_b = " << _b << "\tidx_a=" << idx_a << "\tidx_b=" << idx_b << endl;
                cout << "wTa : " << PoseManipUtils::prettyprintMatrix4d( wTa ) << endl;
                cout << "wTb : " << PoseManipUtils::prettyprintMatrix4d( wTb ) << endl;

                //3d points espressed in 0th camera of the sequence
                MatrixXd sa_c0_X = (all_odom_poses[idx_of_series_a])[0].inverse() *  wTa * aX;
                MatrixXd sb_c0_X = (all_odom_poses[idx_of_series_b])[0].inverse() *  wTb * bX;

                cout << "all_odom_poses[idx_of_series_a])[0].inverse() : " <<
                PoseManipUtils::prettyprintMatrix4d( all_odom_poses[idx_of_series_a][0].inverse() ) << endl;
                cout << "all_odom_poses[idx_of_series_b])[0].inverse() : " << PoseManipUtils::prettyprintMatrix4d( all_odom_poses[idx_of_series_b][0].inverse() ) << endl;
                cout << "sa_c0_X:\n" << sa_c0_X.leftCols(10) << endl;
                cout << "sb_c0_X:\n" << sb_c0_X.leftCols(10) << endl;

                int ggg=0;
                for( int g=0 ; g<sa_c0_X.cols() ; g++ ) {
                    // cout << valids[g] << "\t";
                    if( valids[g] )
                    {
                        ggg++;
                        cout << "g=" << g << "\t";
                        cout << aX.col(g).transpose() << "\t";
                        cout << "<--->\t";
                        cout << bX.col(g).transpose() << "\t";
                        cout << endl;
                    }
                    if( ggg > 10 )
                        break;
                }

                //
                all_sa_c0_X.push_back( sa_c0_X );
                all_sb_c0_X.push_back( sb_c0_X );
                all_valids.push_back( valids );


                // --- Viz
                visualization_msgs::Marker l_mark;
                RosMarkerUtils::init_line_marker( l_mark, sa_c0_X, sb_c0_X , valids);
                RosMarkerUtils::setcolor_to_marker( 1.0, 1.0, 1.0, l_mark );
                l_mark.scale.x *= 0.5;
                l_mark.ns = "pt matches"+to_string(idx_a)+"<->"+to_string(idx_b)+";nvalids="+to_string( MiscUtils::total_true( valids ) );
                marker_pub.publish( l_mark );
                #endif

                // --- Wait Key
                cout << "press `p` for pose computation, press `b` to break, press any other key to keep drawing more pairs\n";
                char ch = cv::waitKey(0) & 0xEFFFFF;
                if( ch == 'b' )
                    break;

                #if 0
                if( ch == 'p' )
                {
                    MatrixXd dst0, dst1;
                    MiscUtils::gather( all_sa_c0_X, all_valids, dst0 );
                    MiscUtils::gather( all_sb_c0_X, all_valids, dst1 );

                    // RawFileIO::write_EigenMatrix( "/app/catkin_ws/src/gmm_pointcloud_align/resources/pointsets/"+to_string(idx_a)+"-"+to_string(idx_b)+"__sa_c0_X.txt", sa_c0_X );
                    // RawFileIO::write_EigenMatrix( "/app/catkin_ws/src/gmm_pointcloud_align/resources/pointsets/"+to_string(idx_a)+"-"+to_string(idx_b)+"__sb_c0_X.txt", sb_c0_X);

                    // Pose Computation
                    Matrix4d a_T_b;
                    // TODO todo need to use valid
                    PoseComputation::closedFormSVD( dst0, dst1, a_T_b );
                    cout << "a_T_b = " << PoseManipUtils::prettyprintMatrix4d( a_T_b ) << endl;

                    MatrixXd AAA = all_odom_poses[ 0 ][0].inverse() * vec_surf_map[ 0 ]->get_surfel_positions();
                    RosPublishUtils::publish_3d( marker_pub, AAA,
                        "AAA", 0,
                        255,0,0, float(1.0), 1.5 );

                    MatrixXd BBB = all_odom_poses[ 1 ][0].inverse() * vec_surf_map[ 1 ]->get_surfel_positions();
                    RosPublishUtils::publish_3d( marker_pub, BBB,
                        "BBB", 0,
                        0,255,0, float(1.0), 1.5 );


                    MatrixXd YYY = a_T_b * BBB;
                    RosPublishUtils::publish_3d( marker_pub, YYY,
                        "tr", 0,
                        0,255,255, float(1.0), 1.5 );


                }
                #endif
            }
            cv::destroyWindow("GMSMatcher");

        }



        if( ch == '2' ) // visualization for normals of a point cloud
        {
            assert( vec_surf_map.size() > 0 );
            int seriesI = 0;
            cout << TermColor::GREEN() << "==== Viz Normals ====\n" << TermColor::RESET();
            MatrixXd wX = vec_surf_map[ seriesI ]->get_surfel_positions();
            MatrixXd __p = all_odom_poses[ seriesI ][0].inverse() * wX;
            MatrixXd normals = vec_surf_map[ seriesI ]->get_surfel_normals();

            vector<cv::Scalar> normals_rep_color;
            FalseColors fp;
            normals_rep_color.clear();
            for( int i=0 ; i<normals.cols() ; i++ ) //make colors for each normal's direction
            {
                // follow calculation for RGB-->Hue
                double _R = normals(0,i), _G = normals(1,i), _B = normals(2,i);
                double hue = 0;
                if( _R >= _G && _R >= _B ) {
                    hue = (_G - _B) / ( _R - min(_G,_B) );
                }

                if( _G >= _R && _G >= _B ) {
                    hue = 2.0 + ( _B - _R ) / ( _G - min(_R,_B) );
                }

                if( _B >= _R && _B >= _G ) {
                    hue = 4.0 + (_R - _G) / ( _B - min(_R,_G));
                }
                hue *= 60;
                if( hue < 0 )
                    hue += 360;

                if( std::isnan( hue ) )
                    hue = 0;
                // cout << "i=" << i << " hue=" << hue << endl;
                normals_rep_color.push_back( fp.getFalseColor( hue / 360. ) );
            }

            cout << "dddd.size() = " << normals_rep_color.size() << endl;
            RosPublishUtils::publish_3d( marker_pub, __p, "normals"+to_string(seriesI), 0, normals_rep_color );

        }



    } // while( ros::ok() )


    // deallocate





}

#endif
