


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
//#include "SurfelMap.h"
//#include "SlicClustering.h"
#include "utils/CameraGeometry.h"
#include "utils/PointFeatureMatching.h"

#include "PoseComputation.h"

//
#include "utils/RosMarkerUtils.h"
#include "utils/RawFileIO.h"
#include "utils/TermColor.h"
#include "utils/ElapsedTime.h"

#include "XLoader.h"

#include "surfel_fusion/surfel_map.h"

#include <theia/theia.h>

void write_to_disk_for_goicp( MatrixXd& __p , const string goicp_fname )
{
    cout << TermColor::iGREEN() << "===  write_to_disk_for_goicp() ===\n" << TermColor::RESET();

    VectorXd m = __p.rowwise().mean();

    VectorXd mx = __p.rowwise().maxCoeff();
    VectorXd mn = __p.rowwise().minCoeff();
    cout << "mean : " << m.transpose() << endl;
    cout << "mx   : " << mx.transpose() << endl;
    cout << "mn   : " << mn.transpose() << endl;
    assert( m(3) == 1.0 );
    m(3) = 0.0;


    cout << "p(before):\n" << __p.leftCols(5) << endl;
    for( int i=0 ; i<__p.cols() ; i++ )
    {
        __p(0,i) = (   (__p(0,i) - mn(0)) / (mx(0) - mn(0)) )*2.0 - 1.0;
        __p(1,i) = (   (__p(1,i) - mn(1)) / (mx(1) - mn(1)) )*2.0 - 1.0;
        __p(2,i) = (   (__p(2,i) - mn(2)) / (mx(2) - mn(2)) )*2.0 - 1.0;
    }
    cout << "p(after):\n" << __p.leftCols(5) << endl;


    ofstream myfile;
    cout << "Open File: " << goicp_fname << endl;
    myfile.open (goicp_fname);
    myfile << __p.cols() << endl;
    for( int i=0 ; i<__p.cols() ; i++ )
    {
        myfile << __p(0,i) << " " << __p(1,i) << " " << __p(2,i) << " " << endl;
    }
    myfile.close();
}

#if 0
// my version of surfel fusion
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
    int nr_superpixels = 1000;
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
#endif

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

        #if 1
        cout << TermColor::YELLOW();
        cout << "series#" << to_string( it->first ) << "\t";
        cout << *(it->second.begin())  << "--->" << *(it->second.rbegin()) << "\t";
        cout << "nitems=" << it->second.size() ;
        cout << endl;
        cout << TermColor::RESET();
        #endif
    }
    MiscUtils::append_status_image( status, msg, .6, cv::Scalar(0,0,0), cv::Scalar(0,255,0) );


}


#if 0
void print_surfelmaps_status( const map< int, SurfelXMap* > vec_map, cv::Mat& status )
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
#endif


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

// Given the json state, and the index of 2 images , gives the image correspondences
// and the 3d points of those correspondences
//      STATE: The loaded json file (can be obtained from checkpoint of cerebro)
//      idx0, idx1 : Index0, Index1. These indices will be looked up from STATE.
//      uv_a, uv_b [Output] : 2d point. xy (ie. col, row) in image plane. 3xN
//      aX, bX     [Output] : 3d points of uv_a, uv_b (respectively) expressed in co-ordinates of the camera. 4xN
bool image_correspondences( const json& STATE, XLoader& xloader,
    int idx_a, int idx_b,
    MatrixXd& uv_a, MatrixXd& uv_b,
    MatrixXd& aX, MatrixXd& bX, vector<bool>& valids )
{
    // -- Retrive Image Data
    json data_node_a = STATE["DataNodes"][idx_a];
    cv::Mat image_a, depth_a;
    bool status = xloader.retrive_image_data_from_json_datanode( data_node_a, image_a, depth_a );

    json data_node_b = STATE["DataNodes"][idx_b];
    cv::Mat image_b, depth_b;
    status = xloader.retrive_image_data_from_json_datanode( data_node_b, image_b, depth_b );


    // cout << TermColor::GREEN() << "=== Showing the 1st image of both seq. Press any key to continue\n"<< TermColor::RESET();
    // cv::imshow( "image_a", image_a );
    // cv::imshow( "image_b", image_b );


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
        xloader.left_camera,
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

int random_in_range( int start, int end )
{
    return ( rand() % (end-start) ) + start;
}


void print_usage( cv::Mat& status )
{
    string msg = "# Usage;";
    msg += "a,s,d,f....: step by 1,  the the idx_ptr;";
    msg += "z,x,c,v....: step by 10, the the idx_ptr;";
    msg += "q,w,e,r.....: Process the series;";
    msg += "n: fork new idx_ptr, m: new idx_ptr from 0;";
    msg += "1:Draw random image pair, map point feature to surfels;";
    msg += "2:viz 3d model normals;";
    msg += "ESC: quit;";
    MiscUtils::append_status_image( status, msg, .45, cv::Scalar(0,0,0), cv::Scalar(255,255,255) );

}


#if 1
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

    // map< int, SurfelXMap* > vec_map;
    map< int, SurfelMap* > vec_surf_map;

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
        // print_surfelmaps_status( vec_map, status_im );
        print_surfelmaps_status( vec_surf_map, status_im );
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

        if( ch == 'q' || ch == 'w' || ch == 'e' || ch == 'r' )
        {
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

            cout << TermColor::BLUE() << "``" << ch << "`` PRESSED, PROCESS\n";
            json data_node = STATE["DataNodes"][idx_ptr[seriesI]];

            //--- Retrive data
            ros::Time dxc__stamp;
            Matrix4d dxc__w_T_c;
            cv::Mat dxc__left_image, dxc__right_image;
            cv::Mat dxc__depth_image, dxc__disparity_for_visualization_gray;
            bool status = xloader.retrive_data_from_json_datanode( data_node,
                            dxc__stamp, dxc__w_T_c,
                            dxc__left_image, dxc__right_image, dxc__depth_image, dxc__disparity_for_visualization_gray );
            if( status == false )  {
                cout << "[main] not processing this node because `retrive_data_from_json_datanode` returned false\n";
                return false;
            }


            //--- Note this data (ie. output of slic)
            if( all_i.count( seriesI ) == 0 ) {
                // not found, so first in the sequence.
                cout << ">>>This is the first element to be processed in seriesI=" <<seriesI << endl;

                all_i[ seriesI ] = vector<int>();
                all_odom_poses[ seriesI ] = vector<Matrix4d>();
                all_sp_cX[ seriesI ] = vector<MatrixXd>();

                vec_surf_map[ seriesI ] = new SurfelMap( true );
            } else {
                cout << ">>>Number of elements processed in this seriesI=" << seriesI << " is : " << all_i[seriesI].size() << endl;
            }
            all_i[ seriesI ].push_back( idx_ptr[seriesI] );
            all_odom_poses[ seriesI ].push_back( dxc__w_T_c );

            #if 1
            //------------ Wang Kaixuan's Dense Surfel Mapping -------------------//
            vec_surf_map[ seriesI ]->image_input( dxc__stamp, dxc__left_image );
            vec_surf_map[ seriesI ]->depth_input( dxc__stamp, dxc__depth_image );
            vec_surf_map[ seriesI ]->camera_pose__w_T_ci__input( dxc__stamp, dxc__w_T_c );

            // : 3d point retrive
            cout << "n_active_surfels = " << vec_surf_map[ seriesI ]->n_active_surfels() << "\t";
            cout << "n_fused_surfels = " << vec_surf_map[ seriesI ]->n_fused_surfels() << "\t";
            cout << "n_surfels = " << vec_surf_map[ seriesI ]->n_surfels() << "\n";

            MatrixXd _w_X = vec_surf_map[ seriesI ]->get_surfel_positions();
            cout << "_w_X : " << _w_X.rows() << "x" << _w_X.cols() << endl;

            MatrixXd __p = all_odom_poses[ seriesI ][0].inverse() * _w_X; // 3d points in this series's 1st camera ref frame
            // cout << "__p\n" << __p << endl;

             vec_surf_map[ seriesI ]->print_persurfel_info( 1 );

            //------------ END Wang Kaixuan's Dense Surfel Mapping -------------------//


            //------ PLOT 3d points in rviz
            // TODO
            #if 1
            cv::Scalar color = FalseColors::randomColor(seriesI);
            #if 1 // fixed color regime
            RosPublishUtils::publish_3d( marker_pub, __p,
                "surfels"+to_string(vec_surf_map.size()-1), 0,
                float(color[2]), float(color[1]), float(color[0]), float(1.0), 1.5 );
            #else
            RosPublishUtils::publish_3d( marker_pub, __p,
                "surfels"+to_string(vec_surf_map.size()), 0,
                1, -1, 5,
                2.0 );
            #endif
            #endif


            //----- PLOT camera
            visualization_msgs::Marker cam_viz_i;
            cam_viz_i.ns = "cam_viz"+to_string(seriesI);
            cam_viz_i.id = all_i[ seriesI ].size();
            RosMarkerUtils::init_camera_marker( cam_viz_i, 4.0 );
            Matrix4d c0_T_ci =  all_odom_poses[ seriesI ][0].inverse() * dxc__w_T_c;
            RosMarkerUtils::setpose_to_marker( c0_T_ci, cam_viz_i  );
            RosMarkerUtils::setcolor_to_marker( color[2]/255., color[1]/255., color[0]/255., 1.0, cam_viz_i );
            marker_pub.publish( cam_viz_i );


            //--- waitkey (to show results)
            cout << TermColor::iGREEN() << "Showing results, press <space> to continue\n" << TermColor::RESET();
            ros::spinOnce();
            cv::waitKey(0);
            idx_ptr[seriesI]++; //move the pointer ahead by 1
            // idx_ptr[seriesI]+=5;


            #endif

        } // end if( ch == 'q' || ch == 'w' || ch == 'e' || ch == 'r' )


        if( ch == '1' ) // basic ICP on surfelmap[0] and surfelmap[1]
        {
            cout << TermColor::BLUE() << "1 pressed PROCESS" << TermColor::RESET() << endl;
            assert( all_i.size() >= 2 );
            cout << "series#0: " << *(all_i[0].begin()) << " --> " << *(all_i[0].rbegin()) << endl;
            cout << "series#1: " << *(all_i[1].begin()) << " --> " << *(all_i[1].rbegin()) << endl;


            vector< MatrixXd > all_sa_c0_X, all_sb_c0_X;
            vector< vector<bool> > all_valids;
            // for( int k=0 ; k< 5 ; k++ )
            while( true )
            {
                int _a = random_in_range( 0, (int)all_i[0].size() );
                int _b = random_in_range( 0, (int)all_i[1].size() );

                int idx_a = all_i[0][_a];
                int idx_b = all_i[1][_b];


                // --- Image correspondences and 3d points from depth images
                MatrixXd uv_a, uv_b, aX, bX;
                vector<bool> valids;
                bool status = image_correspondences( STATE, xloader, idx_a, idx_b, uv_a, uv_b, aX, bX, valids );

                if( status == false ) {
                    cout << TermColor::RED() << "image_correspondences returned false, skip this sample\n" << TermColor::RESET();
                    continue;
                }


                // --- Change co-ordinate system of aX and bX
                Matrix4d wTa, wTb;
                wTa = all_odom_poses[0][_a];
                wTb = all_odom_poses[1][_b];

                //3d points espressed in 0th camera of the sequence
                MatrixXd sa_c0_X = (all_odom_poses[0][0].inverse() * wTa) * aX;
                MatrixXd sb_c0_X = (all_odom_poses[1][0].inverse() * wTb) * bX;

                //
                all_sa_c0_X.push_back( sa_c0_X );
                all_sb_c0_X.push_back( sb_c0_X );
                all_valids.push_back( valids );


                // --- Viz
                visualization_msgs::Marker l_mark;
                RosMarkerUtils::init_line_marker( l_mark, sa_c0_X, sb_c0_X , valids);
                l_mark.scale.x *= 0.5;
                l_mark.ns = "pt matches"+to_string(idx_a)+"<->"+to_string(idx_b)+";nvalids="+to_string( MiscUtils::total_true( valids ) );
                marker_pub.publish( l_mark );


                // --- Wait Key
                cout << "press `p` for pose computation, press `b` to break, press any other key to keep drawing more pairs\n";
                char ch = cv::waitKey(0) & 0xEFFFFF;
                if( ch == 'b' )
                    break;

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
                        "a_T_b x BBB", 0,
                        0,255,255, float(1.0), 1.5 );


                }
            }
            cv::destroyWindow("GMSMatcher");

        }


        if( ch == '2' ) // visualization for normals of a point cloud
        {
            assert( vec_surf_map.size() > 0 );
            cout << TermColor::GREEN() << "==== Viz Normals ====\n" << TermColor::RESET();
            MatrixXd wX = vec_surf_map[ 0 ]->get_surfel_positions();
            MatrixXd __p = all_odom_poses[ 0 ][0].inverse() * wX;
            MatrixXd normals = vec_surf_map[ 0 ]->get_surfel_normals();

            vector<cv::Scalar> ddddd;
            FalseColors fp;
            ddddd.clear();
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
                ddddd.push_back( fp.getFalseColor( hue / 360. ) );
            }

            cout << "dddd.size() = " << ddddd.size() << endl;
            RosPublishUtils::publish_3d( marker_pub, __p, "normals0", 0, ddddd );

        }


        #if 0 // make this to 1 to use mpkuse's mapping implementation. not recommended
        // process the idx_ptr - mpkuse surfel mapping implementation - bad
        if( false && ( ch == 'q' || ch == 'w' || ch == 'e' || ch == 'r' ) )
        {
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

                vec_map[ seriesI ] = new SurfelXMap( xloader.left_camera );
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


            #if 1
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
            #endif


            //--- waitkey (to show results)
            cout << TermColor::iGREEN() << "Showing results, press <space> to continue\n" << TermColor::RESET();
            ros::spinOnce();
            cv::waitKey(0);
            idx_ptr[seriesI]++; //move the pointer ahead by 1


            cout << TermColor::RESET() << endl;
        }


        // various alignment methods
        if( false && ch == '1' ) {
            // ICP, on map0, map1
            assert( vec_map.size() >= 2 && "You requsted ICP computation, however it needs atleast 2 surfel maps");

            MatrixXd __0X;
            {
                int seriesI = 0;
                __0X = all_odom_poses[ seriesI ][0].inverse() * vec_map[seriesI]->surfelWorldPosition();
            } cout << "__0X : " << __0X.rows() << " x " << __0X.cols() << endl;

            MatrixXd __0Y;
            {
                int seriesI = 1;
                __0Y = all_odom_poses[ seriesI ][0].inverse() * vec_map[seriesI]->surfelWorldPosition();
            } cout << "__0Y : " << __0Y.rows() << " x " << __0Y.cols() << endl;

            std::vector<Vector3d> a_X;
            for( int i=0 ; i<__0X.cols() ; i++ )
            {
                a_X.push_back( __0X.col(i).topRows(3) );
            }

            std::vector<Vector3d> b_X;
            for( int i=0 ; i<__0Y.cols() ; i++ )
            {
                b_X.push_back( __0Y.col(i).topRows(3) );
            }

            Matrix3d ____R = Matrix3d::Identity();
            Vector3d ____t = Vector3d::Zero(); double ___s=1.0;
            theia::AlignPointCloudsUmeyama( a_X, b_X, &____R, &____t, &___s );
        }
        #endif
    } //while(ros::ok())



    // Done, just save all the meshes to disk
    cout << TermColor::GREEN() << "========Save " << vec_surf_map.size() << " surfelmaps to disk\n========\n" << TermColor::RESET();
    for( auto it = vec_surf_map.begin() ; it != vec_surf_map.end() ; it++ )
    {
        string mesh_fname = xloader.base_path+"/mesh_"+ to_string(it->first) + ".ply";
        cout << "save vec_surf_map with key=" << it->first << " to file: " << mesh_fname << endl;

        it->second->save_mesh( mesh_fname );

        // it->second->print_persurfel_info();


        MatrixXd _w_X = it->second->get_surfel_positions();
        cout << "_w_X : " << _w_X.rows() << "x" << _w_X.cols() << endl;
        MatrixXd __p = all_odom_poses[ it->first ][0].inverse() * _w_X; // 3d points in this series's 1st camera ref frame

        string goicp_fname = xloader.base_path + "/ptcld_goicp_" + to_string( it->first ) + ".txt";
        write_to_disk_for_goicp( __p, goicp_fname );
    }

}
#endif




#if 0
// Retrive data
int main()
{
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


    int idx = 23;
    json data_node = STATE["DataNodes"][idx];
    ros::Time stamp = ros::Time().fromNSec( data_node["stampNSec"] );
    Matrix4d w_T_c;
    // MatrixXd sp_cX;
    // MatrixXd sp_uv;
    cv::Mat left_image_i, right_image_i, depth_map_i;
    cv::Mat disparity_for_visualization_gray;


    bool status = xloader.retrive_data_from_json_datanode( data_node,
                    stamp, w_T_c,
                    left_image_i, right_image_i, depth_map_i, disparity_for_visualization_gray );
}

#endif
