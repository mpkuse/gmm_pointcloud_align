


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
#include "Triangulation.h"
#include "LocalBundle.h"

//
#include "utils/RosMarkerUtils.h"
#include "utils/RawFileIO.h"
#include "utils/TermColor.h"
#include "utils/ElapsedTime.h"

#include "XLoader.h"

#include "surfel_fusion/surfel_map.h"

// #include <theia/theia.h>

void write_to_disk_for_goicp( MatrixXd& __p , const string goicp_fname )
{
    cout << TermColor::iGREEN() << "===  write_to_disk_for_goicp() ===\n" << TermColor::RESET();


    #if 0 // make this to '1' to enable transform of the point set to substract mean and scale
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
    #endif


    // isnan
    int n_nan = 0;
    for( int i=0 ; i<__p.cols() ; i++ )
    {
        if( isnan( __p(0,i) ) || isnan( __p(1,i) ) || isnan( __p(2,i) ) )
            n_nan++;
    }


    ofstream myfile;
    cout << "[write_to_disk_for_goicp]Open File: " << goicp_fname << endl;
    cout << "[write_to_disk_for_goicp]n_nan=" << n_nan << endl;
    myfile.open (goicp_fname);
    myfile << __p.cols()-n_nan << endl;
    for( int i=0 ; i<__p.cols() ; i++ )
    {
        if( isnan( __p(0,i) ) || isnan( __p(1,i) ) || isnan( __p(2,i) ) )
            continue;
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


// this function will take in aX_sans_depth (the normalized image co-ordinates). The 3d points (ie. normalized image
// co-ordinates) are in same co-ordinate system, ie. appropriate rotation matrux are already multiplied to it.
//
void viz_pointfeature_matches( const ros::Publisher& marker_pub,
    MatrixXd& aX_sans_depth, VectorXd& d_a, MatrixXd& bX_sans_depth, VectorXd& d_b, VectorXd& sf,
const string ns, float red, float green, float blue, int id=0 )
{
    assert( aX_sans_depth.rows() == 3 || aX_sans_depth.rows() == 4 );
    assert( bX_sans_depth.rows() == 3 || bX_sans_depth.rows() == 4 );
    int N = aX_sans_depth.cols();
    assert( aX_sans_depth.cols() == N && d_a.size() == N && bX_sans_depth.cols() == N && d_b.size() == N && sf.size() == N);

    cout << "[viz_pointfeature_matches] ns=" << ns << "\t";
    cout << "r,g,b=" << red << "," << green << "," << blue << endl;
    cout << TermColor::iWHITE() << "---" << TermColor::RESET() << endl;
    cout << "aX_sans_depth:\n" << aX_sans_depth.leftCols(10) << endl;
    cout << "bX_sans_depth:\n" << bX_sans_depth.leftCols(10) << endl;
    cout << TermColor::CYAN() ;
    cout << "d_a:\t" << d_a.topRows(10).transpose() << endl;
    cout << "d_b:\t" << d_b.topRows(10).transpose() << endl;
    cout << "sf:\t" << sf.topRows(10).transpose() << endl;
    cout << TermColor::RESET();

    MatrixXd sa_c0_X, sb_c0_X;
    sa_c0_X = MatrixXd::Constant( 4, aX_sans_depth.cols(), 1.0 );
    sb_c0_X = MatrixXd::Constant( 4, bX_sans_depth.cols(), 1.0 );
    for( int r=0 ; r<3; r++ )
        sa_c0_X.row(r) = aX_sans_depth.row(r).cwiseProduct( d_a.transpose() );

    for( int r=0 ; r<3; r++ )
        sb_c0_X.row(r) = bX_sans_depth.row(r).cwiseProduct( d_b.transpose() );

    vector<bool> valids;
    for( int k=0 ; k<sf.size() ; k++ )
        if( sf(k) > 0.5 )
            valids.push_back(true);
        else
            valids.push_back(false);

    visualization_msgs::Marker l_mark;
    RosMarkerUtils::init_line_marker( l_mark, sa_c0_X, sb_c0_X , valids);
    l_mark.scale.x *= 0.5;
    l_mark.ns = ns ; //"pt matches"+to_string(idx_a)+"<->"+to_string(idx_b)+";nvalids="+to_string( MiscUtils::total_true( valids ) );
    l_mark.id = id;
    l_mark.color.r = red;l_mark.color.g = green;l_mark.color.b = blue;

    marker_pub.publish( l_mark );
    cout << "[viz_pointfeature_matches] FINISHED\n";

}

#if 0
void viz_pose_with_surfelmap( const ros::Publisher& marker_pub,
    MatrixXd& aX, MatrixXd& bX, Matrix4d& a_T_b )
{

    RosPublishUtils::publish_3d( marker_pub, AAA,
        "AAA (yellow)", 0,
        255,255,0, float(1.0), 1.5 );

    MatrixXd BBB = all_odom_poses[ seriesID_b ][0].inverse() * vec_surf_map[ seriesID_b ]->get_surfel_positions();
    RosPublishUtils::publish_3d( marker_pub, BBB,
        "BBB (green)", 0,
        0,255,0, float(1.0), 1.5 );

    MatrixXd YYY = a_T_b * BBB;
    RosPublishUtils::publish_3d( marker_pub, YYY,
        "a_T_b x BBB (cyan)", 0,
        0,255,255, float(1.0), 1.5 );
}
#endif


// Given the json state, and the index of 2 images , gives the image correspondences
// and the 3d points of those correspondences
//      STATE: The loaded json file (can be obtained from checkpoint of cerebro)
//      idx0, idx1 : Index0, Index1. These indices will be looked up from STATE.
//      uv_a, uv_b [Output] : 2d point. xy (ie. col, row) in image plane. 3xN
//      d_a, d_b [Output] : The depth values at the image correspondences. These will be z as seen in the camera frame of reference.
//                          it will be simply a lookup of depth image at the correspondences.
//      sf [Output]: will be 0 at bad or non existing depth values, will be 1 at good depth values
bool image_correspondences( const json& STATE, XLoader& xloader,
    int idx_a, int idx_b,
    MatrixXd& uv_a, MatrixXd& uv_b,
    VectorXd& d_a, VectorXd& d_b, VectorXd& sf )
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
    ElapsedTime elp;

    #if 1
    // -- GMS Matcher
    #define choose_gms_type 0
    // cout << TermColor::GREEN() << "=== GMS Matcher for idx_a="<< idx_a << ", idx_b=" << idx_b << TermColor::RESET() << endl;
    elp.tic();
    MatrixXd gms_uv_a, gms_uv_b;
    cout << "[image_correspondences]attempt gms_point_feature_matches\n";
    StaticPointFeatureMatching::gms_point_feature_matches( image_a, image_b, gms_uv_a, gms_uv_b );
    cout << "gms_uv_a.cols() =" << gms_uv_a.cols() << endl;
    cout << "[image_correspondences]attempt refine_and_sparsify_matches\n";
    StaticPointFeatureMatching::refine_and_sparsify_matches( image_a, image_b, gms_uv_a, gms_uv_b, uv_a, uv_b );
    cout << TermColor::BLUE() << "StaticPointFeatureMatching::gms_point_feature_matches returned in " << elp.toc_milli() << " ms\n" << TermColor::RESET();

    // StaticPointFeatureMatching::gms_point_feature_matches_scaled( image_a, image_b, uv_a, uv_b, 0.5 );
    // cout << TermColor::BLUE() << "StaticPointFeatureMatching::gms_point_feature_matches_scaled returned in " << elp.toc_milli() << " ms\n" << TermColor::RESET();

    cout << "uv_a: " << uv_a.rows() << "x" << uv_a.cols() << "\t";
    cout << "uv_b: " << uv_b.rows() << "x" << uv_b.cols() << "\t";

    if( uv_a.cols() < 50 ) {
        cout << TermColor::YELLOW() << "\nGMSMatcher produced fewer than 50 point matches, return false\n" << TermColor::RESET();
        return false;
    }

    cv::Mat dst_gmsmatcher;
    MiscUtils::plot_point_pair( image_a, uv_a, idx_a,
                                image_b, uv_b, idx_b, dst_gmsmatcher,
                                //3, "gms plot (resize 0.5)" );
                                cv::Scalar( 0,0,255 ), cv::Scalar( 0,255,0 ), false, "gms plot (resize 0.5)" );

    cv::resize(dst_gmsmatcher, dst_gmsmatcher, cv::Size(), 0.5, 0.5);
    cv::imshow( "GMSMatcher", dst_gmsmatcher );
    #endif


    #if 0
    // -- Simple ORB Matcher
    cout << TermColor::GREEN() << "=== Point feature matcher (ORB) for idx_a="<< idx_a << ", idx_b=" << idx_b << TermColor::RESET() << endl;

    elp.tic();
    PointFeatureMatchingSummary summary;
    StaticPointFeatureMatching::point_feature_matches( image_a, image_b, uv_a, uv_b, summary );
    cout << TermColor::BLUE() << "StaticPointFeatureMatching::point_feature_matches returned in " << elp.toc_milli() << " ms\n" << TermColor::RESET();


    cout << "uv_a: " << uv_a.rows() << "x" << uv_a.cols() << "\t";
    cout << "uv_b: " << uv_b.rows() << "x" << uv_b.cols() << "\t";

    if( uv_a.cols() < 5 ) {
        cout << TermColor::YELLOW() << "\npoint_feature_matches() produced fewer than 5 point matches, return false\n" << TermColor::RESET();
        return false;
    }

    cv::Mat dst_matcher;
    MiscUtils::plot_point_pair( image_a, uv_a, idx_a,
                                image_b, uv_b, idx_b, dst_matcher,
                                // 3, "gms plot (resize 0.5)"
                                cv::Scalar( 0,0,255 ), cv::Scalar( 0,255,0 ), false, "gms plot (resize 0.5)"
                            );
    cv::resize(dst_matcher, dst_matcher, cv::Size(), 0.5, 0.5);
    cv::imshow( "GMSMatcher", dst_matcher );
    #endif

    // --- Depth lookup at correspondences
    #if 1

    float near = 0.5;
    float far = 4.5;
    d_a = VectorXd::Zero( uv_a.cols() );
    d_b = VectorXd::Zero( uv_b.cols() );
    assert( uv_a.cols() == uv_b.cols() && uv_a.cols() > 0 );
    sf = VectorXd::Zero( uv_a.cols() );
    for( int i=0 ; i<uv_a.cols() ; i++ )
    {
        float depth_val_a;
        {
            if( depth_a.type() == CV_16UC1 ) {
                depth_val_a = .001 * depth_a.at<uint16_t>( uv_a(1,i), uv_a(0,i) );
            }
            else if( depth_a.type() == CV_32FC1 ) {
                // just assuming the depth values are in meters when CV_32FC1
                depth_val_a = depth_a.at<float>( uv_a(1,i), uv_a(0,i) );
            }
            else {
                assert( false );
                cout << "[image_correspondences_xgfru]depth type is neighter of CV_16UC1 or CV_32FC1\n";
                exit(1);
            }
        }

        float depth_val_b;
        {
            if( depth_b.type() == CV_16UC1 ) {
                depth_val_b = .001 * depth_b.at<uint16_t>( uv_b(1,i), uv_b(0,i) );
            }
            else if( depth_b.type() == CV_32FC1 ) {
                // just assuming the depth values are in meters when CV_32FC1
                depth_val_b = depth_b.at<float>( uv_b(1,i), uv_b(0,i) );
            }
            else {
                assert( false );
                cout << "[image_correspondences_xgfru]depth type is neighter of CV_16UC1 or CV_32FC1\n";
                exit(1);
            }
        }

        d_a( i ) = (double) depth_val_a;
        d_b( i ) = (double) depth_val_b;
        if( depth_val_a > near && depth_val_a < far
            &&
            depth_val_b > near && depth_val_b < far )
        {
            sf( i ) = 1.0;
        } else
            sf(i) = 0.0;
    }
    #endif

    int nvalids = (int) sf.sum();
    cout << TermColor::YELLOW() << "Of the total " << uv_a.cols() << " point-matches, only " << nvalids << " had good depths\n" << TermColor::RESET();
    return true;
}

// Given the json state, and the index of 2 images , gives the image correspondences
// and the 3d points of those correspondences
//      STATE: The loaded json file (can be obtained from checkpoint of cerebro)
//      idx0, idx1 : Index0, Index1. These indices will be looked up from STATE.
//      uv_a, uv_b [Output] : 2d point. xy (ie. col, row) in image plane. 3xN
//      aX, bX     [Output] : 3d points of uv_a, uv_b (respectively) expressed in co-ordinates of the camera. 4xN
//      valids     [output] : an array of size N, which denotes the validity of each 3d points. The points were no depth or where out-of-range depth this will be false
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
    ElapsedTime elp;

    #if 1
    // -- GMS Matcher
    // cout << TermColor::GREEN() << "=== GMS Matcher for idx_a="<< idx_a << ", idx_b=" << idx_b << TermColor::RESET() << endl;
    elp.tic();
    StaticPointFeatureMatching::gms_point_feature_matches( image_a, image_b, uv_a, uv_b );
    cout << TermColor::BLUE() << "StaticPointFeatureMatching::gms_point_feature_matches returned in " << elp.toc_milli() << " ms\n" << TermColor::RESET();

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
    #endif


    #if 0
    // -- Simple ORB Matcher
    cout << TermColor::GREEN() << "[image_correspondences] Simple ORB Matcher for idx_a="<< idx_a << ", idx_b=" << idx_b << TermColor::RESET() << endl;

    elp.tic();
    PointFeatureMatchingSummary summary;
    StaticPointFeatureMatching::point_feature_matches( image_a, image_b, uv_a, uv_b, summary );
    summary.prettyPrint(1);
    cout << TermColor::BLUE() << "[image_correspondences] StaticPointFeatureMatching::point_feature_matches returned in " << elp.toc_milli() << " ms\n" << TermColor::RESET();


    cout << "uv_a: " << uv_a.rows() << "x" << uv_a.cols() << "\t";
    cout << "uv_b: " << uv_b.rows() << "x" << uv_b.cols() << "\t";

    if( uv_a.cols() < 5 ) {
        cout << TermColor::YELLOW() << "\npoint_feature_matches() produced fewer than 5 point matches, return false\n" << TermColor::RESET();
        return false;
    }

    cv::Mat dst_matcher;
    MiscUtils::plot_point_pair( image_a, uv_a, idx_a,
                                image_b, uv_b, idx_b, dst_matcher,
                                // 3, "gms plot (resize 0.5)"
                                cv::Scalar( 0,0,255 ), cv::Scalar( 0,255,0 ), false, "gms plot (resize 0.5)"
                            );
    cv::resize(dst_matcher, dst_matcher, cv::Size(), 0.5, 0.5);
    cv::imshow( "GMSMatcher", dst_matcher );
    #endif


    #if 1
    // -- 3D Points from depth image at the correspondences
    // vector<bool> valids;
    valids.clear();
    StaticPointFeatureMatching::make_3d_3d_collection__using__pfmatches_and_depthimage(
        xloader.left_camera,
        uv_a, depth_a, uv_b, depth_b,
        aX, bX, valids
    );

    int nvalids = 0;
    for( int i=0 ; i<valids.size() ; i++ )
        if( valids[i]  == true )
            nvalids++;
    cout << TermColor::YELLOW() <<  "nvalids_depths=" << nvalids << " of total=" << valids.size() << TermColor::RESET() << "\t";

    cout << "aX: " << aX.rows() << "x" << aX.cols() << "\t";
    cout << "bX: " << bX.rows() << "x" << bX.cols() << endl;

    if( nvalids < 5 ) {
        cout << TermColor::YELLOW() << "Of the total " << valids.size() << " point-matches, only " << nvalids << " had good depths, this is less than the threshold, so return false\n" << TermColor::RESET();
        return false;
    }
    #endif


    // -- 3D points lookup from surfels.
    //      Have the surfel maps of both sequence, given the tuple (frameID, u,v) get the surfelID
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
    msg += "1: Draw random image pair, map point feature to 3dpoints;";
    msg += "2: align3d3d with depth refinement;";
    msg += "3: monocular;";
    msg += "4: manual pose input;";
    msg += "9:viz 3d model normals;";
    msg += "ESC: quit;";
    MiscUtils::append_status_image( status, msg, .45, cv::Scalar(0,0,0), cv::Scalar(255,255,255) );

}


#if 1
int main( int argc, char ** argv )
{

    srand( 0 ); //fixed seed
    // srand (time(NULL));

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
    #if 0
    idx_ptr.push_back(0);
    // idx_ptr.push_back(2885);
    #else
    if( argc > 1 )
    {
        cout << "argc = " << argc << endl;
        for( int p=1 ; p<argc ; p++ ) {
            cout << p << " : " << argv[p] << endl;
            idx_ptr.push_back( std::stoi( argv[p] ) );
        }
        // exit(1);
    } else {
        idx_ptr.push_back(0);
    }
    #endif

    // these maps are indexed by the sequenceID
    map<  int,  vector<int>         > all_i; //other vector to hold multiple sequences
    map<  int,  vector<Matrix4d>    > all_odom_poses;

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

            #if 1
            // show resized
            cv::Mat left_image_resize;
            cv::resize(left_image, left_image_resize, cv::Size(), 0.5, 0.5);
            cv::imshow( win_name.c_str(), left_image_resize );
            #else
            // show original
            cv::imshow( win_name.c_str(), left_image );
            #endif
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

            //  vec_surf_map[ seriesI ]->print_persurfel_info( 1 );

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


        if( ch == '1' ) // Draw random image pair, map point feature to 3dpoints; 3d3dalign
        {
            cout << TermColor::BLUE() << "1 pressed PROCESS" << TermColor::RESET() << endl;
            assert( all_i.size() >= 2 );
            cout << "series#0: " << *(all_i[0].begin()) << " --> " << *(all_i[0].rbegin()) << endl;
            cout << "series#1: " << *(all_i[1].begin()) << " --> " << *(all_i[1].rbegin()) << endl;


            vector< MatrixXd > all_sa_c0_X, all_sb_c0_X;
            vector< vector<bool> > all_valids;
            // for( int k=0 ; k< 5 ; k++ )
            int rand_itr = 0;

            std::map< std::pair<int,int>, bool > repeatl;
            while( true )
            {
                cout << "\n-----------------------\n";
                cout << "--- rand_itr=" << rand_itr ;
                cout << "\n-----------------------\n\n";
                rand_itr++;


                int _a = random_in_range( 0, (int)all_i[0].size() );
                int _b = random_in_range( 0, (int)all_i[1].size() );

                // is repeat? then skip
                if( repeatl.count( std::make_pair(_a,_b) ) > 0 ) {
                    cout << "I have a;lready seen _a=" << _a << ", _b=" << _b << "before...skip\n";
                    continue;
                }
                repeatl[  std::make_pair(_a,_b)  ] = true;

                int idx_a = all_i[0][_a];
                int idx_b = all_i[1][_b];


                // --- Image correspondences and 3d points from depth images
                MatrixXd uv_a, uv_b, aX, bX;
                vector<bool> valids; //< which point correspondences have good/valid depth values
                bool status = image_correspondences( STATE, xloader, idx_a, idx_b, uv_a, uv_b, aX, bX, valids ); //there is an imshow in this

                if( status == false ) {
                    cout << TermColor::RED() << "image_correspondences returned false, skip this sample\n" << TermColor::RESET();
                    continue;
                }


                // --- Change co-ordinate system of aX and bX
                if( aX.cols() > 0 && bX.cols() > 0 ) {
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
                }
                else
                    cout << TermColor::YELLOW() << "WARN no 3d points, so no publish\n" << TermColor::RESET();

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

                    #if 0
                    RawFileIO::write_EigenMatrix( "/app/catkin_ws/src/gmm_pointcloud_align/resources/pointsets/"+to_string(idx_a)+"-"+to_string(idx_b)+"__dst0.txt", dst0 );
                    RawFileIO::write_EigenMatrix( "/app/catkin_ws/src/gmm_pointcloud_align/resources/pointsets/"+to_string(idx_a)+"-"+to_string(idx_b)+"__dst1.txt", dst1 );
                    #endif

                    #if 0
                    // pose computation, closed form only - simple
                    Matrix4d a_T_b = Matrix4d::Identity();
                    PoseComputation::closedFormSVD( dst0, dst1, a_T_b );
                    #endif

                    #if 0
                    // Pose Computation, alternate with closed form and iterative switch-constraint refinement.
                    Matrix4d a_T_b = Matrix4d::Identity();
                    PoseComputation::closedFormSVD( dst0, dst1, a_T_b );
                    cout << "[main] after closedFormSVD a_T_b = " << PoseManipUtils::prettyprintMatrix4d( a_T_b ) << endl;

                    VectorXd switch_weights;
                    for( int yuy=0; yuy<1 ; yuy++ ) {

                    cout << TermColor::iWHITE() << "---- yuy=" << yuy << "----" << TermColor::RESET() << endl;
                    PoseComputation::refine( dst0, dst1, a_T_b, switch_weights );
                    cout << "[main] after refinement    a_T_b = " << PoseManipUtils::prettyprintMatrix4d( a_T_b ) << endl;

                    cout << TermColor::YELLOW() << "[main]Once again do closedFormSVD with weights (len=)" << switch_weights.size() << "\n" << TermColor::RESET();
                    cout << "before weighted closed form : " << PoseManipUtils::prettyprintMatrix4d(a_T_b) << endl;
                    PoseComputation::closedFormSVD( dst0, dst1, switch_weights, a_T_b );
                    cout << "after  weighted closed form : " << PoseManipUtils::prettyprintMatrix4d(a_T_b) << endl;

                    }
                    PoseComputation::testTransform( dst0, dst1, a_T_b );
                    #endif


                    #if 1
                    // pose computation,
                    Matrix4d a_T_b = Matrix4d::Identity();
                    // PoseComputation::closedFormSVD( dst0, dst1, a_T_b );
                    VectorXd switch_weights = VectorXd::Constant( dst0.cols() , 1.0 );

                    ElapsedTime al_tp;
                    PoseComputation::alternatingMinimization( dst0, dst1, a_T_b, switch_weights );
                    cout << TermColor::BLUE() << "[main] PoseComputation::alternatingMinimization took ms=" << al_tp.toc_milli() << endl;

                    #endif

                    MatrixXd AAA = all_odom_poses[ 0 ][0].inverse() * vec_surf_map[ 0 ]->get_surfel_positions();
                    RosPublishUtils::publish_3d( marker_pub, AAA,
                        "AAA (red)", 0,
                        255,0,0, float(1.0), 1.5 );

                    MatrixXd BBB = all_odom_poses[ 1 ][0].inverse() * vec_surf_map[ 1 ]->get_surfel_positions();
                    RosPublishUtils::publish_3d( marker_pub, BBB,
                        "BBB (green)", 0,
                        0,255,0, float(1.0), 1.5 );


                    MatrixXd YYY = a_T_b * BBB;
                    RosPublishUtils::publish_3d( marker_pub, YYY,
                        "a_T_b x BBB (cyan)", 0,
                        0,255,255, float(1.0), 1.5 );


                    cout << "\n pose computation done, press any key to continue drawing more pair of images\n";
                    cv::waitKey(0);
                }
            }
            cv::destroyWindow("GMSMatcher");

        }

        if( ch == '2' ) // align3d3d with depth refinement
        {
            cout << TermColor::BLUE() << "2 pressed PROCESS" << TermColor::RESET() << endl;
            int seriesID_a = 0; //< the series number of the 2 sequences in question
            int seriesID_b = 1;
            assert( all_i.size() >= 2 );
            cout << "series#0: " << *(all_i[seriesID_a].begin()) << " --> " << *(all_i[seriesID_a].rbegin()) << endl;
            cout << "series#1: " << *(all_i[seriesID_b].begin()) << " --> " << *(all_i[seriesID_b].rbegin()) << endl;


            // draw random image pairs
            vector<VectorXd> all_d_a;
            vector<VectorXd> all_d_b;
            vector<VectorXd> all_sf;
            vector<MatrixXd> all_aX_sans_depth;
            vector<MatrixXd> all_bX_sans_depth;


            for( int rand_itr = 0 ; ; rand_itr++ )
            {
                cout << "\n-----------------------\n";
                cout << "--- rand_itr=" << rand_itr ;
                cout << "\n-----------------------\n\n";


                int _a = random_in_range( 0, (int)all_i[seriesID_a].size() ); //< an image in the series a
                int _b = random_in_range( 0, (int)all_i[seriesID_b].size() ); //< an image in the series a

                int idx_a = all_i[seriesID_a][_a]; //<< global image index
                int idx_b = all_i[seriesID_b][_b];

                // --- Image Correspondences
                MatrixXd uv_a, uv_b;
                VectorXd d_a, d_b, sf; //depth values at correspondences, same N as uv_a
                bool status = image_correspondences( STATE, xloader, idx_a, idx_b, uv_a, uv_b, d_a, d_b, sf ); //there is an imshow in this
                if( status == false ) {
                    cout << TermColor::RED() << "image_correspondences returned false, skip this sample\n" << TermColor::RESET();
                    continue;
                }


                // --- Change co-ordinates
                //      express the normalized co-ordinates in world ref frame, so as to allow for the accumulation
                Matrix4d wTa, wTb;
                wTa = all_odom_poses[seriesID_a][_a];
                wTb = all_odom_poses[seriesID_b][_b];



                // MatrixXd uv_a_normalized = MatrixXd::Zero( 3, uv_a.cols() );
                // MatrixXd uv_b_normalized = MatrixXd::Zero( 3, uv_b.cols() );
                MatrixXd uv_a_normalized = MatrixXd::Constant( 4, uv_a.cols(), 1.0 );
                MatrixXd uv_b_normalized = MatrixXd::Constant( 4, uv_b.cols(), 1.0 );
                // uv_a_normalized = K.inverse() * uv_a ; uv_b_normalized := K.inverse() * uv_b ;
                for( int k=0 ; k<uv_a.cols() ; k++ )
                {
                    Vector3d _0P;
                    xloader.left_camera->liftProjective( uv_a.col(k).topRows(2), _0P  );
                    uv_a_normalized.col(k).topRows(3) = _0P;
                    uv_a_normalized(3,k) = (abs(d_a(k)) > 1e-6) ? (1.0/d_a(k)) : -1.0 ; //this is needed when you want to transform the 3d points with camera poses and multiply the depth at the end.


                    Vector3d _1P;
                    xloader.left_camera->liftProjective( uv_b.col(k).topRows(2), _1P  );
                    uv_b_normalized.col(k).topRows(3) = _1P;
                    uv_b_normalized(3,k) = ( d_b(k) > 1e-6 )?( 1.0/d_b(k) ): -1.0;
                }

                MatrixXd aX_sans_depth = all_odom_poses[seriesID_a][0].inverse() * wTa * uv_a_normalized;
                MatrixXd bX_sans_depth = all_odom_poses[seriesID_b][0].inverse() * wTb * uv_b_normalized;


                all_d_a.push_back( d_a );
                all_d_b.push_back( d_b );
                all_sf.push_back( sf );
                all_aX_sans_depth.push_back( aX_sans_depth );
                all_bX_sans_depth.push_back( bX_sans_depth );




                // --- Wait Key
                cout << "press `c` for custom compute, press `p` for pose computation, press `b` to break, press any other key to keep drawing more pairs\n";
                char ch = cv::waitKey(0) & 0xEFFFFF;
                if( ch == 'b' )
                    break;

                    #if 0
                if( ch == 'p' )
                {   // print and publish
                    if( aX_sans_depth.cols() > 10 ) {
                    #if 1
                    cout << TermColor::iWHITE() << "---" << TermColor::RESET() << endl;
                    cout << "aX_sans_depth:\n" << aX_sans_depth.leftCols(10) << endl;
                    cout << "bX_sans_depth:\n" << bX_sans_depth.leftCols(10) << endl;
                    cout << TermColor::CYAN() ;
                    cout << "d_a:\t" << d_a.topRows(10).transpose() << endl;
                    cout << "d_b:\t" << d_b.topRows(10).transpose() << endl;
                    cout << "sf:\t" << sf.topRows(10).transpose() << endl;
                    cout << TermColor::RESET();

                    MatrixXd sa_c0_X, sb_c0_X;
                    sa_c0_X = MatrixXd::Constant( 4, aX_sans_depth.cols(), 1.0 );
                    sb_c0_X = MatrixXd::Constant( 4, bX_sans_depth.cols(), 1.0 );

                    for( int r=0 ; r<4; r++ )
                        sa_c0_X.row(r) = aX_sans_depth.row(r).cwiseProduct( d_a.transpose() );

                    for( int r=0 ; r<4; r++ )
                        sb_c0_X.row(r) = bX_sans_depth.row(r).cwiseProduct( d_b.transpose() );

                    cout << "---\n";
                    cout << "sa_c0_X:\n" << sa_c0_X.leftCols(10) << endl;
                    cout << "sb_c0_X:\n" << sb_c0_X.leftCols(10) << endl;
                    cout << TermColor::iWHITE() << "---" << TermColor::RESET() << endl;


                    #if 1
                    // publish lines (of correspondences)
                    vector<bool> valids;
                    for( int k=0 ; k<sf.size() ; k++ )
                        if( sf(k) > 0.9999 )
                            valids.push_back(true);
                        else
                            valids.push_back(false);

                    visualization_msgs::Marker l_mark;
                    RosMarkerUtils::init_line_marker( l_mark, sa_c0_X, sb_c0_X , valids);
                    l_mark.scale.x *= 0.5;
                    l_mark.ns = "pt matches"+to_string(idx_a)+"<->"+to_string(idx_b)+";nvalids="+to_string( MiscUtils::total_true( valids ) );
                    marker_pub.publish( l_mark );
                    #endif

                    #endif
                    } else cout << "WARN, i didnt publish because less than 10 point-features\n";
                }
                    #endif

                if( ch == 'p' )
                {
                    cout << TermColor::GREEN() << "---p pressed, process 3dpts_sans_depth and depths, alternately optimize the pose and correct the depths\n" << TermColor::RESET();

                    // gather input :
                    //      all_d_a, all_d_b, all_sf, all_aX_sans_depth, all_bX_sans_depth
                    VectorXd dst_d_a, dst_d_b, dst_sf;
                    MatrixXd dst_aX_sans_depth, dst_bX_sans_depth;
                    MiscUtils::gather( all_d_a, dst_d_a );
                    MiscUtils::gather( all_d_b, dst_d_b );
                    MiscUtils::gather( all_sf, dst_sf );
                    MiscUtils::gather( all_aX_sans_depth, dst_aX_sans_depth );
                    MiscUtils::gather( all_bX_sans_depth, dst_bX_sans_depth );

                    #if 1
                    RawFileIO::write_EigenMatrix( "/app/catkin_ws/src/gmm_pointcloud_align/resources/pointsets/dst_d_a.txt", dst_d_a );
                    RawFileIO::write_EigenMatrix( "/app/catkin_ws/src/gmm_pointcloud_align/resources/pointsets/dst_d_b.txt", dst_d_b );
                    RawFileIO::write_EigenMatrix( "/app/catkin_ws/src/gmm_pointcloud_align/resources/pointsets/dst_sf.txt", dst_sf );
                    RawFileIO::write_EigenMatrix( "/app/catkin_ws/src/gmm_pointcloud_align/resources/pointsets/dst_aX_sans_depth.txt", dst_aX_sans_depth );
                    RawFileIO::write_EigenMatrix( "/app/catkin_ws/src/gmm_pointcloud_align/resources/pointsets/dst_bX_sans_depth.txt", dst_bX_sans_depth );
                    #endif


                    cout << "gather: all_* --> dst_*\n";
                    cout << "dst_d_a.size=" << dst_d_a.size() << "\t";
                    cout << "dst_d_b.size=" << dst_d_b.size() << "\t";
                    cout << "dst_sf.size=" << dst_sf.size() << "\t";
                    cout << "dst_aX_sans_depth.rows.cols=" << dst_aX_sans_depth.rows() << "x" << dst_aX_sans_depth.cols() << "\t";
                    cout << "dst_bX_sans_depth.rows.cols=" << dst_bX_sans_depth.rows() << "x" << dst_bX_sans_depth.cols() << "\t";
                    cout << endl;
                    for( int g=0 ; g<all_sf.size() ; g++ )
                        cout << "\t\t[" << g << "] " << all_sf[g].size() << endl;

                    // optimize pose (aTb) while keeping depths as constant
                    Matrix4d a_T_b;
                    PoseComputation::closedFormSVD( dst_aX_sans_depth, dst_d_a, dst_bX_sans_depth, dst_d_b,  dst_sf, a_T_b  );
                    cout << "[main] closed form (with weights) a_T_b=" << PoseManipUtils::prettyprintMatrix4d( a_T_b ) << endl;


                    viz_pointfeature_matches( marker_pub,
                        dst_aX_sans_depth, dst_d_a, dst_bX_sans_depth, dst_d_b,  dst_sf,
                        "pf-matches before opt(pink)", 199./255., 21./255., 133./255.
                    );


                    // optimize depths while keeping pose (aTb) as constant
                    PoseComputation::refine( dst_aX_sans_depth, dst_d_a, dst_bX_sans_depth, dst_d_b,  dst_sf, a_T_b, false, true, true  );
                    PoseComputation::refine( dst_aX_sans_depth, dst_d_a, dst_bX_sans_depth, dst_d_b,  dst_sf, a_T_b, true,  false, true  );

                    viz_pointfeature_matches( marker_pub,
                        dst_aX_sans_depth, dst_d_a, dst_bX_sans_depth, dst_d_b,  dst_sf,
                        "pf-matches after opt(orange)", 1.0, 0.5, 0.0
                    );

                    // viz the results
                    {
                                        MatrixXd AAA = all_odom_poses[ seriesID_a ][0].inverse() * vec_surf_map[ seriesID_a ]->get_surfel_positions();
                                        RosPublishUtils::publish_3d( marker_pub, AAA,
                                            "AAA (yellow)", 0,
                                            255,255,0, float(1.0), 1.5 );

                                        MatrixXd BBB = all_odom_poses[ seriesID_b ][0].inverse() * vec_surf_map[ seriesID_b ]->get_surfel_positions();
                                        RosPublishUtils::publish_3d( marker_pub, BBB,
                                            "BBB (green)", 0,
                                            0,255,0, float(1.0), 1.5 );


                                        MatrixXd YYY = a_T_b * BBB;
                                        RosPublishUtils::publish_3d( marker_pub, YYY,
                                            "a_T_b x BBB (cyan)", 0,
                                            0,255,255, float(1.0), 1.5 );
                    }


                    cout << TermColor::GREEN() << "--- pose computation done...\n" << TermColor::RESET();
                }



            }  //for( rand_itr = 0 ; ; rand_itr++ )
            cv::destroyWindow("GMSMatcher"); //close the imshow from image_correspondences()


        }


        if( ch == '3' ) //monocular
        {

            cout << TermColor::BLUE() << "3 pressed PROCESS" << TermColor::RESET() << endl;
            assert( all_i.size() >= 2 );
            cout << "series#0: " << *(all_i[0].begin()) << " --> " << *(all_i[0].rbegin()) << endl;
            cout << "series#1: " << *(all_i[1].begin()) << " --> " << *(all_i[1].rbegin()) << endl;

            // odom pose of 1st image in series
            Matrix4d w_T_a0, w_T_b0;
            bool status_posea0 = xloader.retrive_pose_from_json_datanode( STATE["DataNodes"][*(all_i[0].begin())], w_T_a0 );
            bool status_poseb0 = xloader.retrive_pose_from_json_datanode( STATE["DataNodes"][*(all_i[1].begin())], w_T_b0 );
            assert( status_posea0 && status_poseb0 );


            // odom for the full seq
            cout << "\nGet Odoms for seq-a:\n";
            vector< Matrix4d > odom__a0_T_a;
            vector< int > odom_a_idx;
            vector< cv::Mat > images_seq_a, depthmap_seq_a;
            for( auto it=all_i[0].begin() ; it!=all_i[0].end() ; it++ )
            {
                cout << *it << "\t";
                Matrix4d w_T_a;
                bool status__ = xloader.retrive_pose_from_json_datanode( STATE["DataNodes"][*it], w_T_a );
                assert( status__ );

                Matrix4d a0_T_a = w_T_a0.inverse() * w_T_a;
                odom__a0_T_a.push_back( a0_T_a );
                odom_a_idx.push_back( *it );


                #if 0 //image only
                cv::Mat __im__;
                bool status_im = xloader.retrive_image_data_from_json_datanode( STATE["DataNodes"][*it], __im__ );
                assert( status_im );
                images_seq_a.push_back( __im__ );
                #endif


                #if 1 //image and depth
                cv::Mat __im__, __depth__;
                bool status_im = xloader.retrive_image_data_from_json_datanode( STATE["DataNodes"][*it], __im__, __depth__ );
                assert( status_im );
                images_seq_a.push_back( __im__ );
                depthmap_seq_a.push_back( __depth__ );
                #endif
            }
            cout << endl;
            cout << "Get Odoms for seq-b:\n";
            vector< Matrix4d > odom__b0_T_b;
            vector< int > odom_b_idx;
            vector< cv::Mat > images_seq_b, depthmap_seq_b;
            for( auto it=all_i[1].begin() ; it!=all_i[1].end() ; it++ )
            {
                cout << *it << "\t";
                Matrix4d w_T_b;
                bool status__ = xloader.retrive_pose_from_json_datanode( STATE["DataNodes"][*it], w_T_b );
                assert( status__ );

                Matrix4d b0_T_b = w_T_b0.inverse() * w_T_b;
                odom__b0_T_b.push_back( b0_T_b );
                odom_b_idx.push_back( *it );

                #if 0
                cv::Mat __im__;
                bool status_im = xloader.retrive_image_data_from_json_datanode( STATE["DataNodes"][*it], __im__ );
                assert( status_im );
                images_seq_b.push_back( __im__ );
                #endif


                #if 1 //image and depth
                cv::Mat __im__, __depth__;
                bool status_im = xloader.retrive_image_data_from_json_datanode( STATE["DataNodes"][*it], __im__, __depth__ );
                assert( status_im );
                images_seq_b.push_back( __im__ );
                depthmap_seq_b.push_back( __depth__ );
                #endif
            }
            cout << endl;



            vector<Matrix4d> all_a0_T_a, all_b0_T_b;
            vector<MatrixXd> all_normed_uv_a, all_normed_uv_b; //each matrix will be 3xN (expressed as homogeneous cords)
            vector<VectorXd> all_d_a, all_d_b, all_sf;
            vector< std::pair<int,int> > all_pair_idx; // the idx of each pairs

            vector<MatrixXd> all_a0X, all_b0X; //each pt will be 4xN
            vector<vector<bool>> all_valids;

            // random pairs, image correspondences only
            std::map< std::pair<int,int>, bool > repeatl;
            for( int rand_itr=0 ; rand_itr<10 ; rand_itr++ )
            {
                cout << "\n-----------------------\n";
                cout << "--- rand_itr=" << rand_itr ;
                cout << "\n-----------------------\n\n";

                //-----
                //--- Pick a random sample for each seq
                //-----
                int seriesID_a = 0; //< the series number of the 2 sequences in question
                int seriesID_b = 1;

                int _a = random_in_range( 0, (int)all_i[seriesID_a].size() );
                int _b = random_in_range( 0, (int)all_i[seriesID_b].size() );
                // is repeat? then skip
                if( repeatl.count( std::make_pair(_a,_b) ) > 0 ) {
                    cout << "I have a;lready seen _a=" << _a << ", _b=" << _b << "before...skip\n";
                    continue;
                }
                repeatl[  std::make_pair(_a,_b)  ] = true;

                int idx_a = all_i[seriesID_a][_a];
                int idx_b = all_i[seriesID_b][_b];
                cout << "I pick the pair with global idx_a="<< idx_a << "  idx_b=" << idx_b << endl;



                //----
                //--- Image Correspondences
                //----
                MatrixXd uv_a, uv_b;
                VectorXd d_a, d_b, sf;
                bool im_corres_status = image_correspondences( STATE, xloader, idx_a, idx_b, uv_a, uv_b, d_a, d_b, sf );

                int n_matches = uv_a.cols();
                if( n_matches <10 ) { //for gmsmatcher keep a 10x threshold
                // if( n_matches < 5 ) {
                    cout << "n_matches=" << n_matches << " these are too few..ignore\n";
                    continue;
                }

                if( im_corres_status == false ) {
                    cout << "im_corres_status is false....continue\n";
                    continue;
                }

                vector<Point2f> pts_to_track_a;
                MiscUtils::eigen_2_point2f( uv_a, pts_to_track_a );

                vector<Point2f> pts_to_track_b;
                MiscUtils::eigen_2_point2f( uv_b, pts_to_track_b );


                MatrixXd normed_uv_a, normed_uv_b; //the normalized image co-ordinates for the original image correspondences
                StaticPointFeatureMatching::image_coordinates_to_normalized_image_coordinates( xloader.left_camera, uv_a, normed_uv_a );
                StaticPointFeatureMatching::image_coordinates_to_normalized_image_coordinates( xloader.left_camera, uv_b, normed_uv_b );

                #if 1
                cout << "press 'b' to exit the loop of random pair draws, any other key to keep drawing more random pairs\n";
                ch = cv::waitKey(0);
                if( ch == 'b' ) {
                    cout << "b pressed, goto end_of_random_draws\n";
                    goto end_of_random_draws;
                }
                #endif


                // make this to 1 to visulize optical flow
                #define _VIZ_ 0

                //----
                //--- Depth Estimation from Optical Flow and Triangulation. Will also use odometry poses
                //----
                //--Seq-A , A+1, A+2,...
                #define __SEQ_A_MONOCULAR__ 0

                #if 0
                VectorXd monocular_d_a = VectorXd::Zero( uv_a.cols() );
                {
                    //---- the `base`
                    cout << "\n\n======= Look at adjacent images of idx_a=" << idx_a << endl;
                    json data_node_a = STATE["DataNodes"][idx_a];
                    cv::Mat image_a;
                    Matrix4d w_T_al;
                    bool status_img  = xloader.retrive_image_data_from_json_datanode( data_node_a, image_a );
                    bool status_pose = xloader.retrive_pose_from_json_datanode( data_node_a, w_T_al );
                    assert( status_img && status_pose );

                    #if _VIZ_ //viz
                    // plot base image with points to track
                    cv::Mat dst_baseimg_with_feat_to_track;
                    MiscUtils::plot_point_sets(  image_a, uv_a, dst_baseimg_with_feat_to_track,
                        cv::Scalar(255,0,0), true, "base image (idx=" + to_string(idx_a)+ ") to track from n_pts="+to_string(uv_a.cols()) );
                    MiscUtils::imshow( "dst_baseimg_with_feat_to_track", dst_baseimg_with_feat_to_track, 1.0 );
                    #endif


                    //---- next image in seq   ==>  track the points uv_a on idx_a+p
                    // we track the correspondences at say 2 frames ahead and 2 frames back to get an estimate of depth at those points.

                    vector<int> list__p; // the p that we used
                    vector<MatrixXd> tracked_uv_at__p; //the uv (image co-ordinates) points at p
                    vector<MatrixXd> tracked_normed_uv_at__p; //the uv (normalized image co-ordinates) points at p
                    vector<Matrix4d> p_T_base_at__p; //the pose of base wrt camera-p. for every p
                    // vector< vector<uchar> > status_at__p; // status of the tracked points at p.
                    vector<VectorXd> status_at__p; // status of the tracked points at p.
                    for( int  p=-6 ; p<8 ; p++ )
                    {
                        cout <<  "\tp=" << p << endl;
                        json data_node_a__p = STATE["DataNodes"][idx_a+p];
                        if( xloader.is_data_available(data_node_a__p) == false ) {
                            cout << "\tno image or pose data so skip this p\n";
                            continue;
                        }

                        if( p==0 ) {
                            cout << "\tskip p=0\n";
                            continue;
                        }

                        cv::Mat image_a__p;
                        Matrix4d w_T_al__p;
                        bool status_img__p  = xloader.retrive_image_data_from_json_datanode( data_node_a__p, image_a__p );
                        bool status_pose__p = xloader.retrive_pose_from_json_datanode( data_node_a__p, w_T_al__p );
                        assert( status_img__p && status_pose__p );


                        Matrix4d al_T_al__p = w_T_al.inverse() * w_T_al__p;
                        cout << "\tbaseline = " << al_T_al__p.col(3).topRows(3).norm() << endl;


                        //---- optical flow
                        vector<Point2f> result_of_opticalflow = pts_to_track_a; //give it an initial guess
                        vector<uchar> status;
                        vector<float> err;
                        cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, 0.01);
                        cv::Size win_size = cv::Size(25,25);
                        int n_pyramids = 4;

                        cout << TermColor::GREEN() << "\tCalculate Optical Flow" << TermColor::RESET() << endl;
                        cout << "\timage_a:" << MiscUtils::cvmat_info( image_a ) << endl;
                        cout << "\timage_a__p:" << MiscUtils::cvmat_info( image_a__p ) << endl;
                        cout << "\tpts_to_track_a.size=" << pts_to_track_a.size() << endl;
                        ElapsedTime t_opticalflow( "optical_flow");
                        cv::calcOpticalFlowPyrLK( image_a, image_a__p, pts_to_track_a, result_of_opticalflow,
                                 status, err, win_size, n_pyramids, criteria );
                        cout << TermColor::BLUE() << t_opticalflow.toc() << TermColor::RESET() << endl;
                                 int nfail = (int)status.size() - MiscUtils::total_positives(status);
                                 cout << "\t\tnfail=" << nfail << endl;

                        cout << "\tshowing adjacent image to idx_a="<< idx_a << "@p=" << p << " pose=" << PoseManipUtils::prettyprintMatrix4d(w_T_al__p) << endl;
                        MatrixXd eigen_result_of_opticalflow;
                        MiscUtils::point2f_2_eigen( result_of_opticalflow, eigen_result_of_opticalflow, true);


                        //---- save for later
                        list__p.push_back(p);
                        tracked_uv_at__p.push_back( eigen_result_of_opticalflow );
                        Matrix4d ___tmp =  w_T_al__p.inverse() * w_T_al ;
                        // Matrix4d ___tmp =  w_T_al__p * w_T_al.inverse() ;
                        p_T_base_at__p.push_back(___tmp);
                        status_at__p.push_back( MiscUtils::to_eigen( status ) );

                        MatrixXd eigen_result_of_opticalflow_normed_im_cord;
                        StaticPointFeatureMatching::image_coordinates_to_normalized_image_coordinates(
                            xloader.left_camera,
                            eigen_result_of_opticalflow, eigen_result_of_opticalflow_normed_im_cord  );
                        tracked_normed_uv_at__p.push_back( eigen_result_of_opticalflow_normed_im_cord );


                        #if 0
                        //----- triangulate
                        cout << TermColor::GREEN() << "\tTriangulate" << TermColor::RESET() << endl;
                        cout << "\tw_T_al " << PoseManipUtils::prettyprintMatrix4d(w_T_al) << "\n";
                        cout << "\tw_T_al__p " << PoseManipUtils::prettyprintMatrix4d(w_T_al__p) << "\n";
                        Matrix4d al_T_al__p = w_T_al.inverse() * w_T_al__p;
                        cout << "\tbaseline = " << al_T_al__p.col(3).topRows(3).norm() << endl;

                        // loop over each point and triangulate it
                        for( int mm=0 ; mm<eigen_result_of_opticalflow.cols() ; mm++ )
                        {
                            if( status[mm] == 0 )
                                continue;

                            // convert to normalized image co-ordinates
                            Vector2d _uv_a_i (  uv_a(0,mm), uv_a(1,mm) );
                            Vector3d normed_uv_a__i;
                            xloader.left_camera->liftProjective( _uv_a_i, normed_uv_a__i );

                            Vector2d _uv_b_i(  eigen_result_of_opticalflow(0,mm), eigen_result_of_opticalflow(1,mm) );
                            Vector3d normed_uv_b__i;
                            xloader.left_camera->liftProjective( _uv_b_i, normed_uv_b__i );

                            Vector4d _triangulated_X;
                            Triangulation::LinearLSTriangulation( normed_uv_a__i, Matrix4d::Identity(), normed_uv_b__i, al_T_al__p.inverse(), _triangulated_X );

                            //^^ This is more understanding and verification

                            // printing
                            cout << "mm=" << mm << "\t";
                            cout << "d_a(mm)=" << d_a(mm) << "\t";
                            cout << "sf(mm)=" << sf(mm) << "\t";
                            cout << TermColor::YELLOW() << "triangulated=" << _triangulated_X.transpose() << "\t" << TermColor::RESET();
                            cout << endl;
                        }
                        #endif //triangulate

                        #if _VIZ_ //viz
                        cv::Mat dst_tracked;
                        MiscUtils::plot_point_sets_masked( image_a__p, eigen_result_of_opticalflow, status, dst_tracked, cv::Scalar(0,0,255), true, "idx_a="+to_string(idx_a)+" p="+to_string(p)+";"+"total_tracked="+to_string(status.size())+";nfail="+to_string(nfail) );
                        MiscUtils::imshow( "dst_tracked", dst_tracked, 1.0 );

                        cout << "press 'b' to exit the loop of random pair draws, any other key to keep drawing more random pairs\n";
                        ch = cv::waitKey(0);
                        if( ch == 'b' ) {
                            cout << "b pressed, goto end_of_random_draws\n";
                            goto end_of_random_draws;
                        }
                        #endif


                    } // END for( int  p=-8 ; p<8 ; p++ )
                    end_of_p__for_seq_a: ;
                    cout << "\tloop on p ends\n";


                    #if 1
                    // ---- Triangulation (Multi-View)
                    //  This will use: list__p, tracked_uv_at__p, tracked_normed_uv_at__p, p_T_base_at__p, status_at__p
                    //print info of the track
                    cout << TermColor::GREEN() << "\tMultiView Triangulate" << TermColor::RESET() << endl;

                    for( int h=0 ; h<list__p.size() ; h++ )
                    {
                        cout << TermColor::YELLOW() << "rel indx p=" << list__p[h] << "\t" << TermColor::RESET();
                        cout << "#tracked points (total) = " << tracked_uv_at__p[h].cols() << ", " << tracked_normed_uv_at__p[h].cols() << "\t";
                        cout << "#successfully tracked   = " << status_at__p[h].sum() << " of total " << status_at__p[h].size() << "\t";
                        cout << endl;
                    }


                    // Triangulate each point
                    #define __SEQ_A_multiview_triangulation 0
                    ElapsedTime t_multiview_triangulation( "Multiview triangulation");
                    for( int f=0 ; f<normed_uv_a.cols() ; f++ ) // loop over each uv_a
                    {
                        Vector3d _u = normed_uv_a.col(f);


                        vector<Vector3d> _u_tracked; _u_tracked.clear();
                        vector<bool> _u_visible; _u_visible.clear();
                        for( int h=0 ; h<tracked_normed_uv_at__p.size() ; h++ ) // loop on each track for the uv_a[i]
                        {
                            Vector3d _tmp;
                            _tmp = tracked_normed_uv_at__p[h].col(f);
                            _u_tracked.push_back( _tmp );

                            double v = status_at__p[h](f);
                            if( v > 0 )
                                _u_visible.push_back(true);
                            else
                                _u_visible.push_back(false);


                        }


                        Vector4d result_X;
                        bool triangulation_status = Triangulation::MultiViewLinearLSTriangulation(
                        // bool triangulation_status = Triangulation::MultiViewIterativeLSTriangulation(
                            _u,
                            _u_tracked, p_T_base_at__p,
                            result_X, _u_visible );
                        monocular_d_a( f ) = result_X(2);

                        cout << "feat#" << f << ": triangulated=" << result_X.transpose() << "\t";
                        cout << TermColor::YELLOW() << "stereo_depth=" << d_a(f) << "\tsf=" << sf(f) << TermColor::RESET() << "\t";
                        cout << "diff=" << abs(result_X(2) - d_a(f)) << "\t";
                        cout << "triangulation_status=" << triangulation_status;
                        cout << endl;
                    }
                    cout << TermColor::BLUE() << t_multiview_triangulation.toc() << TermColor::RESET() << endl;




                    #endif //Multiview Triangulation


                    #if _VIZ_ //viz
                    cout << "Multiview Triangulation Done, press any key to continue\n";
                    cv::waitKey(0);
                    cv::destroyWindow("dst_tracked");
                    cv::destroyWindow("dst_baseimg_with_feat_to_track");
                    #endif



                } // end processing for depth of seq-a
                #endif

                //--Seq-B, B+1, B+2,...
                #define __SEQ_B_MONOCULAR__ 0
                #if 0
                VectorXd monocular_d_b = VectorXd::Zero( uv_b.cols() );
                {
                    //---- the `base`
                    cout << "\n\n======= Look at adjacent images of idx_b=" << idx_b << endl;
                    json data_node_b = STATE["DataNodes"][idx_b];
                    cv::Mat image_b;
                    Matrix4d w_T_bm;
                    bool status_img  = xloader.retrive_image_data_from_json_datanode( data_node_b, image_b );
                    bool status_pose = xloader.retrive_pose_from_json_datanode( data_node_b, w_T_bm );
                    assert( status_img && status_pose );

                    #if _VIZ_ //viz
                    // plot base image with points to track
                    cv::Mat dst_baseimg_with_feat_to_track;
                    MiscUtils::plot_point_sets(  image_b, uv_b, dst_baseimg_with_feat_to_track,
                        cv::Scalar(255,0,0), true, "base image (idx=" + to_string(idx_b)+ ") to track from n_pts="+to_string(uv_b.cols()) );
                    MiscUtils::imshow( "dst_baseimg_with_feat_to_track", dst_baseimg_with_feat_to_track, 1.0 );
                    #endif


                    //---- next image in seq   ==>  track the points uv_a on idx_a+p
                    // we track the correspondences at say 2 frames ahead and 2 frames back to get an estimate of depth at those points.

                    vector<int> list__p; // the p that we used
                    vector<MatrixXd> tracked_uv_at__p; //the uv (image co-ordinates) points at p
                    vector<MatrixXd> tracked_normed_uv_at__p; //the uv (normalized image co-ordinates) points at p
                    vector<Matrix4d> p_T_base_at__p; //the pose of base wrt camera-p. for every p
                    // vector< vector<uchar> > status_at__p; // status of the tracked points at p.
                    vector<VectorXd> status_at__p; // status of the tracked points at p.
                    for( int  p=-6 ; p<8 ; p++ )
                    {
                        cout <<  "\tp=" << p << endl;
                        json data_node_b__p = STATE["DataNodes"][idx_b+p];
                        if( xloader.is_data_available(data_node_b__p) == false ) {
                            cout << "\tno image or pose data so skip this p\n";
                            continue;
                        }

                        if( p==0 ) {
                            cout << "\tskip p=0\n";
                            continue;
                        }

                        cv::Mat image_b__p;
                        Matrix4d w_T_bm__p;
                        bool status_img__p  = xloader.retrive_image_data_from_json_datanode( data_node_b__p, image_b__p );
                        bool status_pose__p = xloader.retrive_pose_from_json_datanode( data_node_b__p, w_T_bm__p );
                        assert( status_img__p && status_pose__p );


                        Matrix4d bm_T_bm__p = w_T_bm.inverse() * w_T_bm__p;
                        cout << "\tbaseline = " << bm_T_bm__p.col(3).topRows(3).norm() << endl;


                        //---- optical flow
                        vector<Point2f> result_of_opticalflow = pts_to_track_b; //give it an initial guess
                        vector<uchar> status;
                        vector<float> err;
                        cv::TermCriteria criteria = cv::TermCriteria((cv::TermCriteria::COUNT) + (cv::TermCriteria::EPS), 30, 0.01);
                        cv::Size win_size = cv::Size(25,25);
                        int n_pyramids = 4;

                        cout << TermColor::GREEN() << "\tCalculate Optical Flow" << TermColor::RESET() << endl;
                        cout << "\timage_n:" << MiscUtils::cvmat_info( image_b ) << endl;
                        cout << "\timage_n__p:" << MiscUtils::cvmat_info( image_b__p ) << endl;
                        cout << "\tpts_to_track_b.size=" << pts_to_track_b.size() << endl;
                        ElapsedTime t_opticalflow("Optical Flow");
                        cv::calcOpticalFlowPyrLK( image_b, image_b__p, pts_to_track_b, result_of_opticalflow,
                                 status, err, win_size, n_pyramids, criteria );
                        cout << TermColor::BLUE() << t_opticalflow.toc() << TermColor::RESET() << endl;
                                 int nfail = (int)status.size() - MiscUtils::total_positives(status);
                                 cout << "\t\tnfail=" << nfail << endl;

                        cout << "\tshowing adjacent image to idx_b="<< idx_b << "@p=" << p << " pose=" << PoseManipUtils::prettyprintMatrix4d(w_T_bm__p) << endl;
                        MatrixXd eigen_result_of_opticalflow;
                        MiscUtils::point2f_2_eigen( result_of_opticalflow, eigen_result_of_opticalflow, true);


                        //---- save for later
                        list__p.push_back(p);
                        tracked_uv_at__p.push_back( eigen_result_of_opticalflow );
                        Matrix4d ___tmp =  w_T_bm__p.inverse() * w_T_bm ;
                        p_T_base_at__p.push_back(___tmp);
                        status_at__p.push_back( MiscUtils::to_eigen( status ) );

                        MatrixXd eigen_result_of_opticalflow_normed_im_cord;
                        StaticPointFeatureMatching::image_coordinates_to_normalized_image_coordinates(
                            xloader.left_camera,
                            eigen_result_of_opticalflow, eigen_result_of_opticalflow_normed_im_cord  );
                        tracked_normed_uv_at__p.push_back( eigen_result_of_opticalflow_normed_im_cord );


                        #if 0
                        //----- triangulate
                        cout << TermColor::GREEN() << "\tTriangulate" << TermColor::RESET() << endl;
                        cout << "\tw_T_bm " << PoseManipUtils::prettyprintMatrix4d(w_T_bm) << "\n";
                        cout << "\tw_T_bm__p " << PoseManipUtils::prettyprintMatrix4d(w_T_bm__p) << "\n";
                        Matrix4d bm_T_bm__p = w_T_bm.inverse() * w_T_bm__p;
                        cout << "\tbaseline = " << bm_T_bm__p.col(3).topRows(3).norm() << endl;

                        // loop over each point and triangulate it
                        for( int mm=0 ; mm<eigen_result_of_opticalflow.cols() ; mm++ )
                        {
                            if( status[mm] == 0 )
                                continue;

                            // convert to normalized image co-ordinates
                            Vector2d _uv_a_i (  uv_b(0,mm), uv_b(1,mm) );
                            Vector3d normed_uv_a__i;
                            xloader.left_camera->liftProjective( _uv_a_i, normed_uv_a__i );

                            Vector2d _uv_b_i(  eigen_result_of_opticalflow(0,mm), eigen_result_of_opticalflow(1,mm) );
                            Vector3d normed_uv_b__i;
                            xloader.left_camera->liftProjective( _uv_b_i, normed_uv_b__i );

                            Vector4d _triangulated_X;
                            Triangulation::LinearLSTriangulation( normed_uv_a__i, Matrix4d::Identity(), normed_uv_b__i, bm_T_bm__p.inverse(), _triangulated_X );

                            //^^ This is more understanding and verification

                            // printing
                            cout << "mm=" << mm << "\t";
                            cout << "d_b(mm)=" << d_b(mm) << "\t";
                            cout << "sf(mm)=" << sf(mm) << "\t";
                            cout << TermColor::YELLOW() << "triangulated=" << _triangulated_X.transpose() << "\t" << TermColor::RESET();
                            cout << endl;
                        }
                        #endif //triangulate

                        #if _VIZ_ //viz
                        cv::Mat dst_tracked;
                        MiscUtils::plot_point_sets_masked( image_b__p, eigen_result_of_opticalflow, status, dst_tracked, cv::Scalar(0,0,255), true,
                            "idx_b="+to_string(idx_b)+" p="+to_string(p)+";"+"total_tracked="+to_string(status.size())+";nfail="+to_string(nfail) );
                        MiscUtils::imshow( "dst_tracked", dst_tracked, 1.0 );

                        cout << "press 'b' to exit the loop of random pair draws, any other key to keep drawing more random pairs\n";
                        ch = cv::waitKey(0);
                        if( ch == 'b' ) {
                            cout << "b pressed, goto end_of_random_draws\n";
                            goto end_of_random_draws;
                        }
                        #endif


                    } // END for( int  p=-8 ; p<8 ; p++ )
                    end_of_p__for_seq_b: ;
                    cout << "\tloop on p ends\n";


                    #if 1
                    // ---- Triangulation (Multi-View)
                    //  This will use: list__p, tracked_uv_at__p, tracked_normed_uv_at__p, p_T_base_at__p, status_at__p
                    //print info of the track
                    cout << TermColor::GREEN() << "\tMultiView Triangulate (seq-b)" << TermColor::RESET() << endl;

                    for( int h=0 ; h<list__p.size() ; h++ )
                    {
                        cout << TermColor::YELLOW() << "rel indx p=" << list__p[h] << "\t" << TermColor::RESET();
                        cout << "#tracked points (total) = " << tracked_uv_at__p[h].cols() << ", " << tracked_normed_uv_at__p[h].cols() << "\t";
                        cout << "#successfully tracked   = " << status_at__p[h].sum() << " of total " << status_at__p[h].size() << "\t";
                        cout << endl;
                    }


                    // Triangulate each point
                    #define __SEQ_B_multiview_triangulation 0
                    ElapsedTime t_multiview_triangulation("Multiview triangulation");
                    for( int f=0 ; f<normed_uv_b.cols() ; f++ ) // loop over each uv_a
                    {
                        Vector3d _u = normed_uv_b.col(f);


                        vector<Vector3d> _u_tracked; _u_tracked.clear();
                        vector<bool> _u_visible; _u_visible.clear();
                        for( int h=0 ; h<tracked_normed_uv_at__p.size() ; h++ ) // loop on each track for the uv_a[i]
                        {
                            Vector3d _tmp;
                            _tmp = tracked_normed_uv_at__p[h].col(f);
                            _u_tracked.push_back( _tmp );

                            double v = status_at__p[h](f);
                            if( v > 0 )
                                _u_visible.push_back(true);
                            else
                                _u_visible.push_back(false);


                        }


                        Vector4d result_X;
                        bool triangulation_status = Triangulation::MultiViewLinearLSTriangulation(
                        // bool triangulation_status = Triangulation::MultiViewIterativeLSTriangulation(
                            _u,
                            _u_tracked, p_T_base_at__p,
                            result_X, _u_visible );
                        monocular_d_b( f ) = result_X(2);

                        cout << "feat#" << f << ": triangulated=" << result_X.transpose() << "\t";
                        cout << TermColor::YELLOW() << "stereo_depth=" << d_b(f) << "\tsf=" << sf(f) << TermColor::RESET() << "\t";
                        cout << "diff=" << abs(result_X(2) - d_b(f)) << "\t";
                        cout << "triangulation_status=" << triangulation_status ;
                        cout << endl;
                    }
                    cout << TermColor::BLUE() << t_multiview_triangulation.toc() << TermColor::RESET() << endl;

                    #endif //Multiview Triangulation


                    #if _VIZ_ //viz
                    cout << "Multiview Triangulation Done, press any key to continue\n";
                    cv::waitKey(0);
                    cv::destroyWindow("dst_tracked");
                    cv::destroyWindow("dst_baseimg_with_feat_to_track");
                    #endif



                } // end processing for depth of seq-a
                #endif


                // put data for this pair in gobal container
                #define _GLOBAL_CONTAINER_FILL_
                cout << "\n===== put data for this pair (idx_a=" << idx_a << ", idx_b=" << idx_b << "), normed_uv_a.cols()=" << normed_uv_a.cols() << " in gobal container\n";

                // 1. a0_T_a, b0_T_b
                // 2. normed_uv_a, normed_uv_b
                // 3. d_a, d_b, s_f --> Either from stereo or from monocular

                //--1.
                Matrix4d w_T_a, w_T_b;
                bool status_posea = xloader.retrive_pose_from_json_datanode( STATE["DataNodes"][idx_a], w_T_a );
                bool status_poseb = xloader.retrive_pose_from_json_datanode( STATE["DataNodes"][idx_b], w_T_b );
                assert( status_posea && status_poseb );

                Matrix4d a0_T_a = w_T_a0.inverse() * w_T_a;
                Matrix4d b0_T_b = w_T_b0.inverse() * w_T_b;

                all_a0_T_a.push_back( a0_T_a );
                all_b0_T_b.push_back( b0_T_b );
                all_pair_idx.push_back( std::make_pair( idx_a, idx_b ) );
                cout << "[main] 1. done\n";

                //--2.
                all_normed_uv_a.push_back( normed_uv_a );
                all_normed_uv_b.push_back( normed_uv_b );
                cout << "[main] 2. done\n";

                //--3.
                all_d_a.push_back( d_a );
                all_d_b.push_back( d_b );
                all_sf.push_back( sf );
                cout << "[main] 3. done\n";

                //--4. (use the above 3 data and make change the co-ordinate frame to a0 and b0 respectively)
                MatrixXd aX = MatrixXd::Constant( 4, normed_uv_a.cols() , 1.0 );
                for( int q=0 ; q<normed_uv_a.cols() ; q++ ) //depth multiplication
                {
                    // cout << "q="<< q << "    "<<  d_a.size() <<  "\n";
                    #if 1
                    double z = d_a(q);
                    #else
                    double z = monocular_d_a(q);
                    #endif

                    aX(0,q) = normed_uv_a(0,q) * z;
                    aX(1,q) = normed_uv_a(1,q) * z;
                    aX(2,q) = normed_uv_a(2,q) * z;
                    aX(3,q) = 1.0;
                }
                cout << "[main] 4.1 done\n";

                MatrixXd bX = MatrixXd::Constant( 4, normed_uv_b.cols() , 1.0 );
                for( int q=0 ; q<normed_uv_b.cols() ; q++ ) //depth multiplication
                {
                    #if 1
                    double z = d_b(q);
                    #else
                    double z = monocular_d_b(q);
                    #endif
                    bX(0,q) = normed_uv_b(0,q) * z;
                    bX(1,q) = normed_uv_b(1,q) * z;
                    bX(2,q) = normed_uv_b(2,q) * z;
                    bX(3,q) = 1.0;
                }
                cout << "[main] 4.2 done\n";

                vector<bool> valids;
                for( int q=0 ; q<normed_uv_a.cols() ; q++ )
                {
                    // use one of these choices.
                    #if 1
                    if( sf(q) > 0  )
                        valids.push_back( true );
                    else
                        valids.push_back( false );
                    #endif

                    #if 0
                    valids.push_back( true );
                    #endif

                    #if 0
                    double far = 5.0;
                    double near = 0.5;
                    if( monocular_d_a(q) > near && monocular_d_a(q) < far && monocular_d_b(q) > near && monocular_d_b(q) < far )
                        valids.push_back( true );
                    else valids.push_back(false);
                    #endif
                }
                cout << "[main] 4.3 done\n";


                assert( valids.size() == normed_uv_a.cols() );


                MatrixXd a0X = a0_T_a * aX;
                MatrixXd b0X = b0_T_b * bX;
                all_a0X.push_back( a0X );
                all_b0X.push_back( b0X );
                all_valids.push_back( valids );

                cout << "END OF ITR=" << rand_itr << endl;


            } // END for( int rand_itr=0 ; rand_itr<10 ; rand_itr++ )
        end_of_random_draws: cout << "label end_of_random_draws\n";

            cout << TermColor::GREEN() << "\n=====Pose Computation\n" << TermColor::RESET();


            // TODO
            // gather
            MatrixXd dst0, dst1;
            MiscUtils::gather( all_a0X, all_valids, dst0 );
            MiscUtils::gather( all_b0X, all_valids, dst1 );
            cout << TermColor::iWHITE() ;
            cout << "dst0:" << dst0.rows() << "x" << dst0.cols() << "\t";
            cout << "dst1:" << dst1.rows() << "x" << dst1.cols() << "\t";
            cout << TermColor::RESET() << endl;


            // Pose Computation here
            // PoseComputation::alternatingMinimization
            Matrix4d a_T_b = Matrix4d::Identity();
            VectorXd switch_weights = VectorXd::Constant( dst0.cols() , 1.0 );
            ElapsedTime t_alternatingminimization( "Altering Minimizations");
            PoseComputation::alternatingMinimization( dst0, dst1, a_T_b, switch_weights );
            cout << "After alternatingMinimization: " << PoseManipUtils::prettyprintMatrix4d( a_T_b ) << endl;
            // PoseComputation::refine_weighted( dst0, dst1, a_T_b, switch_weights );
            // cout << "After refine: " << PoseManipUtils::prettyprintMatrix4d( a_T_b ) << endl;
            cout << TermColor::BLUE() << t_alternatingminimization.toc() << TermColor::RESET() << endl;


            // Local Bundle
            LocalBundle bundle;
            bundle.inputOdometry( 0, odom__a0_T_a );
            bundle.inputOdometry( 1, odom__b0_T_b );
            bundle.inputInitialGuess( 0, 1, a_T_b );
            bundle.inputOdometry_a0_T_b0( 0, 1, w_T_a0.inverse() * w_T_b0 );
            bundle.inputFeatureMatches( 0, 1, all_normed_uv_a, all_normed_uv_b );
            bundle.inputFeatureMatchesDepths( 0, 1, all_d_a, all_d_b, all_sf );
            bundle.inputFeatureMatchesPoses( 0, 1, all_a0_T_a, all_b0_T_b );
            bundle.inputFeatureMatchesImIdx( 0, 1, all_pair_idx ); //< optional , debug
            bundle.inputOdometryImIdx( 0, odom_a_idx );
            bundle.inputOdometryImIdx( 1, odom_b_idx );

            #if 1 // input images
            bundle.inputSequenceImages( 0, images_seq_a); // if image sequences are not set then cannot do reprojection_test_plot
            bundle.inputSequenceImages( 1, images_seq_b);
            #endif

            #if 1 //input depthmaps
            bundle.inputSequenceDepthMaps( 0, depthmap_seq_a);
            bundle.inputSequenceDepthMaps( 1, depthmap_seq_b);
            #endif

            bundle.print_inputs_info();
            bundle.toJSON("/app/catkin_ws/src/gmm_pointcloud_align/resources/local_bundle/");
            bundle.solve();
            a_T_b = bundle.retrive_optimized_pose( 0, 0, 1, 0 );
            xloader.left_camera->writeParametersToYamlFile( "/app/catkin_ws/src/gmm_pointcloud_align/resources/local_bundle/camera.yaml" );




            //------- Refine Y, tx,ty,tz. get pitch and roll from odometry
            cout << TermColor::RED() << "---\n" << TermColor::RESET();
            cout << "computed a_T_b   = " << PoseManipUtils::prettyprintMatrix4d( a_T_b, " " ) << endl;

            Matrix4d odometry_a_T_b = w_T_a0.inverse() * w_T_b0;
            cout << "odometry_a_T_b = " << PoseManipUtils::prettyprintMatrix4d( odometry_a_T_b, " " ) << endl;

            Matrix4d odometry__ad_T_bd = xloader.imu_T_cam * ( w_T_a0.inverse() * w_T_b0 ) * xloader.imu_T_cam.inverse();
            cout << "odometry__ad_T_bd = " << PoseManipUtils::prettyprintMatrix4d( odometry__ad_T_bd, " " ) << endl;
            Matrix4d computed__ad_T_bd = xloader.imu_T_cam * a_T_b * xloader.imu_T_cam.inverse();
            cout << "computed__ad_T_bd = " << PoseManipUtils::prettyprintMatrix4d( computed__ad_T_bd, " " ) << endl;

            #if 0
            // {
                // get pitch and roll from `odometry__ad_T_bd`
                double odom_ad_ypr_bd[5], odom_ad_xyz_bd[5];
                PoseManipUtils::eigenmat_to_rawyprt( odometry__ad_T_bd, odom_ad_ypr_bd, odom_ad_xyz_bd );


                // get yaw, tx, ty, tz from `computed__ad_T_bd`
                double comp_ad_ypr_bd[5], comp_ad_xyz_bd[5];
                PoseManipUtils::eigenmat_to_rawyprt( computed__ad_T_bd, comp_ad_ypr_bd, comp_ad_xyz_bd );


                // ^^^ at this point the yaw, pitch, roll, tx, ty tz are in imu frame of reference
                //     The 4DOF optimization need to happen in imu frame of reference
                // refine( dst0, dst1, imu_T_cam, yaw, pitch, roll, tx, ty, tz )
                //                                 ^                 ^   ^   ^
                #if 0
                PoseComputation::refine4DOF( dst0, dst1, imu_T_cam,
                    comp_ad_ypr_bd[0], odom_ad_ypr_bd[1], odom_ad_ypr_bd[2],
                    comp_ad_xyz_bd[0],  comp_ad_xyz_bd[1],  comp_ad_xyz_bd[2]
                 )
                 #endif

                // after optimization, convert the refined pose to camera frame of reference
                double hybrid_ad_ypr_bd[5], hybrid_ad_xyz_bd[5];
                hybrid_ad_ypr_bd[0] = comp_ad_ypr_bd[0];
                hybrid_ad_ypr_bd[1] = comp_ad_ypr_bd[1];
                hybrid_ad_ypr_bd[2] = comp_ad_ypr_bd[2];

                hybrid_ad_xyz_bd[0] = comp_ad_xyz_bd[0];
                hybrid_ad_xyz_bd[1] = comp_ad_xyz_bd[1];
                hybrid_ad_xyz_bd[2] = comp_ad_xyz_bd[2];

                Matrix4d hybrid_ad_T_bd, hybrid_a_T_b;
                PoseManipUtils::rawyprt_to_eigenmat( hybrid_ad_ypr_bd, hybrid_ad_xyz_bd, hybrid_ad_T_bd);
                hybrid_a_T_b =  xloader.imu_T_cam.inverse() * hybrid_ad_T_bd *  xloader.imu_T_cam;
                cout << "hybrid_ad_T_bd = " << PoseManipUtils::prettyprintMatrix4d( hybrid_ad_T_bd, " " ) << endl;
                cout << "hybrid_a_T_b = " << PoseManipUtils::prettyprintMatrix4d( hybrid_a_T_b, " " ) << endl;

                // cout << "a_T_b := hybrid_a_T_b\n";
                // a_T_b = hybrid_a_T_b;
            // }
            #endif

            cout << TermColor::RED() << "---\n" << TermColor::RESET();

            #if 1
            MatrixXd AAA = all_odom_poses[ 0 ][0].inverse() * vec_surf_map[ 0 ]->get_surfel_positions();
            RosPublishUtils::publish_3d( marker_pub, AAA,
                "AAA (red)", 0,
                255,0,0, float(1.0), 1.5 );

            MatrixXd BBB = all_odom_poses[ 1 ][0].inverse() * vec_surf_map[ 1 ]->get_surfel_positions();
            RosPublishUtils::publish_3d( marker_pub, BBB,
                "BBB (green)", 0,
                0,255,0, float(1.0), 1.5 );


            MatrixXd YYY = a_T_b * BBB;
            RosPublishUtils::publish_3d( marker_pub, YYY,
                "a_T_b x BBB (cyan)", 0,
                0,255,255, float(1.0), 1.5 );

            MatrixXd YYYx = odometry_a_T_b * BBB;
            RosPublishUtils::publish_3d( marker_pub, YYYx,
                "odometry_a_T_b x BBB (yellow)", 0,
                255,255,0, float(1.0), 1.5 );


            cout << "\n pose computation done, press any key to continue drawing more pair of images\n";
            cv::waitKey(0);
            #endif
        }

        if( ch == '4' ) //manually input pose for visualization
        {
            cout << TermColor::GREEN() << "==== Manual Pose Input ====\n" << TermColor::RESET();

            #if 0
            // ypr
            double ypr[4] = {-32.145,32.535,-14.899};
            double xyz[4] = {-1.113,-0.113,0.122};

            cout << "Keyboard input ypr a_T_b (space separated):";
            cin >> ypr[0] >> ypr[1] >> ypr[2];
            cout << "Keyboard input xyz a_T_b (space separated):";
            cin >> xyz[0] >> xyz[1] >> xyz[2];

            cout << "You input ypr=" << TermColor::GREEN() << ypr[0] << ", " << ypr[1] << ", " << ypr[2] << "\t";
            cout << "xyz=" << xyz[0] << ", " << xyz[1] << ", " << xyz[2] << TermColor::RESET() << endl;

            Matrix4d a_T_b = Matrix4d::Identity();
            PoseManipUtils::rawyprt_to_eigenmat( ypr, xyz, a_T_b );

            #endif


            #if 1
            // 3x3 matrix followed by translation vector
            Matrix4d a_T_b = Matrix4d::Identity();
            cout << "Keyboard input 0th row of rotation matrix a_T_b (3 floats, space separated):";
            cin >> a_T_b(0,0) >>  a_T_b(0,1) >>  a_T_b(0,2) ;
            cout << "Keyboard input 1st row of rotation matrix a_T_b (3 floats, space separated):";
            cin >> a_T_b(1,0) >>  a_T_b(1,1) >>  a_T_b(1,2) ;
            cout << "Keyboard input 2nd row of rotation matrix a_T_b (3 floats, space separated):";
            cin >> a_T_b(2,0) >>  a_T_b(2,1) >>  a_T_b(2,2) ;


            cout << "\nInput translation vector of a_T_b (space separated)";
            cin >> a_T_b(0,3) >>  a_T_b(1,3) >>  a_T_b(2,3) ;

            #endif


            cout << "a_T_b: " << PoseManipUtils::prettyprintMatrix4d( a_T_b ) << endl;
            cout << "a_T_b:\n" << a_T_b << endl;

            MatrixXd AAA = all_odom_poses[ 0 ][0].inverse() * vec_surf_map[ 0 ]->get_surfel_positions();
            RosPublishUtils::publish_3d( marker_pub, AAA,
                "AAA (red)", 0,
                255,0,0, float(1.0), 1.5 );

            MatrixXd BBB = all_odom_poses[ 1 ][0].inverse() * vec_surf_map[ 1 ]->get_surfel_positions();
            RosPublishUtils::publish_3d( marker_pub, BBB,
                "BBB (green)", 0,
                0,255,0, float(1.0), 1.5 );


            MatrixXd YYY = a_T_b * BBB;
            RosPublishUtils::publish_3d( marker_pub, YYY,
                "a_T_b x BBB (cyan)", 0,
                0,255,255, float(1.0), 1.5 );



        }

        if( ch == '9' ) // visualization for normals of a point cloud
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

        // string goicp_fname = xloader.base_path + "/ptcld_goicp_" + to_string( it->first ) + ".txt";
        string goicp_fname = string("/app/catkin_ws/src/gmm_pointcloud_align/resources/") + "/ptcld_goicp_" + to_string( it->first ) + ".txt";

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
