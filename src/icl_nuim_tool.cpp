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
    for( int i=0 ; i<all_loaders.size() ; i++ )
    {
        msg += all_loaders[i].DB_NAME + "_" + to_string( all_loaders[i].DB_IDX ) + ": ";
        msg += to_string( curr_ptr[i] ) + " of " + to_string( all_loaders[i].len() ) + ";";
    }
    MiscUtils::append_status_image( status, msg, .6, cv::Scalar(0,0,0), cv::Scalar(255,255,255) );
}

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


    //--- Loader
    string DB_BASE = "/Bulk_Data/ICL_NUIM_RGBD/";
    //living_room, office_room
    // string DB_NAME = "living_room";
    string DB_NAME = "office_room";
    int DB_IDX = 0; //0,1,2,3

    vector<ICLNUIMLoader> all_loaders;
    vector<int> curr_ptr;
    vector<bool> changed;
    for( int seriesI=0 ; seriesI<4 ; seriesI++ )
    {
        ICLNUIMLoader xloader( DB_BASE, DB_NAME, seriesI );
        // xloader.print_info();
        all_loaders.push_back( xloader );

        curr_ptr.push_back( 0 );
        changed.push_back( true );
    }


    //--- control loop
    while( ros::ok() )
    {
        ros::spinOnce();


        cv::Mat status_im = cv::Mat::zeros( 10, 500, CV_8UC3 );
        print_pointer_status( all_loaders, curr_ptr, status_im );
        cv::imshow( "status", status_im );

        // imshow
        for( int seriesI=0 ; seriesI<4 ; seriesI++ )
        {
            cv::Mat im, depth, depth_falsecolor;
            if( changed[seriesI] ) {
                changed[seriesI] = false;

                // all_loaders[seriesI].retrive_im( curr_ptr[seriesI], im );
                // all_loaders[seriesI].retrive_im_depth( curr_ptr[seriesI], im, depth );
                all_loaders[seriesI].retrive_im_depth( curr_ptr[seriesI], im, depth, depth_falsecolor );

                string winname = all_loaders[seriesI].DB_NAME+"_"+to_string( seriesI );
                cv::imshow( winname.c_str(), im );

                cv::imshow( (winname+"(depth_map)").c_str(), depth_falsecolor );


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

        }

        if( ch == 'z' || ch == 'x' || ch == 'c' || ch == 'v' )
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
            if( curr_ptr[seriesI]+10 < all_loaders[seriesI].len() ) {
                changed[seriesI] = true;
                curr_ptr[seriesI]+=10;
            }

        }



    }




}
