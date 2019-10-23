// this will look at STATE json. The output from cerebro



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


#include "XLoader.h"


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



    for( int i=0 ; i< (int) STATE["DataNodes"].size() ; i++ )
    {
        auto data_node = STATE["DataNodes"][i];
        int64_t t_sec = data_node["stampNSec"];
        ros::Time stamp = ros::Time().fromNSec( t_sec );
        cout << "json.seq = " << data_node["seq"] << endl;

        if( xloader.is_data_available(data_node) == false ) {
            cout << "no image or pose data, skip...\n";
            continue;
        }


        cv::Mat left_image;
        xloader.retrive_image_data_from_json_datanode( data_node, left_image );


        cv::imshow( "win", left_image );
        cv::waitKey( 0 );
    }

}
