// just to loop arounbd the log json and look arounbd


#include <iostream>
#include <vector>
#include <fstream>
using namespace std;


// JSON
#include "utils/nlohmann/json.hpp"
using json = nlohmann::json;

#include <ros/ros.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>


#include "utils/TermColor.h"

const string base_path = "/Bulk_Data/chkpts_cerebro/";


int main()
{
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



    for( int i=0 ; i< (int) STATE["DataNodes"].size() ; i++ )
    {
        auto data_node = STATE["DataNodes"][i];
        int64_t t_sec = data_node["stampNSec"];
        ros::Time stamp = ros::Time().fromNSec( t_sec );


        if( data_node["isPoseAvailable"] == false || data_node["isKeyFrame"] == false ) {
            // cout << "\tno pose or image data...return false\n";
            continue;
        }


        string imleft_fname = base_path+"/cerebro_stash/left_image__" + std::to_string(stamp.toNSec()) + ".jpg";
        cout << "\tload imleft_fname: " << imleft_fname << endl;
        cv::Mat left_image = cv::imread( imleft_fname, 0 );
        if( !left_image.data ) {
            cout << TermColor::RED() << "[process_this_datanode]ERROR cannot load image...return false\n" << TermColor::RESET();
            return false;
        }
        // cout << "left_image " << MiscUtils::cvmat_info( left_image ) << endl;


        cout << "json.seq = " << data_node["seq"] << endl;
        cv::imshow( "win", left_image );
        cv::waitKey( 0 );
    }
}
