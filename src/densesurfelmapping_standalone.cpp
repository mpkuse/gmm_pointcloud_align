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

#include "utils/RawFileIO.h"
#include "utils/MiscUtils.h"
#include "utils/TermColor.h"

#include "surfel_fusion/surfel_map.h"

const string base_path = "/Bulk_Data/chkpts_cerebro/";


SurfelMap surfel_map(true);

bool densesurfelfusion( json data_node )
{
    //----------- LOAD DATA -------------------//
    ros::Time stamp = ros::Time().fromNSec( data_node["stampNSec"] );

    Matrix4d w_T_c;
    bool status = RawFileIO::read_eigen_matrix4d_fromjson(  data_node["w_T_c"], w_T_c  );
    assert( true );

    string imleft_fname = base_path+"/cerebro_stash/left_image__" + std::to_string(stamp.toNSec()) + ".jpg";
    cout << "\tload imleft_fname: " << imleft_fname << endl;
    cv::Mat left_image = cv::imread( imleft_fname, 0 );
    if( !left_image.data ) {
        cout << TermColor::RED() << "[retrive_data_from_json_datanode]ERROR cannot load image...return false\n" << TermColor::RESET();
        return false;
    }
    cout << "\tleft_image " << MiscUtils::cvmat_info( left_image ) << endl;



    string depth_image_fname = base_path+"/cerebro_stash/depth_image__" + std::to_string(stamp.toNSec()) + ".jpg.png";
    cout << "\tload depth_image_fname: " << depth_image_fname << endl;
    cv::Mat depth_image = cv::imread( depth_image_fname, -1 );
    if( !depth_image.data ) {
        cout << TermColor::RED() << "[retrive_data_from_json_datanode]ERROR cannot load depth-image...return false\n" << TermColor::RESET();
        return false;
    }
    cout << "\tdepth_image " << MiscUtils::cvmat_info( depth_image ) << endl;

    /////
    //// USE :
    ////    stamp, w_T_c, left_image, depth_image


    //------------------- Dense Surfel Fusion ------------------//
    surfel_map.image_input( stamp, left_image );
    surfel_map.depth_input( stamp, depth_image );
    surfel_map.camera_pose__w_T_ci__input( stamp, w_T_c );

    // TODO: 3d point retrive
    cout << "n_active_surfels = " << surfel_map.n_active_surfels() << "\t";
    cout << "n_fused_surfels = " << surfel_map.n_fused_surfels() << "\t";
    cout << "n_surfels = " << surfel_map.n_surfels() << "\n";

    MatrixXd _w_X = surfel_map.get_surfel_positions();
    cout << "_w_X : " << _w_X.rows() << "x" << _w_X.cols() << endl;
    return true;
}


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
        cout << "Press 'k' to process_this_datanode, 'q' to quit, any other key to skip this image.\n";
        char ch = cv::waitKey( 0 );

        if( ch == 'k' )
        {
            densesurfelfusion( data_node );
        }

        if( ch == 'q' )
            break;
    }

    cout << "-----------\nSave: " << base_path+"/file.ply" << "\n--------------\n";
    surfel_map.save_mesh( base_path+"/file.ply" );
}
