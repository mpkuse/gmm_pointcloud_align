#pragma once

//
// This provides utilities creating ros markers
//
// Author : Manohar Kuse <mpkuse@connect.ust.hk>
// Notes: http://wiki.ros.org/rviz/DisplayTypes/Marker

#include <iostream>
#include <string>


#include <Eigen/Dense>
#include <Eigen/Geometry>


#include <ros/ros.h>
#include <ros/package.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseWithCovariance.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point32.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>


using namespace std;
using namespace Eigen;


class RosMarkerUtils
{
public:
    ////////////////// INIT /////////////////////////


    /// Initialize a camera with a few lines. You need to set the `ns` and `id` before publishing.
    static void init_camera_marker( visualization_msgs::Marker& marker, float cam_size );
    static void init_XYZ_axis_marker( visualization_msgs::Marker& marker, float size=1.0, float linewidth_multiplier=1.0 );
    static void init_plane_marker( visualization_msgs::Marker& marker,
        float width, float height,
        float clr_r=1.0, float clr_g=1.0, float clr_b=1.0, float clr_a=0.6 ); //!< built as TRIANGLE_LIST. have 2 triangles, you can set the pose to get to whatever position it is needed

    // a line strip to visualize mu and sigma in either 2d or 3d.
    static void init_mu_sigma_marker( visualization_msgs::Marker& marker, const VectorXd& mu, const MatrixXd& sigma, float linewidth_multiplier=1.0 );

    static void init_text_marker( visualization_msgs::Marker &marker );
    static void init_line_strip_marker( visualization_msgs::Marker &marker );
    static void init_line_marker( visualization_msgs::Marker &marker );
    static void init_line_marker( visualization_msgs::Marker &marker, const Vector3d& p1, const Vector3d& p2 );
    static void init_line_marker( visualization_msgs::Marker &marker, const MatrixXd& p1, const MatrixXd& p2, const vector<bool>& valids );
    static void init_points_marker( visualization_msgs::Marker &marker );

    static void init_mesh_marker( visualization_msgs::Marker &marker );

    //////////////// SET //////////////////////////
    static void setpose_to_marker( const Matrix4d& w_T_c, visualization_msgs::Marker& marker );
    static void setposition_to_marker( const Vector3d& w_t_c, visualization_msgs::Marker& marker );
    static void setposition_to_marker( const Vector4d& w_t_c, visualization_msgs::Marker& marker );
    static void setposition_to_marker( float x, float y, float z, visualization_msgs::Marker& marker );
    static void setcolor_to_marker( float r, float g, float b, visualization_msgs::Marker& marker  );
    static void setcolor_to_marker( float r, float g, float b, float a, visualization_msgs::Marker& marker  );

    static void setscaling_to_marker( float sc_x, float sc_y, float sc_z, visualization_msgs::Marker& marker );
    static void setscaling_to_marker( float sc, visualization_msgs::Marker& marker );
    static void setXscaling_to_marker( float sc_x, visualization_msgs::Marker& marker );
    static void setYscaling_to_marker( float sc_y, visualization_msgs::Marker& marker );
    static void setZscaling_to_marker( float sc_z, visualization_msgs::Marker& marker );

    //////////////// Add Points ////////////////////
    static void add_point_to_marker( float x, float y, float z, visualization_msgs::Marker& marker, bool clear_prev_points=true );
    static void add_point_to_marker( const Vector3d& X, visualization_msgs::Marker& marker, bool clear_prev_points=true );
    static void add_point_to_marker( const Vector4d& X, visualization_msgs::Marker& marker, bool clear_prev_points=true );
    static void add_points_to_marker( const MatrixXd& X, visualization_msgs::Marker& marker, bool clear_prev_points=true ); //X : 3xN or 4xN.


    ////////////// Add colors to individual points ///////////////
    static void add_colors_to_marker( const Vector3d& color_rgb, visualization_msgs::Marker& marker, bool clear_prev_colors );
    static void add_colors_to_marker( float c_r, float c_g, float c_b, visualization_msgs::Marker& marker, bool clear_prev_colors );
    static void add_colors_to_marker( const MatrixXd& X, visualization_msgs::Marker& marker, bool clear_prev_colors );


};


#include "MiscUtils.h"
class RosPublishUtils
{
public:

    // Given a set of 3d points, publish them.
    //      pub: The publisher object
    //      _3dpts: either 3xN or 4xN
    //      ns, id: namespace and id of the visualization_msgs::Marker
    //      red,green,blue : 0, 255.
    static void publish_3d( ros::Publisher& pub, MatrixXd& _3dpts, string ns, int id,
        float red, float green, float blue, float alpha=1.0,
        int size_multiplier=1.0 );

    static void publish_3d( ros::Publisher& pub, MatrixXd& _3dpts, string ns, int id,
        int colorcode_by_dim, double min_val, double max_val,
        int size_multiplier=1.0 );


    static void publish_3d( ros::Publisher& pub, MatrixXd& _3dpts, string ns, int id,
        vector< cv::Scalar > per_pt_color, 
        int size_multiplier=1.0 );
};
