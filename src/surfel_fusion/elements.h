#pragma once
#include <ros/ros.h>
#include <geometry_msgs/Pose.h>

#include <vector>

struct Superpixel_seed
{
    float x, y;
    float size;
    float norm_x, norm_y, norm_z;
    float posi_x, posi_y, posi_z;
    float view_cos;
    float mean_depth;
    float mean_intensity;
    bool fused;
    bool stable;

    // for debug
    float min_eigen_value;
    float max_eigen_value;
};

struct SurfelElement
{
    float px, py, pz;
    float nx, ny, nz;
    float size;
    float color;
    float weight;
    int update_times;
    int last_update;

    // added by mpkuse
    // Which frames and which pixels were used to construct this Surfel elements
    std::vector<int> updates_frameid;
    std::vector<float> updates_imx;
    std::vector<float> updates_imy;
    

};


struct PoseElement{
    //pose_index is the index in the vector in the database
    std::vector<SurfelElement> attached_surfels;
    geometry_msgs::Pose cam_pose;
    geometry_msgs::Pose loop_pose;
    std::vector<int> linked_pose_index;
    int points_begin_index;
    int points_pose_index;
    ros::Time cam_stamp;
    PoseElement() : points_begin_index(-1), points_pose_index(-1) {}
};
