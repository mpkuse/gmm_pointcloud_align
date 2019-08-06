#pragma once

/**
Defination of class MeshObject. This is to be used to load mesh.
Functionality provided include
  - Loading of mesh
  - setting mesh pose in the world, ie. w_T_{object}
  - Render object on image, specified a camera.
**/

#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <queue>
#include <ostream>
#include <stdlib.h>

#include <thread>
#include <mutex>
#include <atomic>

//opencv
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <cv_bridge/cv_bridge.h>


#include <ros/ros.h>
#include <ros/package.h>

// #include <nav_msgs/Odometry.h>
// #include <geometry_msgs/Pose.h>
// #include <geometry_msgs/PoseWithCovariance.h>
// #include <geometry_msgs/PoseStamped.h>
// #include <sensor_msgs/PointCloud.h>
// #include <geometry_msgs/Point32.h>
// #include <sensor_msgs/Image.h>
// #include <nav_msgs/Path.h>
// #include <geometry_msgs/Point.h>
// #include <visualization_msgs/Marker.h>
// #include <visualization_msgs/MarkerArray.h>



#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace Eigen;
#include <opencv2/core/eigen.hpp>

using namespace std;


class MeshObject
{
public:
  MeshObject();
  MeshObject( string obj_name, double scaling );

  const string getMeshObjectName() const { return this->obj_name; }
  const string getMeshObjectFullPathName() const { return this->obj_file_fullpath_name; }

  void setObjectWorldPose( Matrix4d w_T_ob );
  bool getObjectWorldPose( Matrix4d& w_T_ob );
  const Matrix4d& getObjectWorldPose() const { return w_T_ob; }

  bool isMeshLoaded( ) const { return m_loaded; }
  bool isWorldPoseAvailable() const { return m_world_pose_available; }



  const MatrixXd& getVertices() const { return  o_X; }
  const MatrixXi& getFaces() const { return  eigen_faces; }

  // void write_debug_xml( const char * fname );
  // bool load_debug_xml( char * fname );
  // bool load_debug_xml( const string& fname );



private:
  string obj_name, obj_file_fullpath_name;
  bool m_loaded; ///< true when object mesh was successfully loaded
  bool m_world_pose_available; ///< true when the object has w_T_{object}.

  Matrix4d w_T_ob;



  bool load_obj( string fname, double scaling );
  vector<Vector3d> vertices;
  MatrixXd o_X; //vertices in object frame-of-ref
  vector<Vector3i> faces;
  MatrixXi eigen_faces;


  void split(const std::string &s, char delim, vector<string>& vec_of_items);


 public:
     double getScalingFactor() {return scaling_; }

 private:
  // This is set from the constructor. However, the vertices are already scaled when load_obj is called.
  // This is essentially just for reference. No need to scale the 3d points again with this value.
  double scaling_;
};

// CUrrently not in use
class MeshVertex
{
public:
  MeshVertex(double x, double y, double z) { v << x,y,z; }
  Vector3d getVertex( ) { return v; }
  Vector4d getVertexHomogeneous( ) { Vector4d v_h; v_h << v, 1.0; return v_h; }

private:
  Vector3d v;
};

// CUrrently not in use
class MeshTriangulatedFace
{
public:
  MeshTriangulatedFace(double f1, double f2, double f3) { f << f1,f2,f3; }
  Vector3i getVertex( ) { return f; }

private:
  Vector3i f;
};
