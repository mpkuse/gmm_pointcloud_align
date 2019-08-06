#include "MeshObject.h"

MeshObject::MeshObject()
{
  m_loaded = false;
  m_world_pose_available = false;
  obj_name = string( "N/A" );

  scaling_ = 1.0;
}

MeshObject::MeshObject(const string obj_name, double scaling )
{
  // when the vertices are read, a scaling is applied on the co-ordinates

  cout << "====Constructor MeshObject====\n";
  m_loaded = false;
  m_world_pose_available = false;
  this->obj_name = string(obj_name);

  string path = ros::package::getPath("gmm_pointcloud_align") + "/resources/";
  string obj_file_nme = path + obj_name;
  this->obj_file_fullpath_name = obj_file_nme;

  // cout << "Resource Path : " << path << endl;
  cout << "Open File     : " << obj_file_nme << endl;
  cout << "Name          : " << obj_name << endl;

  // Loading mesh here
  load_obj( obj_file_nme , scaling);


  m_world_pose_available = false;
  m_loaded = true;

  scaling_ = scaling;

}


void MeshObject::setObjectWorldPose( Matrix4d w_T_ob )
{
  cout << "[MeshObject::setObjectWorldPose::" << obj_name << "]w_T_ob\n" << w_T_ob << endl;
  this->w_T_ob = Matrix4d( w_T_ob );
  m_world_pose_available = true;
}


bool MeshObject::getObjectWorldPose( Matrix4d& w_T_ob )
{
  w_T_ob = Matrix4d( this->w_T_ob );
  return m_world_pose_available;
}


bool MeshObject::load_obj( string fname, double scaling )
{

  cout << "MeshObject::load_obj()\n";

  ifstream myfile( fname.c_str() );

  if( !myfile.is_open() )
  {
    ROS_ERROR_STREAM( "Fail to open file: "<< fname );
    return false;
  }


  // line-by-line reading
  vector<string> vec_of_items;
  int nvertex=0, nfaces = 0;


  for( string line; getline( myfile, line ) ; )
  {
    // cout << "l:" << line << endl;
    split( line, ' ', vec_of_items );
    if( vec_of_items.size() <= 0 )
      continue;

    if( vec_of_items[0] == "v" )
    {
      nvertex++;
      Vector3d vv;
      vv << stod( vec_of_items[1] ), stod( vec_of_items[2] ), stod( vec_of_items[3] ) ;
      vertices.push_back( vv );
    }


    if( vec_of_items[0] == "f" )
    {
      nfaces++;
      Vector3i vv;
      vv << stoi( vec_of_items[1] ), stod( vec_of_items[2] ), stod( vec_of_items[3] ) ;
      faces.push_back( vv );
    }


  }


  cout << "Vertex: "<< nvertex << "  Faces: " << nfaces << endl;
  o_X = MatrixXd( 4, nvertex );
  for( int i=0 ; i<nvertex ; i++ )
  {
    o_X.col(i) << scaling * vertices[i], 1.0 ;
  }

  eigen_faces = MatrixXi( 3, nfaces );
  for( int i=0 ; i<nfaces ; i++ )
  {
    eigen_faces.col(i) = faces[i];
  }

  cout << "end MeshObject::load_obj()\n";
  return true;

}


void MeshObject::split(const std::string &s, char delim, vector<string>& vec_of_items)
{
    std::stringstream ss(s);
    std::string item;
    // vector<string> vec_of_items;
    vec_of_items.clear();
    while (std::getline(ss, item, delim)) {
        // *(result++) = item;
        vec_of_items.push_back( item );
    }
}
