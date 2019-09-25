#include "RosMarkerUtils.h"

// cam_size = 1: means basic size. 1.5 will make it 50% bigger.
void RosMarkerUtils::init_camera_marker( visualization_msgs::Marker& marker, float cam_size )
{
     marker.header.frame_id = "world";
     marker.header.stamp = ros::Time::now();
     marker.action = visualization_msgs::Marker::ADD;
     marker.color.a = .7; // Don't forget to set the alpha!
     marker.type = visualization_msgs::Marker::LINE_LIST;
    //  marker.id = i;
    //  marker.ns = "camerapose_visual";

     marker.scale.x = 0.003; //width of line-segments
     float __vcam_width = 0.07*cam_size;
     float __vcam_height = 0.04*cam_size;
     float __z = 0.1*cam_size;




     marker.points.clear();
     geometry_msgs::Point pt;
     pt.x = 0; pt.y=0; pt.z=0;
     marker.points.push_back( pt );
     pt.x = __vcam_width; pt.y=__vcam_height; pt.z=__z;
     marker.points.push_back( pt );
     pt.x = 0; pt.y=0; pt.z=0;
     marker.points.push_back( pt );
     pt.x = -__vcam_width; pt.y=__vcam_height; pt.z=__z;
     marker.points.push_back( pt );
     pt.x = 0; pt.y=0; pt.z=0;
     marker.points.push_back( pt );
     pt.x = __vcam_width; pt.y=-__vcam_height; pt.z=__z;
     marker.points.push_back( pt );
     pt.x = 0; pt.y=0; pt.z=0;
     marker.points.push_back( pt );
     pt.x = -__vcam_width; pt.y=-__vcam_height; pt.z=__z;
     marker.points.push_back( pt );

     pt.x = __vcam_width; pt.y=__vcam_height; pt.z=__z;
     marker.points.push_back( pt );
     pt.x = -__vcam_width; pt.y=__vcam_height; pt.z=__z;
     marker.points.push_back( pt );
     pt.x = -__vcam_width; pt.y=__vcam_height; pt.z=__z;
     marker.points.push_back( pt );
     pt.x = -__vcam_width; pt.y=-__vcam_height; pt.z=__z;
     marker.points.push_back( pt );
     pt.x = -__vcam_width; pt.y=-__vcam_height; pt.z=__z;
     marker.points.push_back( pt );
     pt.x = __vcam_width; pt.y=-__vcam_height; pt.z=__z;
     marker.points.push_back( pt );
     pt.x = __vcam_width; pt.y=-__vcam_height; pt.z=__z;
     marker.points.push_back( pt );
     pt.x = __vcam_width; pt.y=__vcam_height; pt.z=__z;
     marker.points.push_back( pt );


     // TOSET
    marker.pose.position.x = 0.;
    marker.pose.position.y = 0.;
    marker.pose.position.z = 0.;
    marker.pose.orientation.x = 0.;
    marker.pose.orientation.y = 0.;
    marker.pose.orientation.z = 0.;
    marker.pose.orientation.w = 1.;
    // marker.id = i;
    // marker.ns = "camerapose_visual";
    marker.color.r = 0.2;marker.color.b = 0.;marker.color.g = 0.;
}


void RosMarkerUtils::init_XYZ_axis_marker( visualization_msgs::Marker& axis, float scale, float linewidth_multiplier )
{
    // visualization_msgs::Marker axis;
    RosMarkerUtils::init_line_marker( axis );
    // axis.id = id;
    // axis.ns = ns.c_str();
    axis.scale.x = 0.2*linewidth_multiplier; // (scale<0)?1.0:scale;

    float f = scale;
    // Add pts
    RosMarkerUtils::add_point_to_marker( 0,0,0, axis, true );
    RosMarkerUtils::add_point_to_marker( f,0,0, axis, false );
    RosMarkerUtils::add_point_to_marker( 0,0,0, axis, false );
    RosMarkerUtils::add_point_to_marker( 0,f,0, axis, false );
    RosMarkerUtils::add_point_to_marker( 0,0,0, axis, false );
    RosMarkerUtils::add_point_to_marker( 0,0,f, axis, false );

    // Add colors to each pt
    RosMarkerUtils::add_colors_to_marker( 1.0,0,0, axis, true );
    RosMarkerUtils::add_colors_to_marker( 1.0,0,0, axis, false );
    RosMarkerUtils::add_colors_to_marker( 0,1.0,0, axis, false );
    RosMarkerUtils::add_colors_to_marker( 0,1.0,0, axis, false );
    RosMarkerUtils::add_colors_to_marker( 0,0,1.0, axis, false );
    RosMarkerUtils::add_colors_to_marker( 0,0,1.0, axis, false );

}

void RosMarkerUtils::init_plane_marker( visualization_msgs::Marker& marker,
    float width, float height,
    float clr_r, float clr_g, float clr_b, float clr_a )
{
    marker.header.frame_id = "world";
    marker.header.stamp = ros::Time::now();
    marker.action = visualization_msgs::Marker::ADD;
    marker.type = visualization_msgs::Marker::TRIANGLE_LIST;

    marker.scale.x = 1.0;
    marker.scale.y = 1.0;
    marker.scale.z = 1.0;



    //// Done . no need to edit firther
    marker.pose.position.x = 0.;
    marker.pose.position.y = 0.;
    marker.pose.position.z = 0.;
    marker.pose.orientation.x = 0.;
    marker.pose.orientation.y = 0.;
    marker.pose.orientation.z = 0.;
    marker.pose.orientation.w = 1.;
    // marker.id = i;
    // marker.ns = "_plane_";
    marker.color.r = 0.2;marker.color.b = 0.;marker.color.g = 0.;
    marker.color.a = .8; // Don't forget to set the alpha!

    float w = width/2.0;
    float h = height/2.0;

    marker.points.clear();
    geometry_msgs::Point pt1, pt2, pt3;

    // Triangle1: (-w,-h), (-w,h), (w,h)
    pt1.x = -w;  pt1.y = -h; pt1.z = 0.;
    pt2.x = -w;  pt2.y =  h; pt2.z = 0.;
    pt3.x =  w;  pt3.y =  h; pt3.z = 0.;
    marker.points.push_back( pt1 );
    marker.points.push_back( pt2 );
    marker.points.push_back( pt3 );

    // Triangle2: (-w,-h), (w,h), (w,-h)
    pt1.x = -w;  pt1.y = -h; pt1.z = 0.;
    pt2.x =  w;  pt2.y =  h; pt2.z = 0.;
    pt3.x =  w;  pt3.y = -h; pt3.z = 0.;
    marker.points.push_back( pt1 );
    marker.points.push_back( pt2 );
    marker.points.push_back( pt3 );

    std_msgs::ColorRGBA vertex_color;
    vertex_color.r = clr_r; vertex_color.g = clr_g; vertex_color.b = clr_b; vertex_color.a = clr_a;
    for( int i=0 ; i<6 ; i++ )
        marker.colors.push_back( vertex_color );

}


void RosMarkerUtils::init_mesh_marker( visualization_msgs::Marker &marker )
{
    marker.scale.x = 1.0; //msg.scale * 0.45;
    marker.scale.y = 1.0; //msg.scale * 0.45;
    marker.scale.z = 1.0; //msg.scale * 0.45;

    marker.color.r = 1.0;
    marker.color.g = 0.5;
    marker.color.b = 0.5;
    marker.color.a = .9;

    marker.header.frame_id = "world";
    marker.header.stamp = ros::Time::now();

    // marker.ns = "";
    // marker.id = 0;
    // marker.mesh_resource = "package://nap/resources/1.obj";

    marker.type = visualization_msgs::Marker::MESH_RESOURCE;


}


void RosMarkerUtils::init_mu_sigma_marker( visualization_msgs::Marker& marker,
    const VectorXd& mu, const MatrixXd& sigma, float linewidth_multiplier)
{
    int d = mu.rows();
    assert( d > 0 && linewidth_multiplier > 0. );
    if( ( d == 2 || d == 3) == false ) {
        cout << "[RosMarkerUtils::init_mu_sigma_marker] ERROR dimension can only be either 2 or 3\n";
        exit(1);
    }

    if( (sigma.rows() == d && sigma.cols() == d) == false )
    {
        cout << "[[RosMarkerUtils::init_mu_sigma_marker] ERROR dimensions of sigma and mu are not same\n";
        exit(1);
    }

    RosMarkerUtils::init_line_marker( marker );
    // axis.id = id;
    // axis.ns = ns.c_str();
    marker.scale.x = 0.2*linewidth_multiplier; // (scale<0)?1.0:scale;

    if( d == 2 ) {
    cout << "[RosMarkerUtils::init_mu_sigma_marker]\n";
    cout << "input mu = " << mu << endl;
    cout << "input sigma = " << sigma << endl;

    // Eigen decomposition of sigma
    EigenSolver<Matrix2d> es(sigma, true);
    Vector2cd eigvals_org = es.eigenvalues();
    Matrix2cd eigvecs_org = es.eigenvectors();

    Vector2d eigvals = eigvals_org.real();
    Matrix2d eigvecs = eigvecs_org.real();

    // VectorXd
    cout << "eigvals=\n" << eigvals.real() << endl;
    cout << "eigvecs=\n" << eigvecs.real() << endl;


    Vector3d e0, e1, mu3d;
    e0 << eigvecs.col(0) , 0.0;
    e1 << eigvecs.col(1) , 0.0;
    mu3d << mu, 0.0;


    // // Add pts
    RosMarkerUtils::add_point_to_marker( mu3d , marker, true );
    RosMarkerUtils::add_point_to_marker( Vector3d(mu3d + sqrt(eigvals(0)) * e0), marker, false );

    RosMarkerUtils::add_point_to_marker( mu3d , marker, false );
    RosMarkerUtils::add_point_to_marker( Vector3d(mu3d + sqrt(eigvals(1)) * e1), marker, false );


    cout << "END [RosMarkerUtils::init_mu_sigma_marker]\n";
    return;

    }

    if( d == 3 ) {

        #if 1
        cout << "[RosMarkerUtils::init_mu_sigma_marker]\n";
        cout << "Implemented but not tested. please test this before removing this exit(1).\n";
        // exit(1);
        cout << "input mu = " << mu << endl;
        cout << "input sigma = " << sigma << endl;

        // Eigen decomposition of sigma
        EigenSolver<Matrix3d> es(sigma, true);
        Vector3cd eigvals_org = es.eigenvalues();
        Matrix3cd eigvecs_org = es.eigenvectors();

        Vector3d eigvals = eigvals_org.real();
        Matrix3d eigvecs = eigvecs_org.real();

        // VectorXd
        cout << "eigvals=\n" << eigvals.real() << endl;
        cout << "eigvecs=\n" << eigvecs.real() << endl;


        Vector3d e0, e1, e2, mu3d;
        e0 << eigvecs.col(0);
        e1 << eigvecs.col(1);
        e2 << eigvecs.col(2);
        mu3d << mu ;


        // // Add pts
        RosMarkerUtils::add_point_to_marker( mu3d , marker, true );
        RosMarkerUtils::add_point_to_marker( Vector3d(mu3d + eigvals(0) * e0), marker, false );

        RosMarkerUtils::add_point_to_marker( mu3d , marker, false );
        RosMarkerUtils::add_point_to_marker( Vector3d(mu3d + eigvals(1) * e1), marker, false );

        RosMarkerUtils::add_point_to_marker( mu3d , marker, false );
        RosMarkerUtils::add_point_to_marker( Vector3d(mu3d + eigvals(2) * e2), marker, false );

        cout << "END [RosMarkerUtils::init_mu_sigma_marker]\n";
        return;
        #endif

    }

    cout << "d was something bizzare...\n";
    exit(2);

}

void RosMarkerUtils::setpose_to_marker( const Matrix4d& w_T_c, visualization_msgs::Marker& marker )
{
    Quaterniond quat( w_T_c.topLeftCorner<3,3>() );
    marker.pose.position.x = w_T_c(0,3);
    marker.pose.position.y = w_T_c(1,3);
    marker.pose.position.z = w_T_c(2,3);
    marker.pose.orientation.x = quat.x();
    marker.pose.orientation.y = quat.y();
    marker.pose.orientation.z = quat.z();
    marker.pose.orientation.w = quat.w();
}

void RosMarkerUtils::setposition_to_marker( const Vector3d& w_t_c, visualization_msgs::Marker& marker )
{
    marker.pose.position.x = w_t_c(0);
    marker.pose.position.y = w_t_c(1);
    marker.pose.position.z = w_t_c(2);
}

void RosMarkerUtils::setposition_to_marker( const Vector4d& w_t_c, visualization_msgs::Marker& marker )
{
    marker.pose.position.x = w_t_c(0) / w_t_c(3); ;
    marker.pose.position.y = w_t_c(1) / w_t_c(3); ;
    marker.pose.position.z = w_t_c(2) / w_t_c(3); ;
}

void RosMarkerUtils::setposition_to_marker( float x, float y, float z, visualization_msgs::Marker& marker )
{
    marker.pose.position.x = x;
    marker.pose.position.y = y;
    marker.pose.position.z = z;
}

void RosMarkerUtils::setcolor_to_marker( float r, float g, float b, visualization_msgs::Marker& marker  )
{
    assert( r>=0. && r<=1.0 && g>=0. && g<=1.0 && b>=0 && b<=1.0 );
    marker.color.a = 1.0;
    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;
}

void RosMarkerUtils::setcolor_to_marker( float r, float g, float b, float a, visualization_msgs::Marker& marker  )
{
    assert( r>=0. && r<=1.0 && g>=0. && g<=1.0 && b>=0 && b<=1.0 && a>0. && a<=1.0);
    marker.color.a = a;
    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;
}


void RosMarkerUtils::setscaling_to_marker( float sc_x, float sc_y, float sc_z, visualization_msgs::Marker& marker )
{
    assert( sc_x> 0 && sc_y > 0 && sc_z > 0 );
    marker.scale.x = sc_x;
    marker.scale.y = sc_y;
    marker.scale.z = sc_z;
}

void RosMarkerUtils::setscaling_to_marker( float sc, visualization_msgs::Marker& marker )
{
    assert( sc > 0 );
    marker.scale.x = sc;
    marker.scale.y = sc;
    marker.scale.z = sc;
}

void RosMarkerUtils::setXscaling_to_marker( float sc_x, visualization_msgs::Marker& marker )
{
    assert( sc_x> 0 );
    marker.scale.x = sc_x;
}

void RosMarkerUtils::setYscaling_to_marker( float sc_y, visualization_msgs::Marker& marker )
{
    assert( sc_y> 0 );
    marker.scale.y = sc_y;
}

void RosMarkerUtils::setZscaling_to_marker( float sc_z, visualization_msgs::Marker& marker )
{
    assert( sc_z> 0 );
    marker.scale.z = sc_z;
}

void RosMarkerUtils::init_text_marker( visualization_msgs::Marker &marker )
{
    marker.header.frame_id = "world";
    marker.header.stamp = ros::Time::now();
    marker.action = visualization_msgs::Marker::ADD;
    marker.color.a = .8; // Don't forget to set the alpha!
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;

    marker.scale.x = 1.; //not in use
    marker.scale.y = 1.; //not in use
    marker.scale.z = 1.;

    //// Done . no need to edit firther
    marker.pose.position.x = 0.;
    marker.pose.position.y = 0.;
    marker.pose.position.z = 0.;
    marker.pose.orientation.x = 0.;
    marker.pose.orientation.y = 0.;
    marker.pose.orientation.z = 0.;
    marker.pose.orientation.w = 1.;
    // marker.id = i;
    // marker.ns = "camerapose_visual";
    marker.color.r = 0.2;marker.color.b = 0.;marker.color.g = 0.;

}

void RosMarkerUtils::init_line_strip_marker( visualization_msgs::Marker &marker )
{
    marker.header.frame_id = "world";
    marker.header.stamp = ros::Time::now();
    marker.action = visualization_msgs::Marker::ADD;
    marker.color.a = .8; // Don't forget to set the alpha!
    marker.type = visualization_msgs::Marker::LINE_STRIP;

    marker.scale.x = 0.02;

    marker.points.clear();

    //// Done . no need to edit firther
    marker.pose.position.x = 0.;
    marker.pose.position.y = 0.;
    marker.pose.position.z = 0.;
    marker.pose.orientation.x = 0.;
    marker.pose.orientation.y = 0.;
    marker.pose.orientation.z = 0.;
    marker.pose.orientation.w = 1.;
    // marker.id = i;
    // marker.ns = "camerapose_visual";
    marker.color.r = 0.2;marker.color.b = 0.;marker.color.g = 0.;

}

void RosMarkerUtils::init_line_marker( visualization_msgs::Marker &marker )
{
    marker.header.frame_id = "world";
    marker.header.stamp = ros::Time::now();
    marker.action = visualization_msgs::Marker::ADD;
    marker.color.a = .8; // Don't forget to set the alpha!
    marker.type = visualization_msgs::Marker::LINE_LIST;

    marker.scale.x = 0.02;

    marker.points.clear();

    //// Done . no need to edit firther
    marker.pose.position.x = 0.;
    marker.pose.position.y = 0.;
    marker.pose.position.z = 0.;
    marker.pose.orientation.x = 0.;
    marker.pose.orientation.y = 0.;
    marker.pose.orientation.z = 0.;
    marker.pose.orientation.w = 1.;
    // marker.id = i;
    // marker.ns = "camerapose_visual";
    marker.color.r = 0.2;marker.color.b = 0.;marker.color.g = 0.;

}

void RosMarkerUtils::init_line_marker( visualization_msgs::Marker &marker, const Vector3d& p1, const Vector3d& p2 )
{
    marker.header.frame_id = "world";
    marker.header.stamp = ros::Time::now();
    marker.action = visualization_msgs::Marker::ADD;
    marker.color.a = .8; // Don't forget to set the alpha!
    marker.type = visualization_msgs::Marker::LINE_LIST;

    marker.scale.x = 0.02;

    marker.points.clear();
    geometry_msgs::Point pt;
    pt.x = p1(0);
    pt.y = p1(1);
    pt.z = p1(2);
    marker.points.push_back( pt );
    pt.x = p2(0);
    pt.y = p2(1);
    pt.z = p2(2);
    marker.points.push_back( pt );

    //// Done . no need to edit firther
    marker.pose.position.x = 0.;
    marker.pose.position.y = 0.;
    marker.pose.position.z = 0.;
    marker.pose.orientation.x = 0.;
    marker.pose.orientation.y = 0.;
    marker.pose.orientation.z = 0.;
    marker.pose.orientation.w = 1.;
    // marker.id = i;
    // marker.ns = "camerapose_visual";
    marker.color.r = 0.2;marker.color.b = 0.;marker.color.g = 0.;

}



void RosMarkerUtils::init_line_marker( visualization_msgs::Marker &marker,
    const MatrixXd& p1, const MatrixXd& p2,
    const vector<bool>& valids )
{
    assert( p1.rows() == 3 || p1.rows() ==4 );
    assert( p1.rows() == p2.rows() );
    assert( p1.cols() == p2.cols() );

    bool selective = false;
    if( valids.size() == p1.cols() )
        selective = true;

    marker.header.frame_id = "world";
    marker.header.stamp = ros::Time::now();
    marker.action = visualization_msgs::Marker::ADD;
    marker.color.a = .8; // Don't forget to set the alpha!
    marker.type = visualization_msgs::Marker::LINE_LIST;

    marker.scale.x = 0.02;

    marker.points.clear();

    for( int i=0 ; i<p1.cols() ; i++ )
    {
        if( selective && valids[i]==false )
            continue;
        geometry_msgs::Point pt;
        pt.x = p1(0,i);
        pt.y = p1(1,i);
        pt.z = p1(2,i);
        marker.points.push_back( pt );
        pt.x = p2(0,i);
        pt.y = p2(1,i);
        pt.z = p2(2,i);
        marker.points.push_back( pt );
    }

    //// Done . no need to edit firther
    marker.pose.position.x = 0.;
    marker.pose.position.y = 0.;
    marker.pose.position.z = 0.;
    marker.pose.orientation.x = 0.;
    marker.pose.orientation.y = 0.;
    marker.pose.orientation.z = 0.;
    marker.pose.orientation.w = 1.;
    // marker.id = i;
    // marker.ns = "camerapose_visual";
    marker.color.r = .9;marker.color.b = 0.2;marker.color.g = 0.8;

}


void RosMarkerUtils::init_points_marker( visualization_msgs::Marker &marker )
{
    marker.header.frame_id = "world";
    marker.header.stamp = ros::Time::now();
    marker.action = visualization_msgs::Marker::ADD;
    marker.color.a = .8; // Don't forget to set the alpha!
    marker.type = visualization_msgs::Marker::POINTS;

    marker.scale.x = 0.04;
    marker.scale.y = 0.04;

    marker.points.clear();

    //// Done . no need to edit firther
    marker.pose.position.x = 0.;
    marker.pose.position.y = 0.;
    marker.pose.position.z = 0.;
    marker.pose.orientation.x = 0.;
    marker.pose.orientation.y = 0.;
    marker.pose.orientation.z = 0.;
    marker.pose.orientation.w = 1.;
    // marker.id = i;
    // marker.ns = "camerapose_visual";
    marker.color.r = 0.2;marker.color.b = 0.;marker.color.g = 0.;

}




void RosMarkerUtils::add_point_to_marker( float x, float y, float z, visualization_msgs::Marker& marker, bool clear_prev_points )
{
    if( clear_prev_points )
        marker.points.clear();

    geometry_msgs::Point pt;
    pt.x = x; pt.y = y; pt.z = z;
    marker.points.push_back( pt );
}

void RosMarkerUtils::add_point_to_marker( const Vector3d& X, visualization_msgs::Marker& marker, bool clear_prev_points )
{
    if( clear_prev_points )
        marker.points.clear();

    geometry_msgs::Point pt;
    pt.x = X(0); pt.y = X(1); pt.z = X(2);
    marker.points.push_back( pt );
}

void RosMarkerUtils::add_point_to_marker( const Vector4d& X, visualization_msgs::Marker& marker, bool clear_prev_points )
{
    if( clear_prev_points )
        marker.points.clear();

    geometry_msgs::Point pt;
    assert( abs(X(3)) > 1E-5 );
    pt.x = X(0)/X(3); pt.y = X(1)/X(3); pt.z = X(2)/X(3);
    marker.points.push_back( pt );
}

void RosMarkerUtils::add_points_to_marker( const MatrixXd& X, visualization_msgs::Marker& marker, bool clear_prev_points ) //X : 3xN or 4xN.
{
    assert( (X.rows() == 3 || X.rows() == 4) && "[RosMarkerUtils::add_points_to_marker] X need to of size 3xN or 4xN\n" );
    geometry_msgs::Point pt;

    if( clear_prev_points )
        marker.points.clear();


    int has_nan = 0;
    for( int i=0 ; i<X.cols() ; i++ ) {
        if( std::isnan(X.col(i).sum()) || std::isnan(X.col(i).sum()) )
        {
            // cout << "[RosMarkerUtils::add_points_to_marker] WARN found Nan or Inf at col#" << i << endl;
            has_nan++;
            continue;
        }

        if( X.rows() == 3 ) {
            pt.x = X(0,i); pt.y = X(1,i); pt.z = X(2,i);
        }
        else {
            pt.x = X(0,i) / X(3,i); pt.y = X(1,i) / X(3,i); pt.z = X(2,i) / X(3,i);
        }
        marker.points.push_back( pt );
    }

    if( has_nan > 0 )
    cout << "[RosMarkerUtils::add_points_to_marker] found nan for " << has_nan << " out of total " << X.cols() << endl;
}



void RosMarkerUtils::add_colors_to_marker( const Vector3d& color_rgb, visualization_msgs::Marker& marker, bool clear_prev_colors )
{
    // assert( (X.rows() == 3) && "[RosMarkerUtils::add_colors_to_marker] X need to of size 3xN representing rgb colors of the points\n" );
    // geometry_msgs::Point pt;
    std_msgs::ColorRGBA pt_color;

    if( clear_prev_colors )
        marker.colors.clear();



        pt_color.r = color_rgb(0); pt_color.g = color_rgb(1); pt_color.b = color_rgb(2); pt_color.a = 1.0;
        marker.colors.push_back( pt_color );

}

void RosMarkerUtils::add_colors_to_marker( float c_r, float c_g, float c_b, visualization_msgs::Marker& marker, bool clear_prev_colors )
{
    // assert( (X.rows() == 3) && "[RosMarkerUtils::add_colors_to_marker] X need to of size 3xN representing rgb colors of the points\n" );
    // geometry_msgs::Point pt;
    std_msgs::ColorRGBA pt_color;

    if( clear_prev_colors )
        marker.colors.clear();



        pt_color.r = c_r; pt_color.g = c_g; pt_color.b = c_b; pt_color.a = 1.0;
        marker.colors.push_back( pt_color );

}



void RosMarkerUtils::add_colors_to_marker( const MatrixXd& X, visualization_msgs::Marker& marker, bool clear_prev_colors )
{
    assert( (X.rows() == 3) && "[RosMarkerUtils::add_colors_to_marker] X need to of size 3xN representing rgb colors of the points\n" );
    // geometry_msgs::Point pt;
    std_msgs::ColorRGBA pt_color;

    if( clear_prev_colors )
        marker.colors.clear();


    for( int i=0 ; i<X.cols() ; i++ ) {
        pt_color.r = X(0,i); pt_color.g = X(1,i); pt_color.b = X(2,i); pt_color.a = 1.0;
        marker.colors.push_back( pt_color );
    }
}


void RosPublishUtils::publish_3d( ros::Publisher& pub, MatrixXd& _3dpts,
    string ns, int id,
    float red, float green, float blue, float alpha,
    int size_multiplier )
{
    assert( size_multiplier > 0 );
    assert( alpha>0 && alpha<= 1.0 );

    visualization_msgs::Marker local_3dpt;
    RosMarkerUtils::init_points_marker( local_3dpt );
    local_3dpt.ns = ns;
    local_3dpt.id = id;
    local_3dpt.scale.x = 0.02*size_multiplier;
    local_3dpt.scale.y = 0.02*size_multiplier;

    RosMarkerUtils::add_points_to_marker( _3dpts, local_3dpt, true );
    RosMarkerUtils::setcolor_to_marker( red/255., green/255., blue/255., alpha, local_3dpt );

    pub.publish( local_3dpt );
}



// This will colorcode the 3dpoints by x `colorcode_by_dim=0`, [min, max] will get the false color.
void RosPublishUtils::publish_3d( ros::Publisher& pub, MatrixXd& _3dpts,
    string ns, int id,
    int colorcode_by_dim, double min_val, double max_val,
    int size_multiplier )
{
    assert( size_multiplier > 0 );
    assert( colorcode_by_dim>=0 && colorcode_by_dim < _3dpts.rows() );

    visualization_msgs::Marker local_3dpt;
    RosMarkerUtils::init_points_marker( local_3dpt );
    local_3dpt.ns = ns;
    local_3dpt.id = id;
    local_3dpt.scale.x = 0.02*size_multiplier;
    local_3dpt.scale.y = 0.02*size_multiplier;

    RosMarkerUtils::add_points_to_marker( _3dpts, local_3dpt, true );

    #if 0
    // RosMarkerUtils::setcolor_to_marker( red/255., green/255., blue/255., .8, local_3dpt );
    #else
    FalseColors co;
    for( int i=0 ; i<_3dpts.cols() ; i++ ) //loop over all the 3d points to determine the color
    {
        float f = ( _3dpts(colorcode_by_dim,i) - min_val ) / (max_val - min_val);
        cv::Scalar col= co.getFalseColor( float( _3dpts(2,i)/10. ) );
        RosMarkerUtils::add_colors_to_marker(  col[2]/255., col[1]/255., col[0]/255., local_3dpt, (i==0)?true:false );

    }
    #endif

    pub.publish( local_3dpt );
}
