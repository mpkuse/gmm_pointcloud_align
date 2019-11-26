#include "EdgeAlignment.h"

void EdgeAlignment::solve( Matrix4d& initial_guess____ref_T_curr )
{

    #if 1
    //----- distance transform will be made with edgemap of reference image
    ElapsedTime t_distanceTransform( "Distance Transform");
    cv::Mat disTrans, edge_map;
    get_distance_transform( im_ref, disTrans, edge_map );
    Eigen::MatrixXd e_disTrans;
    cv::cv2eigen( disTrans, e_disTrans );
    cout << TermColor::uGREEN() <<  t_distanceTransform.toc() << endl << TermColor::RESET();
    cv::imshow("Distance Transform of ref image", disTrans); //numbers between 0 and 1.
    cv::imshow( "edge_map of ref image", edge_map );
    #endif


     //---- 3d points will be made from curr. Only at edges
     ElapsedTime t_3dpts( "3D edge-points of curr image" );
     MatrixXd cX = get_cX_at_edge_pts( im_curr, depth_curr ); //cX will be 4xN
     cout << TermColor::uGREEN() << t_3dpts.toc() << TermColor::RESET() << endl;;

     cout << "cX(1st 10 cols)\n" << cX.leftCols(10) << endl;

     //  use the initial guess and try projecting these points on im_ref
     cout << "Initial Guess : " << PoseManipUtils::prettyprintMatrix4d( initial_guess____ref_T_curr ) << endl;
     MatrixXd ref_uv = reproject( cX, initial_guess____ref_T_curr );
     cv::Mat dst;
     MiscUtils::plot_point_sets( im_ref, ref_uv, dst, cv::Scalar(0,0,255), false, "initial" );
     cv::imshow( "reprojecting 3d pts of curr on ref using initial guess of rel-pose", dst );



     //---
     //---- Setup the optimization problem
     //---
    cout << "e_disTrans.shape = " << e_disTrans.rows() << ", " << e_disTrans.cols() << endl;
    ceres::Grid2D<double,1> grid( e_disTrans.data(), 0, e_disTrans.cols(), 0, e_disTrans.rows() );
    ceres::BiCubicInterpolator< ceres::Grid2D<double,1> > interpolated_imb_disTrans( grid );




    //---- Solve

    // opt var
    Eigen::Matrix4d ref_T_curr_optvar = initial_guess____ref_T_curr; //Eigen::Matrix4d::Identity();
    double ref_quat_curr[10], ref_t_curr[10];
    PoseManipUtils::eigenmat_to_raw( ref_T_curr_optvar, ref_quat_curr, ref_t_curr );



    // Residues for each 3d points
    ceres::Problem problem;

    std::vector<double> parameterVec;
    cam->writeParameters( parameterVec );
    double fx=parameterVec.at(4);
    double fy=parameterVec.at(5);
    double cx=parameterVec.at(6);
    double cy=parameterVec.at(7);
    cout << "fx=" << fx << "\t";
    cout << "fy=" << fy << "\t";
    cout << "cx=" << cx << "\t";
    cout << "cy=" << cy << "\n";

    auto robust_loss = new ceres::CauchyLoss(1.);
    for( int i=0 ; i< cX.cols() ; i+=30 )
    {
        // ceres::CostFunction * cost_function = EAResidue::Create( K, a_X.col(i), interpolated_imb_disTrans);

        ceres::CostFunction * cost_function = EAResidue::Create( fx,fy,cx,cy, cX(0,i),cX(1,i),cX(2,i), interpolated_imb_disTrans);
        problem.AddResidualBlock( cost_function, robust_loss, ref_quat_curr, ref_t_curr );
    }

    ceres::LocalParameterization * quaternion_parameterization = new ceres::QuaternionParameterization;
    problem.SetParameterization( ref_quat_curr, quaternion_parameterization );

    // Run
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.linear_solver_type = ceres::DENSE_QR;
    ceres::Solver::Summary summary;
    ceres::Solve( options, &problem, &summary );
    std::cout << summary.FullReport() << "\n";


    PoseManipUtils::raw_to_eigenmat( ref_quat_curr, ref_t_curr, ref_T_curr_optvar );
    cout << "Initial Guess : " << PoseManipUtils::prettyprintMatrix4d( initial_guess____ref_T_curr ) << endl;
    cout << "Final Guess : " << PoseManipUtils::prettyprintMatrix4d( ref_T_curr_optvar ) << endl;
    MatrixXd ref_uv_final = reproject( cX, ref_T_curr_optvar );
    cv::Mat dst_final;
    MiscUtils::plot_point_sets( im_ref, ref_uv_final, dst_final, cv::Scalar(0,0,255), false, "final" );
    cv::imshow( "dst_final", dst_final );



}



//utils

// #define get_distance_transform_debug(msg) msg;
#define get_distance_transform_debug(msg);
void EdgeAlignment::get_distance_transform( const cv::Mat& input, cv::Mat& out_distance_transform, cv::Mat& out_edge_map )
{
    // Thresholds that influcence this:
    // a. Gaussian Blur size
    // b. Threshold for the gradient map. How you compute gradient also matter. here i m using Laplacian operator. Results will differ with Sobel for example.
    // c. Window size for median blur
    // d. Params for distance transform computation.

    //
    // Edge Map
    //

    cv::Mat _blur, _gray;
    cv::GaussianBlur( input, _blur, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
    if( _blur.channels() == 1 )
        _gray = _blur;
    else
        cv::cvtColor( _blur, _gray, CV_RGB2GRAY );

    #if 0 // Laplacian
    cv::Mat _laplacian, _laplacian_8uc1;
    cv::Laplacian( _gray, _laplacian, CV_16S, 3, 1, 0, cv::BORDER_DEFAULT );
    cv::convertScaleAbs( _laplacian, _laplacian_8uc1 );
    get_distance_transform_debug( cv::imshow( "_laplacian_8uc1", _laplacian_8uc1) );
    out_edge_map = _laplacian_8uc1;

    //
    // Threshold gradients
    // TODO - use cv::Threshold
    cv::Mat B = cv::Mat::ones( _laplacian.rows, _laplacian.cols, CV_8UC1 ) * 255;
    for( int v=0 ; v<_laplacian.rows ; v++ )
    {
        for( int u=0 ; u<_laplacian.cols ; u++ )
        {
            if( _laplacian_8uc1.at<uchar>(v,u) > 25 )
            {
                B.at<uchar>(v,u) = 0;
            }
        }
    }

    //
    // Suppress noise with median filter
    cv::Mat B_filtered;
    cv::medianBlur( B, B_filtered, 3 );
    get_distance_transform_debug( cv::imshow( "edge map", B_filtered ) );
    #endif


    #if 1 // Canny
    cv::Mat B_filtered;

    cv::Mat dst, detected_edges;
    // int edgeThresh = 1;
    int lowThreshold=30;
    int ratio = 3;
    int kernel_size = 3;
    cv::Canny( _blur, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
    dst = cv::Scalar::all(0);

    _blur.copyTo( dst, detected_edges);
    out_edge_map = dst;


    get_distance_transform_debug( cv::imshow( "edge map", dst ); )
    double min_dst, max_dst;
    get_distance_transform_debug(
    cv::minMaxLoc(dst, &min_dst, &max_dst);
    cout << "dst: " << MiscUtils::cvmat_info( dst ) << endl;
    cout << "dst_min=" << min_dst << "\tdst_max=" << max_dst << endl; )

    // B_filtered = 255 - dst;
    cv::threshold( dst, B_filtered, 10, 255, cv::THRESH_BINARY_INV );
    get_distance_transform_debug( cv::imshow( "B_filtered edge map", B_filtered ) );

    #endif


    //
    // Distance Transform
    //
    cv::Mat dist;
    cv::distanceTransform(B_filtered, dist, cv::DIST_L2, 3);
    cv::normalize(dist, dist, 0, 1., cv::NORM_MINMAX);
    get_distance_transform_debug( cout << "dist : " << MiscUtils::cvmat_info(dist ) << endl; )
    get_distance_transform_debug( imshow("Distance Transform Image", dist) );

    out_distance_transform = dist;

}



// #define __EdgeAlignment__get_cX_at_edge_pts( msg ) msg;
#define __EdgeAlignment__get_cX_at_edge_pts( msg ) ;
MatrixXd EdgeAlignment::get_cX_at_edge_pts( const cv::Mat im, const cv::Mat depth_map   )
{
    assert( !im.empty() && !depth_map.empty() );
    assert( im.rows == depth_map.rows && im.cols == depth_map.cols );
    assert( depth_map.type() == CV_16UC1 || depth_map.type() == CV_32FC1 );
    assert( cam );

    //---- get edgemap with canny
    cv::Mat _blur, _gray;
    cv::GaussianBlur( im, _blur, cv::Size(3,3), 0, 0, cv::BORDER_DEFAULT );
    if( _blur.channels() == 1 )
        _gray = _blur;
    else
        cv::cvtColor( _blur, _gray, CV_RGB2GRAY );


    cv::Mat dst, detected_edges;
    // int edgeThresh = 1;
    int lowThreshold=30;
    int ratio = 3;
    int kernel_size = 3;
    cv::Canny( _blur, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );
    dst = cv::Scalar::all(0);

    _blur.copyTo( dst, detected_edges);

    __EdgeAlignment__get_cX_at_edge_pts( cv::imshow( "edge map get_cX", dst ); )



    //---- loop over all the pixels and process only at edge points
    vector<cv::Point3f> vec_of_pt;
    for( int v=0 ; v< im.rows ; v++ )
    {
        for( int u=0 ; u<im.cols ; u++ )
        {
            float depth_val;
            if( depth_map.type() == CV_16UC1 ) {
                depth_val = .001 * depth_map.at<uint16_t>( v, u );
            }
            else if( depth_map.type() == CV_32FC1 ) {
                // just assuming the depth values are in meters when CV_32FC1
                depth_val = depth_map.at<float>(v, u );
            }
            else {
                cout << "[EdgeAlignment::get_cX_at_edge_pts]depth type is neighter of CV_16UC1 or CV_32FC1\n";
                throw "[EdgeAlignment::get_cX_at_edge_pts]depth type is neighter of CV_16UC1 or CV_32FC1\n";
            }
            // cout << "at u=" << u << ", v=" << v << "\tdepth_val = " << depth_val << endl;


            if( dst.at<uchar>(v,u) < 10 || depth_val < 0.5 || depth_val > 5. )
                continue;


            Vector3d _1P;
            Vector2d _1p; _1p << u, v;
            cam->liftProjective( _1p, _1P );

            cv::Point3f pt;
            pt.x = depth_val * _1P(0);
            pt.y = depth_val * _1P(1);
            pt.z = depth_val;

            vec_of_pt.push_back( pt );
        }
    }

    MatrixXd cX;
    MiscUtils::point3f_2_eigen( vec_of_pt, cX );
    __EdgeAlignment__get_cX_at_edge_pts(
    cout << "cX.shape=" << cX.rows() << "x" << cX.cols() << endl;
    cout << "vec_of_pt.size() = " << vec_of_pt.size() << endl; )
    return cX;

}





Eigen::MatrixXd EdgeAlignment::reproject( const Eigen::MatrixXd& a_X, const Eigen::Matrix4d& b_T_a )
{
    assert( cam );
    assert( a_X.rows() == 4 && a_X.cols() > 0 );
    Eigen::MatrixXd b_X = b_T_a * a_X;

    MatrixXd uv = MatrixXd::Constant( 3, b_X.cols(), 1.0 );
    for( int i=0 ; i<b_X.cols() ; i++ )
    {

        Vector2d p_dst;
        cam->spaceToPlane( b_X.col(i).topRows(3), p_dst  );

        uv.col(i).topRows(2) = p_dst;
    }
    return uv;


}
