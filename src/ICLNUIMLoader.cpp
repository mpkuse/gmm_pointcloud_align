#include "ICLNUIMLoader.h"

ICLNUIMLoader::ICLNUIMLoader( const string DB_BASE, const string DB_NAME, const int DB_IDX ):
    DB_BASE(DB_BASE), DB_NAME(DB_NAME), DB_IDX(DB_IDX)
{
    cout << "DB_BASE=" << DB_BASE << endl;
    cout << "DB_NAME=" << DB_NAME << endl;
    cout << "DB_IDX=" << DB_IDX << endl;


    cout << "is_path_a_directory(" << DB_BASE << ")?\n";
    if(  RawFileIO::is_path_a_directory( DB_BASE ) == false )
    {
        cout << TermColor::RED() <<"[ICLNUIMLoader::ICLNUIMLoader]DB_BASE=" << DB_BASE << " is not accessible. This could be because, so such directory or you do not have permission to access it\n" << TermColor::RESET();
        exit(2);
    }
    cout << "\t...OK!\n";

    cout << "is_path_a_directory(" << DB_BASE+"/"+DB_NAME << ")?\n";
    if( RawFileIO::is_path_a_directory( DB_BASE+"/"+DB_NAME ) == false )
    {
        cout << TermColor::RED() << "[ICLNUIMLoader::ICLNUIMLoader]No such sequence DB_NAME=" << DB_NAME << TermColor::RESET() << endl;
        exit(2);
    }
    cout << "\t...OK!\n";

    img_fld = DB_BASE+"/"+DB_NAME+"/"+DB_NAME+"_traj"+to_string( DB_IDX )+"_loop/";
    traj_file = DB_BASE+"/"+DB_NAME+"/"+DB_NAME+"_traj"+to_string( DB_IDX )+".gt.freiburg";



    cout << "is_path_a_directory(" << img_fld << ")?\n";
    if( RawFileIO::is_path_a_directory(img_fld) == false )
    {
        cout << TermColor::RED() << "Image Folder = " << img_fld << " doesnt exist\n" << TermColor::RESET();
        exit(2);
    }
    cout << "\t...OK!\n";

    #if 1
    cout << "if_file_exist(" << traj_file << ")?\n";
    if( RawFileIO::if_file_exist_2(traj_file.c_str()) == false )
    {
        cout << TermColor::RED() << "The trajectory file = `" << traj_file << "` doesnt exist\n" << TermColor::RESET();
        exit(2);
    }
    cout << "\t...OK!\n";
    #endif

    // start and end image
    int i =0;
    IM_LIST.clear();
    DEPTH_LIST.clear();
    for(  ; ; i++ )
    {
        std::ostringstream ss1, ss2;
        if( DB_NAME == "living_room" || (DB_NAME == "office_room" &&  DB_IDX == 0 ) ) {
        ss1 << img_fld << "/scene_" << std::setw(2) << std::setfill('0') << 0 << "_" << std::setw(4) << std::setfill('0')  << i << ".png";
        ss2 << img_fld << "/scene_" << std::setw(2) << std::setfill('0') << 0 << "_" << std::setw(4) << std::setfill('0')  << i << ".depth";
    } else if( DB_NAME == "office_room" && DB_IDX >=1 && DB_IDX <= 3 ) {
        ss1 << img_fld << "/scene_" << std::setw(3) << std::setfill('0')  << i << ".png";
        ss2 << img_fld << "/scene_" << std::setw(3) << std::setfill('0')  << i << ".depth";
        }
        else assert( false );


        // cout << "Check if_fie " << ss1.str() << endl;
        if( RawFileIO::if_file_exist_2( ss1.str() ) == false ||
            RawFileIO::if_file_exist_2( ss2.str() ) == false
        )
        {
            // cout << "NO!\n";
            break;
        }

        IM_LIST.push_back( ss1.str() );
        DEPTH_LIST.push_back( ss2.str() );
        // string fname_im = img_fld+"/";
        // string fname_depth = "";
    }
    cout << TermColor::GREEN() << "Image: start=0 ; end=" << IM_LIST.size() << TermColor::RESET()<<  endl;


    // read the traj file
    MatrixXd TRAJ;
    bool status = RawFileIO::read_eigen_matrix( traj_file, TRAJ );
    assert( status );
    // cout << "TRAJ: " << TRAJ.rows() << "x" << TRAJ.cols() << endl;

    assert( TRAJ.cols() == 8 );
    for( int i=0 ; i<TRAJ.rows() ; i++ )
    {
        //  'timestamp tx ty tz qx qy qz qw'

        double tr_xyz[5], quat_xyzw[5];
        tr_xyz[0] = TRAJ(i,1);
        tr_xyz[1] = TRAJ(i,2);
        tr_xyz[2] = TRAJ(i,3);

        quat_xyzw[0] = TRAJ(i,4);
        quat_xyzw[1] = TRAJ(i,5);
        quat_xyzw[2] = TRAJ(i,6);
        quat_xyzw[3] = TRAJ(i,7);

        Matrix4d wTc;
        PoseManipUtils::raw_xyzw_to_eigenmat(quat_xyzw, tr_xyz, wTc );
        POSE_LIST.push_back( wTc );
    }
    // print_info();
    cout << "---Constructor OK!, len=" << len() << "\n";

}

void ICLNUIMLoader::print_info( ) const
{
    cout << "DB_BASE=" << DB_BASE << endl;
    cout << "DB_NAME=" << DB_NAME << endl;
    cout << "DB_IDX=" << DB_IDX << endl;
    cout << "IM_LIST.size=" << IM_LIST.size() << endl;
    cout << "POSE_LIST.size=" << POSE_LIST.size() << endl;
}


bool ICLNUIMLoader::retrive_im( int i, cv::Mat& im )
{
    // cout << TermColor::BLUE() << "retrive_im" << TermColor::RESET() << endl;
    if( !( i >=0 && i<len() ) ) {
        cout << TermColor::RED() << "[ICLNUIMLoader::retrive_im]invalid i\n" << TermColor::RESET();
        return false;
    }
    assert( i >=0 && i<len() );

    im = cv::imread( IM_LIST[i] );
    return true;
}

bool ICLNUIMLoader::retrive_im_depth( int i, cv::Mat& im, cv::Mat& depth )
{
    assert( i >=0 && i<len() );
    if( !( i >=0 && i<len() ) ) {
        cout << TermColor::RED() << "[ICLNUIMLoader::retrive_im_depth]invalid i\n" << TermColor::RESET();
        return false;
    }

    im = cv::imread( IM_LIST[i] );

    // TODO depth
    depth = cv::Mat::zeros( im.rows, im.cols, CV_32FC1 );
    imread_depth( DEPTH_LIST[i], depth );

    return true;
}

bool ICLNUIMLoader::retrive_im_depth( int i, cv::Mat& im, cv::Mat& depth, cv::Mat& depth_falsecolor )
{
    assert( i >=0 && i<len() );
    if( !( i >=0 && i<len() ) ) {
        cout << TermColor::RED() << "[ICLNUIMLoader::retrive_im_depth]invalid i\n" << TermColor::RESET();
        return false;
    }

    im = cv::imread( IM_LIST[i] );

    // depth
    depth = cv::Mat::zeros( im.rows, im.cols, CV_32FC1 );
    imread_depth( DEPTH_LIST[i], depth );

    // depth false colormap
    // depth_falsecolor = cv::Mat::zeros( im.rows, im.cols, CV_8UC3 );
    cv::Mat depth_falsecolor_gray = cv::Mat::zeros( im.rows, im.cols, CV_8UC1 );
    double d_min, d_max;
    cv::minMaxLoc(depth, &d_min, &d_max);
    // cout << "d_min=" << d_min << "\td_max=" << d_max << endl;
    for( int r=0; r<depth.rows ; r++ )
    {
        for( int c=0 ; c<depth.cols ; c++ )
        {
            depth_falsecolor_gray.at<uint8_t>(r,c) = (uint8_t)( 250. * depth.at<float>(r,c) );
        }
    }
    cv::applyColorMap(depth_falsecolor_gray, depth_falsecolor, cv::COLORMAP_HOT );

    return true;
}

bool ICLNUIMLoader::imread_depth( string fname, cv::Mat& depth )
{
    assert( depth.rows > 0 && depth.cols > 0 );
    ifstream file;
    file.open (fname);
    if (!file.is_open()) {
        cout << "[ICLNUIMLoader::imread_depth]ERROR, cannot open depth file\n";
        return false;
    }

    string word;
    int c = 0;
    while (file >> word)
    {
        float f = std::stof( word );
        if( DB_NAME == "office_room" )
            f /= 1000.; //for `office_room` depth values are in mm and for `living_room` depth values are in meters
        int j = c % depth.cols;
        int i = c / depth.cols;

        depth.at<float>( i,j ) = f;
        c++;
    }

    assert( c == depth.rows * depth.cols );
    // cout << "ICLNUIMLoader::imread_depth loaded " << c << " floats\n";
    return true;
}

bool ICLNUIMLoader::retrive_pose( int i, Matrix4d& wTc )
{
    assert( i >=0 && i<len() );
    if( !( i >=0 && i<len() ) ) {
        cout << TermColor::RED() << "[ICLNUIMLoader::retrive_pose]invalid i\n" << TermColor::RESET();
        return false;
    }

    wTc = POSE_LIST[i];
    return true;
}
