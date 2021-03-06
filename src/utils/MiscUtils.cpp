#include "MiscUtils.h"

string MiscUtils::type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

string MiscUtils::cvmat_info( const cv::Mat& mat )
{
    std::stringstream buffer;
    buffer << "shape=" << mat.rows << "," << mat.cols << "," << mat.channels() ;
    buffer << "\t" << "dtype=" << MiscUtils::type2str( mat.type() );
    return buffer.str();
}

string MiscUtils::cvmat_minmax_info( const cv::Mat& mat )
{
    double min_dst, max_dst;
    cv::minMaxLoc(mat, &min_dst, &max_dst);

    std::stringstream buffer;
    buffer << "(min,max)=(" << min_dst << "," << max_dst << ")";
    return buffer.str();
}

string MiscUtils::imgmsg_info(const sensor_msgs::ImageConstPtr &img_msg)
{
    std::stringstream buffer;
    buffer << "---\n";
    buffer << "\theader:\n";
    buffer << "\t\tseq: " << img_msg->header.seq << endl;
    buffer << "\t\tstamp: " << img_msg->header.stamp << endl;
    buffer << "\t\tframe_id: " << img_msg->header.frame_id << endl;
    buffer << "\twidth:" << img_msg->width << endl;
    buffer << "\theight:" << img_msg->height << endl;
    buffer << "\tencoding:" << img_msg->encoding << endl;
    buffer << "\tis_bigendian:" << (int) img_msg->is_bigendian << endl;
    buffer << "\tstep:" << img_msg->step << endl;
    buffer << "\tdata:" << "[..." << img_msg->step * img_msg->height << " sized array...]" << endl;

    return buffer.str();

}

string MiscUtils::imgmsg_info(const sensor_msgs::Image& img_msg)
{
    std::stringstream buffer;
    buffer << "---\n";
    buffer << "\theader:\n";
    buffer << "\t\tseq: " << img_msg.header.seq << endl;
    buffer << "\t\tstamp: " << img_msg.header.stamp << endl;
    buffer << "\t\tframe_id: " << img_msg.header.frame_id << endl;
    buffer << "\twidth:" << img_msg.width << endl;
    buffer << "\theight:" << img_msg.height << endl;
    buffer << "\tencoding:" << img_msg.encoding << endl;
    buffer << "\tis_bigendian:" << (int) img_msg.is_bigendian << endl;
    buffer << "\tstep:" << img_msg.step << endl;
    buffer << "\tdata:" << "[..." << img_msg.step * img_msg.height << " sized array...]" << endl;

    return buffer.str();

}

cv::Mat MiscUtils::getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::MONO8);

    cv::Mat img = ptr->image.clone();
    return img;
}

std::vector<std::string>
MiscUtils::split( std::string const& original, char separator )
{
    std::vector<std::string> results;
    std::string::const_iterator start = original.begin();
    std::string::const_iterator end = original.end();
    std::string::const_iterator next = std::find( start, end, separator );
    while ( next != end ) {
        results.push_back( std::string( start, next ) );
        start = next + 1;
        next = std::find( start, end, separator );
    }
    results.push_back( std::string( start, next ) );
    return results;
}

void MiscUtils::keypoint_2_eigen( const std::vector<cv::KeyPoint>& kp, MatrixXd& uv, bool make_homogeneous )
{
    assert( kp.size() > 0 && "MiscUtils::keypoint_2_eigen empty kp.");
    uv = MatrixXd::Constant( (make_homogeneous?3:2), kp.size(), 1.0 );
    for( int i=0; i<kp.size() ; i++ )
    {
        uv(0,i) = kp[i].pt.x;
        uv(1,i) = kp[i].pt.y;
    }
}

void MiscUtils::dmatch_2_eigen( const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2,
                            const std::vector<cv::DMatch> matches,
                            MatrixXd& M1, MatrixXd& M2,
                            bool make_homogeneous
                        )
{
    assert( matches.size() > 0 && "MiscUtils::dmatch_2_eigen DMatch cannot be empty" );
    assert( kp1.size() > 0 && kp2.size() > 0 && "MiscUtils::dmatch_2_eigen keypoints cannot be empty" );

    M1 = MatrixXd::Constant( (make_homogeneous?3:2), matches.size(), 1.0 );
    M2 = MatrixXd::Constant( (make_homogeneous?3:2), matches.size(), 1.0 );
    for( int i=0 ; i<matches.size() ; i++ ) {
        int queryIdx = matches[i].queryIdx; //kp1
        int trainIdx = matches[i].trainIdx; //kp2
        assert( queryIdx >=0 && queryIdx < kp1.size() );
        assert( trainIdx >=0 && trainIdx < kp2.size() );
        M1(0,i) = kp1[queryIdx].pt.x;
        M1(1,i) = kp1[queryIdx].pt.y;

        M2(0,i) = kp2[trainIdx].pt.x;
        M2(1,i) = kp2[trainIdx].pt.y;
    }
}


void MiscUtils::point2f_2_eigen( const std::vector<cv::Point2f>& p, MatrixXd& dst, bool make_homogeneous )
{
    assert( p.size() > 0 && "[MiscUtils::point2f_2_eigen] input point vector looks empty\n");
    // cout << "p.size=" << p.size() << endl;
    dst = MatrixXd::Constant( (make_homogeneous?3:2), p.size(), 1.0 );

    for( int i=0 ;i<p.size() ; i++ )
    {
        dst( 0, i ) = p[i].x;
        dst( 1, i ) = p[i].y;
    }

}

void MiscUtils::eigen_2_point2f( const MatrixXd& inp, std::vector<cv::Point2f>& p )
{
    assert( inp.rows() == 2 || inp.rows() == 3 );
    assert( inp.cols() > 0 );
    p.clear();

    bool homogeneous = false;
    if( inp.rows() == 3 )
    {
        homogeneous = true;
    }

    for( int i=0 ; i<inp.cols() ; i++ )
    {
        cv::Point2f pt;
        pt.x = (float) inp(0,i);
        pt.y = (float) inp(1,i);
        if( homogeneous == true ) {
            assert( abs(inp(2,i))>1e-7 ); //z cannot be zero
            pt.x /= (float) inp(2,i);
            pt.y /= (float) inp(2,i);
        }
        p.push_back(pt);
    }

}


void MiscUtils::point3f_2_eigen( const std::vector<cv::Point3f>& p, MatrixXd& dst, bool make_homogeneous )
{
    assert( p.size() > 0 && "[MiscUtils::point3f_2_eigen] input point vector looks empty\n");
    // cout << "p.size=" << p.size() << endl;
    dst = MatrixXd::Constant( (make_homogeneous?4:3), p.size(), 1.0 );

    for( int i=0 ;i<p.size() ; i++ )
    {
        dst( 0, i ) = p[i].x;
        dst( 1, i ) = p[i].y;
        dst( 2, i ) = p[i].z;
    }

}

void MiscUtils::eigen_2_point3f( const MatrixXd& inp, std::vector<cv::Point3f>& p )
{
    assert( inp.rows() == 3 || inp.rows() == 4 );
    assert( inp.cols() > 0 );
    p.clear();

    bool homogeneous = false;
    if( inp.rows() == 3 )
    {
        homogeneous = true;
    }

    for( int i=0 ; i<inp.cols() ; i++ )
    {
        cv::Point3f pt;
        pt.x = (float) inp(0,i);
        pt.y = (float) inp(1,i);
        pt.z = (float) inp(2,i);
        #if 0
        if( homogeneous == true ) {
            assert( abs(inp(3,i))>1e-7 ); //z cannot be zero
            pt.x /= (float) inp(3,i);
            pt.y /= (float) inp(3,i);
            pt.z /= (float) inp(3,i);
        }
        #endif
        p.push_back(pt);
    }

}

int MiscUtils::total_true( const vector<bool>& V )
{
    int s=0;
    for( int i=0 ; i<(int)V.size() ; i++ )
        if( V[i] == true )
            s++;

    return s;
}

int MiscUtils::total_positives( const vector<uchar>& V )
{
    int s=0;
    for( int i=0 ; i<(int)V.size() ; i++ )
        if( V[i] > (uchar) 0 )
            s++;

    return s;
}


vector<bool> MiscUtils::filter_near_far( const VectorXd& dd, double near, double far )
{
    vector<bool> valids;
    valids.clear();
    for( int i=0 ; i<dd.size() ; i++ )
    {
        if( dd(i) > near && dd(i) < far )
            valids.push_back(true);
        else
            valids.push_back(false);

    }
    return valids;
}

vector<bool> MiscUtils::vector_of_bool_AND( const vector<bool>& A, const vector<bool>& B )
{
    assert( A.size() == B.size() );
    int n = (int) A.size();
    vector<bool> valids;
    valids.clear();
    for( int i=0 ; i<n ; i++ )
    {
        if( A[i] && B[i] )
            valids.push_back( true );
        else
            valids.push_back( false );
    }
    return valids;
}



vector<uchar> MiscUtils::vector_of_uchar_AND( const vector<uchar>& A, const vector<uchar>& B )
{
    assert( A.size() == B.size() );
    int n = (int) A.size();
    vector<uchar> valids;
    valids.clear();
    for( int i=0 ; i<n ; i++ )
    {
        if( A[i] > 0 && B[i] > 0  )
            valids.push_back( A[i] );
        else
            valids.push_back( (uchar)0 );
    }
    return valids;
}

VectorXd MiscUtils::to_eigen( const vector<uchar>& V )
{
    assert( V.size() > 0 );
    VectorXd out = VectorXd::Zero( (int) V.size() );
    for( int i=0 ; i<(int)V.size() ; i++ )
        out(i) = (double) V[i];

    return out;
}

void MiscUtils::reduce_vector(vector<cv::Point2f> &v, const vector<uchar> status) //inplace
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
       if (status[i])
           v[j++] = v[i];
   v.resize(j);
}

void MiscUtils::reduce_vector(const vector<cv::Point2f> &v, const vector<uchar> status, vector<cv::Point2f>& out )
{
    out.clear();
    for (int i = 0; i < int(v.size()); i++)
       if (status[i])
           out.push_back( v[i] );

}


void MiscUtils::imshow( const string& win_name, const cv::Mat& im, float scale )
{
    if( scale == 1.0 ) {
        cv::imshow( win_name, im );
        return;
    }

    assert( scale > 0.1 && scale < 10.0 );
    cv::Mat outImg;
    cv::resize(im, outImg, cv::Size(), scale, scale );
    cv::imshow( win_name, outImg );

}

// #define __MiscUtils___gather( msg ) msg;
#define __MiscUtils___gather( msg ) ;
void MiscUtils::gather( const vector<MatrixXd>& mats, const vector<  vector<bool> >& valids, MatrixXd& dst )
{
    __MiscUtils___gather(
    cout << "-----------MiscUtils::gather()\n";

    cout << "mats.size = " << mats.size() << "\tvalids.size=" << valids.size() << endl;)
    assert( mats.size() == valids.size() && mats.size() > 0 );
    int ntotalvalids = 0;
    for( int i=0 ; i<(int)mats.size() ; i++ )
    {
        int nvalids = 0;
        for( int j=0 ; j<(int)valids[i].size() ; j++  )
        {
            if( valids[i][j] == true )
                nvalids++;
        }
        ntotalvalids += nvalids;

        __MiscUtils___gather(
        cout << "i=" << i << "\tmats[i]=" << mats[i].rows() << "x" << mats[i].cols() << "\t";
        cout << "valids.size=" << valids[i].size() << " " << "nvalids=" << nvalids << endl;)
    }
    __MiscUtils___gather( cout << "ntotalvalids=" << ntotalvalids << endl; )


    dst = MatrixXd::Zero( mats[0].rows() , ntotalvalids );
    int c = 0;
    for( int i=0 ; i<(int)mats.size() ; i++ )
    {
        for( int j=0 ; j<(int)valids[i].size() ; j++  )
        {
            if( valids[i][j] == true ) {
                dst.col(c) = mats[i].col(j);
                c++;
            }
        }
    }
    assert( c == ntotalvalids );

    __MiscUtils___gather( cout << "-----------END MiscUtils::gather()\n"; )
}


void MiscUtils::gather( const vector<MatrixXd>& mats, MatrixXd& dst )
{
    assert( mats.size() > 0 );
    int total_cols = 0;
    int nrows = mats[0].rows();
    for( int i=0 ; i<(int)mats.size() ; i++ )
    {
        assert( mats[i].rows() == nrows );
        total_cols += mats[i].cols();
    }

    dst = MatrixXd::Zero( nrows, total_cols );
    int c=0;
    for( int i=0 ; i<(int)mats.size() ; i++ )
    {
        for( int j=0 ; j<mats[i].cols() ; j++ )
        {
            dst.col(c) = mats[i].col(j);
            c++;
        }
    }
    assert( c == total_cols );

}


void MiscUtils::gather( const vector<VectorXd>& mats, VectorXd& dst )
{
    assert( mats.size() > 0 );
    int total = 0;
    for( int i=0 ; i<mats.size() ; i++ )
        total += mats[i].size();

    dst = VectorXd::Zero( total );
    int c = 0;
    for( int i=0 ; i<(int)mats.size() ; i++ )
    {
        for( int j=0 ; j<mats[i].size() ; j++ )
        {
            dst(c) = (mats[i])(j);
            c++;
        }
    }
    assert( c == total );
}

void MiscUtils::plot_point_sets( const cv::Mat& im, const MatrixXd& pts_set, cv::Mat& dst,
                                        const cv::Scalar& color, bool enable_keypoint_annotation, const string& msg )
{
    #if 0
  MatrixXf pts_set_float;
  pts_set_float = pts_set.cast<float>();

  cv::Mat pts_set_mat;
  cv::eigen2cv( pts_set_float, pts_set_mat );

  MiscUtils::plot_point_sets( im, pts_set_mat, dst, color, enable_keypoint_annotation, msg );
  #endif



    assert( im.rows > 0 && im.cols > 0 && "\n[MiscUtils::plot_point_sets]Image appears to be emoty. cannot plot.\n");
    // assert( pts_set.cols() > 0 && pts_set.cols() == annotations.rows() && "[MiscUtils::plot_point_sets] VectorXi annotation size must be equal to number of points. If you wish to use the default annotation ie. 0,1,...n use `true` instead. If you do not want annotation use `false`." );
    if( im.data == dst.data ) {
      //   cout << "src and dst are same\n";
        // src and dst images are same, so dont copy. just ensure it is a 3 channel image.
        assert( im.channels() == 3 && dst.channels() == 3 && "[MiscUtils::plot_point_sets]src and dst image are same physical image in memory. They need to be 3 channel." );
    }
    else {
      //   dst = cv::Mat( im.rows, im.cols, CV_8UC3 );
        if( im.channels() == 1 )
          cv::cvtColor( im, dst, cv::COLOR_GRAY2BGR );
        else
          im.copyTo(dst);
    }

    // cv::putText( dst, to_string(msg.length()), cv::Point(5,5), cv::FONT_HERSHEY_COMPLEX_SMALL, .95, cv::Scalar(0,255,255) );
    if( msg.length() > 0 ) {
      vector<std::string> msg_split;
      msg_split = MiscUtils::split( msg, ';' );
    //   cv::Scalar text_color = cv::Scalar(0,255,255);
      cv::Scalar text_color = color;
      for( int q=0 ; q<msg_split.size() ; q++ )
        cv::putText( dst, msg_split[q], cv::Point(5,20+20*q), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.5, color, 2.  );
    }


    //pts_set is 2xN
    cv::Point2d pt;
    for( int i=0 ; i<pts_set.cols() ; i++ )
    {
      // pt = cv::Point2d(pts_set.at<float>(0,i),pts_set.at<float>(1,i) );
      pt = cv::Point2d( (float)pts_set(0,i), (float)pts_set(1,i) );
      cv::circle( dst, pt, 2, color, -1 );

      if( enable_keypoint_annotation ) {
          char to_s[20];
          sprintf( to_s, "%d", i );
          cv::putText( dst, to_s, pt, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, color  );
      }


    }
}

// in place manip
void MiscUtils::plot_point_sets( cv::Mat& im, const MatrixXd& pts_set,
                                        const cv::Scalar& color, bool enable_keypoint_annotation, const string& msg )
{
    #if 0
  MatrixXf pts_set_float;
  pts_set_float = pts_set.cast<float>();

  cv::Mat pts_set_mat;
  cv::eigen2cv( pts_set_float, pts_set_mat );

  MiscUtils::plot_point_sets( im, pts_set_mat, im, color, enable_keypoint_annotation, msg );
    #endif

  plot_point_sets( im, pts_set, im, color, enable_keypoint_annotation, msg );
}

// TODO: call the next function instead of actually doing the work.
void MiscUtils::plot_point_sets( const cv::Mat& im, const cv::Mat& pts_set, cv::Mat& dst, const cv::Scalar& color, bool enable_keypoint_annotation, const string& msg )
{

  if( im.data == dst.data ) { // this is pointer comparison to know if this operation is inplace
    //   cout << "src and dst are same\n";
      // src and dst images are same, so dont copy. just ensure it is a 3 channel image.
      assert( im.rows > 0 && im.cols > 0 && "\n[MiscUtils::plot_point_sets]Image appears to be emoty. cannot plot.\n");
      assert( im.channels() == 3 && dst.channels() == 3 && "[MiscUtils::plot_point_sets]src and dst image are same physical image in memory. They need to be 3 channel." );
  }
  else {
    //   dst = cv::Mat( im.rows, im.cols, CV_8UC3 );
      if( im.channels() == 1 )
        cv::cvtColor( im, dst, cv::COLOR_GRAY2BGR );
      else
        im.copyTo(dst);
  }

  // cv::putText( dst, to_string(msg.length()), cv::Point(5,5), cv::FONT_HERSHEY_COMPLEX_SMALL, .95, cv::Scalar(0,255,255) );
  if( msg.length() > 0 ) {
    vector<std::string> msg_split;
    msg_split = MiscUtils::split( msg, ';' );
    //   cv::Scalar text_color = cv::Scalar(0,255,255);
      cv::Scalar text_color = color;
    for( int q=0 ; q<msg_split.size() ; q++ )
      cv::putText( dst, msg_split[q], cv::Point(5,20+20*q), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.5, text_color, 2.0 );
  }


  //pts_set is 2xN
  cv::Point2d pt;
  for( int i=0 ; i<pts_set.cols ; i++ )
  {
    pt = cv::Point2d(pts_set.at<float>(0,i),pts_set.at<float>(1,i) );
    cv::circle( dst, pt, 2, color, -1 );

    char to_s[20];
    sprintf( to_s, "%d", i);
    if( enable_keypoint_annotation ) {
        // cv::putText( dst, to_s, pt, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255,255,255) - color  );
        cv::putText( dst, to_s, pt, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, color  );
    }

  }
}


void MiscUtils::plot_point_sets( cv::Mat& im, const MatrixXd& pts_set,
                                        const cv::Scalar& color, const VectorXi& annotations, const string& msg )
{
    plot_point_sets( im, pts_set, im, color, annotations, msg );
}


void MiscUtils::plot_point_sets( cv::Mat& im, const MatrixXd& pts_set, cv::Mat& dst,
                                        const cv::Scalar& color, const VectorXi& annotations, const string& msg )

{
  // TODO consider addressof(a) == addressof(b)
  // dst = im.clone();


  assert( im.rows > 0 && im.cols > 0 && "\n[MiscUtils::plot_point_sets]Image appears to be emoty. cannot plot.\n");
  assert( pts_set.cols() > 0 && pts_set.cols() == annotations.rows() && "[MiscUtils::plot_point_sets] VectorXi annotation size must be equal to number of points. If you wish to use the default annotation ie. 0,1,...n use `true` instead. If you do not want annotation use `false`." );
  if( im.data == dst.data ) {
    //   cout << "src and dst are same\n";
      // src and dst images are same, so dont copy. just ensure it is a 3 channel image.
      assert( im.channels() == 3 && dst.channels() == 3 && "[MiscUtils::plot_point_sets]src and dst image are same physical image in memory. They need to be 3 channel." );
  }
  else {
    //   dst = cv::Mat( im.rows, im.cols, CV_8UC3 );
      if( im.channels() == 1 )
        cv::cvtColor( im, dst, cv::COLOR_GRAY2BGR );
      else
        im.copyTo(dst);
  }

  // cv::putText( dst, to_string(msg.length()), cv::Point(5,5), cv::FONT_HERSHEY_COMPLEX_SMALL, .95, cv::Scalar(0,255,255) );
  if( msg.length() > 0 ) {
    vector<std::string> msg_split;
    msg_split = MiscUtils::split( msg, ';' );
    for( int q=0 ; q<msg_split.size() ; q++ )
      cv::putText( dst, msg_split[q], cv::Point(5,20+20*q), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.5, cv::Scalar(0,255,255), 2.0 );
  }


  //pts_set is 2xN
  cv::Point2d pt;
  for( int i=0 ; i<pts_set.cols() ; i++ )
  {
    // pt = cv::Point2d(pts_set.at<float>(0,i),pts_set.at<float>(1,i) );
    pt = cv::Point2d( (float)pts_set(0,i), (float)pts_set(1,i) );
    cv::circle( dst, pt, 2, color, -1 );

    char to_s[20];
    sprintf( to_s, "%d", annotations(i) );
    cv::putText( dst, to_s, pt, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, color  );


  }
}


void MiscUtils::plot_point_sets( const cv::Mat& im, const MatrixXd& pts_set, cv::Mat& dst,
                                        vector<cv::Scalar>& color_annotations, float alpha, const string& msg )
{

      assert( im.rows > 0 && im.cols > 0 && "\n[MiscUtils::plot_point_sets]Image appears to be emoty. cannot plot.\n");
      assert( pts_set.cols() > 0 && pts_set.cols() == color_annotations.size() && "[MiscUtils::plot_point_sets] len of color vector must be equal to number of pts.\n" );
      if( im.data == dst.data ) {
        //   cout << "src and dst are same\n";
          // src and dst images are same, so dont copy. just ensure it is a 3 channel image.
          assert( im.channels() == 3 && dst.channels() == 3 && "[MiscUtils::plot_point_sets]src and dst image are same physical image in memory. They need to be 3 channel." );
      }
      else {
        //   dst = cv::Mat( im.rows, im.cols, CV_8UC3 );
          if( im.channels() == 1 )
            cv::cvtColor( im, dst, cv::COLOR_GRAY2BGR );
          else
            im.copyTo(dst);
      }

      // cv::putText( dst, to_string(msg.length()), cv::Point(5,5), cv::FONT_HERSHEY_COMPLEX_SMALL, .95, cv::Scalar(0,255,255) );
      if( msg.length() > 0 ) {
        vector<std::string> msg_split;
        msg_split = MiscUtils::split( msg, ';' );
        for( int q=0 ; q<msg_split.size() ; q++ )
          cv::putText( dst, msg_split[q], cv::Point(5,20+20*q), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.5, cv::Scalar(0,255,255), 2.0 );
      }


      //pts_set is 2xN
      cv::Point2d pt;
      int n_outside_image_domain = 0;
      for( int i=0 ; i<pts_set.cols() ; i++ )
      {
          if(  pts_set(0,i) < 0 || pts_set(0,i) > im.cols  ||  pts_set(1,i) < 0 || pts_set(1,i) > im.rows   ) { // avoid points which are outside
              n_outside_image_domain++;
              continue;
        }
        pt = cv::Point( (int)pts_set(0,i), (int)pts_set(1,i) );


        cv::Scalar _color = color_annotations[i];
        dst.at< cv::Vec3b >( pt )[0] = (uchar) ( (1.0-alpha)*(float)dst.at< cv::Vec3b >( pt )[0] + (alpha)*(float)_color[0] );
        dst.at< cv::Vec3b >( pt )[1] = (uchar) ( (1.0-alpha)*(float)dst.at< cv::Vec3b >( pt )[1] + (alpha)*(float)_color[1] );
        dst.at< cv::Vec3b >( pt )[2] = (uchar) ( (1.0-alpha)*(float)dst.at< cv::Vec3b >( pt )[2] + (alpha)*(float)_color[2] );

        // cv::Vec3d new_color( .5*_orgcolor[0]+.5*_color[0] + .5*_orgcolor[1]+.5*_color[1] + .5*_orgcolor[2]+.5*_color[2] )
      }

      if( float(n_outside_image_domain)/ float(pts_set.cols() ) > 0.2 ) // print only if u see too many outside the image
          cout << "[MiscUtils::plot_point_sets] with color at every point, found " << n_outside_image_domain << " outside the image of total points to plot=" << pts_set.cols() << "\n";
}




// plot point set on image.
//  im : Input image
//  pts_set : 2xN or 3xN matrix with x,y in a col, in terms of image row and colidx this will be c,r.
//  status : same size as im, once with status[k] == 0 will not be plotted
//  dst [output]: output image
void MiscUtils::plot_point_sets_masked( const cv::Mat& im, const MatrixXd& pts_set, const vector<uchar>& status,
        cv::Mat& dst,
        const cv::Scalar& color, bool enable_keypoint_annotation, const string msg )
{


    assert( im.rows > 0 && im.cols > 0 && "\n[MiscUtils::plot_point_sets]Image appears to be emoty. cannot plot.\n");
    // assert( pts_set.cols() > 0 && pts_set.cols() == annotations.rows() && "[MiscUtils::plot_point_sets] VectorXi annotation size must be equal to number of points. If you wish to use the default annotation ie. 0,1,...n use `true` instead. If you do not want annotation use `false`." );
    assert( status.size() > 0 && status.size() == pts_set.cols() );
    if( im.data == dst.data ) {
      //   cout << "src and dst are same\n";
        // src and dst images are same, so dont copy. just ensure it is a 3 channel image.
        assert( im.channels() == 3 && dst.channels() == 3 && "[MiscUtils::plot_point_sets]src and dst image are same physical image in memory. They need to be 3 channel." );
    }
    else {
      //   dst = cv::Mat( im.rows, im.cols, CV_8UC3 );
        if( im.channels() == 1 )
          cv::cvtColor( im, dst, cv::COLOR_GRAY2BGR );
        else
          im.copyTo(dst);
    }

    // cv::putText( dst, to_string(msg.length()), cv::Point(5,5), cv::FONT_HERSHEY_COMPLEX_SMALL, .95, cv::Scalar(0,255,255) );
    if( msg.length() > 0 ) {
      vector<std::string> msg_split;
      msg_split = MiscUtils::split( msg, ';' );
      for( int q=0 ; q<msg_split.size() ; q++ )
        cv::putText( dst, msg_split[q], cv::Point(5,20+20*q), cv::FONT_HERSHEY_COMPLEX_SMALL, .95, cv::Scalar(0,255,255) );
    }


    //pts_set is 2xN
    cv::Point2d pt;
    for( int i=0 ; i<pts_set.cols() ; i++ )
    {
      // pt = cv::Point2d(pts_set.at<float>(0,i),pts_set.at<float>(1,i) );
      if( status[i] == 0 )
        continue;

      pt = cv::Point2d( (float)pts_set(0,i), (float)pts_set(1,i) );
      cv::circle( dst, pt, 2, color, -1 );

      if( enable_keypoint_annotation ) {
          char to_s[20];
          sprintf( to_s, "%d", i );
          cv::putText( dst, to_s, pt, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, color  );
      }


    }
}


// plot point set on image.
//  im : Input image
//  pts_set : 2xN or 3xN matrix with x,y in a col, in terms of image row and colidx this will be c,r.
//  status : same size as im, once with status[k] == false will not be plotted
//  dst [output]: output image
void MiscUtils::plot_point_sets_masked( const cv::Mat& im, const MatrixXd& pts_set, const vector<bool>& status,
            cv::Mat& dst,
            const cv::Scalar& color, bool enable_keypoint_annotation , const string msg  )
{


    assert( im.rows > 0 && im.cols > 0 && "\n[MiscUtils::plot_point_sets]Image appears to be emoty. cannot plot.\n");
    // assert( pts_set.cols() > 0 && pts_set.cols() == annotations.rows() && "[MiscUtils::plot_point_sets] VectorXi annotation size must be equal to number of points. If you wish to use the default annotation ie. 0,1,...n use `true` instead. If you do not want annotation use `false`." );
    assert( status.size() > 0 && status.size() == pts_set.cols() );
    if( im.data == dst.data ) {
      //   cout << "src and dst are same\n";
        // src and dst images are same, so dont copy. just ensure it is a 3 channel image.
        assert( im.channels() == 3 && dst.channels() == 3 && "[MiscUtils::plot_point_sets]src and dst image are same physical image in memory. They need to be 3 channel." );
    }
    else {
      //   dst = cv::Mat( im.rows, im.cols, CV_8UC3 );
        if( im.channels() == 1 )
          cv::cvtColor( im, dst, cv::COLOR_GRAY2BGR );
        else
          im.copyTo(dst);
    }

    // cv::putText( dst, to_string(msg.length()), cv::Point(5,5), cv::FONT_HERSHEY_COMPLEX_SMALL, .95, cv::Scalar(0,255,255) );
    if( msg.length() > 0 ) {
      vector<std::string> msg_split;
      msg_split = MiscUtils::split( msg, ';' );
      for( int q=0 ; q<msg_split.size() ; q++ )
        cv::putText( dst, msg_split[q], cv::Point(5,20+20*q), cv::FONT_HERSHEY_COMPLEX_SMALL, .95, cv::Scalar(0,255,255) );
    }


    //pts_set is 2xN
    cv::Point2d pt;
    for( int i=0 ; i<pts_set.cols() ; i++ )
    {
      // pt = cv::Point2d(pts_set.at<float>(0,i),pts_set.at<float>(1,i) );
      if( status[i] == false )
        continue;

      pt = cv::Point2d( (float)pts_set(0,i), (float)pts_set(1,i) );
      cv::circle( dst, pt, 2, color, -1 );

      if( enable_keypoint_annotation ) {
          char to_s[20];
          sprintf( to_s, "%d", i );
          cv::putText( dst, to_s, pt, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, color  );
      }


    }
}




void MiscUtils::plot_point_sets_masked( const cv::Mat& im, const MatrixXd& pts_set,
            const VectorXd& status, double show_only_greater_than_this_value,
            cv::Mat& dst,
            const cv::Scalar& color, bool enable_keypoint_annotation , const string msg  )
{


    assert( im.rows > 0 && im.cols > 0 && "\n[MiscUtils::plot_point_sets]Image appears to be emoty. cannot plot.\n");
    // assert( pts_set.cols() > 0 && pts_set.cols() == annotations.rows() && "[MiscUtils::plot_point_sets] VectorXi annotation size must be equal to number of points. If you wish to use the default annotation ie. 0,1,...n use `true` instead. If you do not want annotation use `false`." );
    assert( status.size() > 0 && status.size() == pts_set.cols() );
    if( im.data == dst.data ) {
      //   cout << "src and dst are same\n";
        // src and dst images are same, so dont copy. just ensure it is a 3 channel image.
        assert( im.channels() == 3 && dst.channels() == 3 && "[MiscUtils::plot_point_sets]src and dst image are same physical image in memory. They need to be 3 channel." );
    }
    else {
      //   dst = cv::Mat( im.rows, im.cols, CV_8UC3 );
        if( im.channels() == 1 )
          cv::cvtColor( im, dst, cv::COLOR_GRAY2BGR );
        else
          im.copyTo(dst);
    }

    // cv::putText( dst, to_string(msg.length()), cv::Point(5,5), cv::FONT_HERSHEY_COMPLEX_SMALL, .95, cv::Scalar(0,255,255) );
    if( msg.length() > 0 ) {
      vector<std::string> msg_split;
      msg_split = MiscUtils::split( msg, ';' );
      for( int q=0 ; q<msg_split.size() ; q++ )
        cv::putText( dst, msg_split[q], cv::Point(5,20+20*q), cv::FONT_HERSHEY_COMPLEX_SMALL, .95, cv::Scalar(0,255,255) );
    }


    //pts_set is 2xN
    cv::Point2d pt;
    for( int i=0 ; i<pts_set.cols() ; i++ )
    {
      // pt = cv::Point2d(pts_set.at<float>(0,i),pts_set.at<float>(1,i) );
      if( status(i) < show_only_greater_than_this_value )
        continue;

      pt = cv::Point2d( (float)pts_set(0,i), (float)pts_set(1,i) );
      cv::circle( dst, pt, 2, color, -1 );

      if( enable_keypoint_annotation ) {
          char to_s[20];
          sprintf( to_s, "%d", i );
          cv::putText( dst, to_s, pt, cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, color  );
      }


    }
}


void MiscUtils::plot_point_pair( const cv::Mat& imA, const MatrixXd& ptsA, int idxA,
                      const cv::Mat& imB, const MatrixXd& ptsB, int idxB,
                    //   const VectorXd& mask,
                    cv::Mat& dst,
                    const cv::Scalar& color_marker,
                    const cv::Scalar& color_line,
                      bool annotate_pts,
                      /*const vector<string>& msg,*/
                      const string& msg
                    )
{
  // ptsA : ptsB : 2xN or 3xN

  assert( imA.rows == imB.rows && imA.rows > 0  );
  assert( imA.cols == imB.cols && imA.cols > 0  );
  // assert( ptsA.cols() == ptsB.cols() && ptsA.cols() > 0 );
  assert( ptsA.cols() == ptsB.cols()  );
  // assert( mask.size() == ptsA.cols() );

  cv::Mat outImg_;
  cv::hconcat(imA, imB, outImg_);

    cv::Mat outImg;
    if( outImg_.channels() == 3 )
        outImg = outImg_;
    else
        cv::cvtColor( outImg_, outImg, CV_GRAY2BGR );




  // loop over all points
  int count = 0;
  for( int kl=0 ; kl<ptsA.cols() ; kl++ )
  {
    // if( mask(kl) == 0 )
    //   continue;

    count++;
    cv::Point2d A( ptsA(0,kl), ptsA(1,kl) );
    cv::Point2d B( ptsB(0,kl), ptsB(1,kl) );

    cv::circle( outImg, A, 2,color_marker, -1 );
    cv::circle( outImg, B+cv::Point2d(imA.cols,0), 2,color_marker, -1 );

    cv::line( outImg,  A, B+cv::Point2d(imA.cols,0), color_line );

    if( annotate_pts )
    {
      cv::putText( outImg, to_string(kl), A, cv::FONT_HERSHEY_SIMPLEX, 0.3, color_marker, 1 );
      cv::putText( outImg, to_string(kl), B+cv::Point2d(imA.cols,0), cv::FONT_HERSHEY_SIMPLEX, 0.3, color_marker, 1 );
    }
  }



  cv::Mat status = cv::Mat(150, outImg.cols, CV_8UC3, cv::Scalar(0,0,0) );
  cv::putText( status, to_string(idxA).c_str(), cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2 );
  cv::putText( status, to_string(idxB).c_str(), cv::Point(imA.cols+10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2 );
  cv::putText( status, "marked # pts: "+to_string(count), cv::Point(10,60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 1.5 );


  // put msg in status image
  if( msg.length() > 0 ) { // ':' separated. Each will go in new line
      std::vector<std::string> msg_tokens = split(msg, ';');
      for( int h=0 ; h<msg_tokens.size() ; h++ )
          cv::putText( status, msg_tokens[h].c_str(), cv::Point(10,80+20*h), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 1.5 );
  }


  cv::vconcat( outImg, status, dst );

}



void MiscUtils::plot_point_pair( const cv::Mat& imA, const MatrixXd& ptsA,
                      const cv::Mat& imB, const MatrixXd& ptsB,
                      cv::Mat& dst,
                      const cv::Scalar& color_marker,
                      const string& msg,
                      const cv::Scalar& color_line,
                      bool annotate_pts )
{
  // ptsA : ptsB : 2xN or 3xN

  assert( imA.rows == imB.rows && imA.rows > 0  );
  assert( imA.cols == imB.cols && imA.cols > 0  );
  // assert( ptsA.cols() == ptsB.cols() && ptsA.cols() > 0 );
  assert( ptsA.cols() == ptsB.cols()  );
  // assert( mask.size() == ptsA.cols() );

  cv::Mat outImg_;
  cv::hconcat(imA, imB, outImg_);

    cv::Mat outImg;
    if( outImg_.channels() == 3 )
        outImg = outImg_;
    else
        cv::cvtColor( outImg_, outImg, CV_GRAY2BGR );




  // loop over all points
  int count = 0;
  for( int kl=0 ; kl<ptsA.cols() ; kl++ )
  {
    // if( mask(kl) == 0 )
    //   continue;

    count++;
    cv::Point2d A( ptsA(0,kl), ptsA(1,kl) );
    cv::Point2d B( ptsB(0,kl), ptsB(1,kl) );

    cv::circle( outImg, A, 2,color_marker, -1 );
    cv::circle( outImg, B+cv::Point2d(imA.cols,0), 2,color_marker, -1 );

    cv::line( outImg,  A, B+cv::Point2d(imA.cols,0), color_line );

    if( annotate_pts )
    {
      cv::putText( outImg, to_string(kl), A, cv::FONT_HERSHEY_SIMPLEX, 0.3, color_marker, 1 );
      cv::putText( outImg, to_string(kl), B+cv::Point2d(imA.cols,0), cv::FONT_HERSHEY_SIMPLEX, 0.3, color_marker, 1 );
    }
  }


  #if 0
  cv::Mat status = cv::Mat(150, outImg.cols, CV_8UC3, cv::Scalar(0,0,0) );
  // cv::putText( status, to_string(idxA).c_str(), cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2 );
  // cv::putText( status, to_string(idxB).c_str(), cv::Point(imA.cols+10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2 );
  cv::putText( status, "marked # pts: "+to_string(count), cv::Point(10,60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 1.5 );


  // put msg in status image
  if( msg.length() > 0 ) { // ':' separated. Each will go in new line
      std::vector<std::string> msg_tokens = split(msg, ';');
      for( int h=0 ; h<msg_tokens.size() ; h++ )
          cv::putText( status, msg_tokens[h].c_str(), cv::Point(10,80+20*h), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 1.5 );
  }


  cv::vconcat( outImg, status, dst );
  #endif

  append_status_image( outImg, "marked # pts: "+to_string(count) + ";;" + msg, 1.0 );
  dst = outImg;

}


cv::Scalar MiscUtils::getFalseColor( float f )
{
    cv::Mat colormap_gray = cv::Mat::zeros( 1, 256, CV_8UC1 );
    cv::Mat colormap_color;
    for( int i=0 ; i<256; i++ ) colormap_gray.at<uchar>(0,i) = i;
    cv::applyColorMap(colormap_gray, colormap_color, cv::COLORMAP_JET	);
    if( f<0 ) {
        cv::Vec3b f_ = colormap_color.at<cv::Vec3b>(0,  (int)0 );
        cv::Scalar color_marker = cv::Scalar(f_[0],f_[1],f_[2]);
        return color_marker;
    }
    if( f>1. ) {
        cv::Vec3b f_ = colormap_color.at<cv::Vec3b>(0,  (int)255 );
        cv::Scalar color_marker = cv::Scalar(f_[0],f_[1],f_[2]);
        return color_marker;
    }

    int idx = (int) (f*255.);
    cv::Vec3b f_ = colormap_color.at<cv::Vec3b>(0,  (int)idx );
    cv::Scalar color_marker = cv::Scalar(f_[0],f_[1],f_[2]);
    return color_marker;
}

void MiscUtils::plot_point_pair( const cv::Mat& imA, const MatrixXd& ptsA, int idxA,
                      const cv::Mat& imB, const MatrixXd& ptsB, int idxB,
                      cv::Mat& dst,
                      short color_map_direction,
                      const string& msg
                     )
{
  // ptsA : ptsB : 2xN or 3xN
  assert( color_map_direction >= 0 && color_map_direction<=3 );
  assert( imA.rows == imB.rows && imA.rows > 0  );
  assert( imA.cols == imB.cols && imA.cols > 0  );
  // assert( ptsA.cols() == ptsB.cols() && ptsA.cols() > 0 );
  assert( ptsA.cols() == ptsB.cols()  );
  // assert( mask.size() == ptsA.cols() );


  // make colormap
  cv::Mat colormap_gray = cv::Mat::zeros( 1, 256, CV_8UC1 );
  for( int i=0 ; i<256; i++ ) colormap_gray.at<uchar>(0,i) = i;
  cv::Mat colormap_color;
  cv::applyColorMap(colormap_gray, colormap_color, cv::COLORMAP_HSV);



  cv::Mat outImg_;
  cv::hconcat(imA, imB, outImg_);

  cv::Mat outImg;
  if( outImg_.channels() == 3 )
      outImg = outImg_;
  else
      cv::cvtColor( outImg_, outImg, CV_GRAY2BGR );





  // loop over all points
  int count = 0;
  for( int kl=0 ; kl<ptsA.cols() ; kl++ )
  {
    // if( mask(kl) == 0 )
    //   continue;

    count++;
    cv::Point2d A( ptsA(0,kl), ptsA(1,kl) );
    cv::Point2d B( ptsB(0,kl), ptsB(1,kl) );

    int coloridx;
    if( color_map_direction==0 ) coloridx = (int) (ptsA(0,kl)/imA.cols*256.); // horizontal-gradiant
    if( color_map_direction==1 ) coloridx = (int) (ptsA(1,kl)/imA.rows*256.); // vertical-gradiant
    if( color_map_direction==2 ) coloridx = (int) (   ( ptsA(0,kl) + ptsA(1,kl) ) / (imA.rows + imA.cols )*256.  ); // manhattan-gradiant
    if( color_map_direction==3 ) coloridx = (int) (   abs( ptsA(0,kl) - imA.rows/2. + ptsA(1,kl) - imA.cols/2. ) / (imA.rows/2. + imA.cols/2. )*256.  ); // image centered manhattan-gradiant
    if( coloridx<0 || coloridx>255 ) coloridx=0;
    cv::Vec3b f = colormap_color.at<cv::Vec3b>(0,  (int)coloridx );
    cv::Scalar color_marker = cv::Scalar(f[0],f[1],f[2]);

    cv::circle( outImg, A, 2,color_marker, -1 );
    cv::circle( outImg, B+cv::Point2d(imA.cols,0), 2,color_marker, -1 );

    /*
    cv::line( outImg,  A, B+cv::Point2d(imA.cols,0), color_line );

    if( annotate_pts )
    {
      cv::putText( outImg, to_string(kl), A, cv::FONT_HERSHEY_SIMPLEX, 0.3, color_marker, 1 );
      cv::putText( outImg, to_string(kl), B+cv::Point2d(imA.cols,0), cv::FONT_HERSHEY_SIMPLEX, 0.3, color_marker, 1 );
    }
    */
  }



  cv::Mat status = cv::Mat(150, outImg.cols, CV_8UC3, cv::Scalar(0,0,0) );
  cv::putText( status, to_string(idxA).c_str(), cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2 );
  cv::putText( status, to_string(idxB).c_str(), cv::Point(imA.cols+10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2 );
  cv::putText( status, "marked # pts: "+to_string(count), cv::Point(10,60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 1.5 );


  // put msg in status image
  if( msg.length() > 0 ) { // ':' separated. Each will go in new line
      std::vector<std::string> msg_tokens = split(msg, ';');
      for( int h=0 ; h<msg_tokens.size() ; h++ )
          cv::putText( status, msg_tokens[h].c_str(), cv::Point(10,80+20*h), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 1.5 );
  }


  cv::vconcat( outImg, status, dst );

}



void MiscUtils::plot_point_pair( const cv::Mat& imA, const MatrixXd& ptsA,
                      const cv::Mat& imB, const MatrixXd& ptsB,
                      cv::Mat& dst,
                      short color_map_direction,
                      const string& msg
                     )
{
  // ptsA : ptsB : 2xN or 3xN
  assert( color_map_direction >= 0 && color_map_direction<=3 );
  assert( imA.rows == imB.rows && imA.rows > 0  );
  assert( imA.cols == imB.cols && imA.cols > 0  );
  // assert( ptsA.cols() == ptsB.cols() && ptsA.cols() > 0 );
  assert( ptsA.cols() == ptsB.cols()  );
  // assert( mask.size() == ptsA.cols() );


  // make colormap
  cv::Mat colormap_gray = cv::Mat::zeros( 1, 256, CV_8UC1 );
  for( int i=0 ; i<256; i++ ) colormap_gray.at<uchar>(0,i) = i;
  cv::Mat colormap_color;
  cv::applyColorMap(colormap_gray, colormap_color, cv::COLORMAP_HSV);



  cv::Mat outImg_;
  cv::hconcat(imA, imB, outImg_);

  cv::Mat outImg;
  if( outImg_.channels() == 3 )
      outImg = outImg_;
  else
      cv::cvtColor( outImg_, outImg, CV_GRAY2BGR );





  // loop over all points
  int count = 0;
  for( int kl=0 ; kl<ptsA.cols() ; kl++ )
  {
    // if( mask(kl) == 0 )
    //   continue;

    count++;
    cv::Point2d A( ptsA(0,kl), ptsA(1,kl) );
    cv::Point2d B( ptsB(0,kl), ptsB(1,kl) );

    int coloridx;
    if( color_map_direction==0 ) coloridx = (int) (ptsA(0,kl)/imA.cols*256.); // horizontal-gradiant
    if( color_map_direction==1 ) coloridx = (int) (ptsA(1,kl)/imA.rows*256.); // vertical-gradiant
    if( color_map_direction==2 ) coloridx = (int) (   ( ptsA(0,kl) + ptsA(1,kl) ) / (imA.rows + imA.cols )*256.  ); // manhattan-gradiant
    if( color_map_direction==3 ) coloridx = (int) (   abs( ptsA(0,kl) - imA.rows/2. + ptsA(1,kl) - imA.cols/2. ) / (imA.rows/2. + imA.cols/2. )*256.  ); // image centered manhattan-gradiant
    if( coloridx<0 || coloridx>255 ) coloridx=0;
    cv::Vec3b f = colormap_color.at<cv::Vec3b>(0,  (int)coloridx );
    cv::Scalar color_marker = cv::Scalar(f[0],f[1],f[2]);

    cv::circle( outImg, A, 2,color_marker, -1 );
    cv::circle( outImg, B+cv::Point2d(imA.cols,0), 2,color_marker, -1 );

    /*
    cv::line( outImg,  A, B+cv::Point2d(imA.cols,0), color_line );

    if( annotate_pts )
    {
      cv::putText( outImg, to_string(kl), A, cv::FONT_HERSHEY_SIMPLEX, 0.3, color_marker, 1 );
      cv::putText( outImg, to_string(kl), B+cv::Point2d(imA.cols,0), cv::FONT_HERSHEY_SIMPLEX, 0.3, color_marker, 1 );
    }
    */
  }


  #if 0 //todo remove
  cv::Mat status = cv::Mat(150, outImg.cols, CV_8UC3, cv::Scalar(0,0,0) );
  cv::putText( status, to_string(idxA).c_str(), cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2 );
  cv::putText( status, to_string(idxB).c_str(), cv::Point(imA.cols+10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 2 );
  cv::putText( status, "marked # pts: "+to_string(count), cv::Point(10,60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,255), 1.5 );


  // put msg in status image
  if( msg.length() > 0 ) { // ':' separated. Each will go in new line
      std::vector<std::string> msg_tokens = split(msg, ';');
      for( int h=0 ; h<msg_tokens.size() ; h++ )
          cv::putText( status, msg_tokens[h].c_str(), cv::Point(10,80+20*h), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255,255,255), 1.5 );
  }


  cv::vconcat( outImg, status, dst );
  #endif

  append_status_image( outImg, "marked # pts: "+to_string(count) + ";" + msg, 1.0 );
  dst = outImg;

}


void MiscUtils::mask_overlay( const cv::Mat& src, const cv::Mat& mask, cv::Mat& dst, cv::Scalar color )
{
    if( src.empty() || mask.empty() ) {
        cout << __FILE__ << ":" << __LINE__ << "Either of src or mask is not allocated\n";
        throw 32;
    }

    if( src.rows == mask.rows && src.cols == mask.cols ) {
        ; // ok
    }
    else
    {
        cout << __FILE__ << ":" << __LINE__ << "[ MiscUtils::mask_overlay] invalid input\n";
        cout << "src :" << MiscUtils::cvmat_info( src ) << endl;
        cout << "mask :" << MiscUtils::cvmat_info( mask ) << endl;
        throw 32;
    }

    if( &src == &dst )
    {
        cout << __FILE__ << ":" << __LINE__ << "&src == &dst failed. expecting different src and dst in memory\n";
        throw 32;
    }


    if( src.channels() == 3 )
        src.copyTo(dst);
    else
        cv::cvtColor( src, dst, CV_GRAY2BGR );


    for( int r=0 ; r<src.rows ; r++ )
    {
        for( int c=0 ; c<src.cols ; c++ )
        {
            if( mask.at<uchar>(r,c) > 0 )
            {
                // auto pt = cv::Point2f( (float)c, (float)r );
                // cv::circle( dst, pt, 2, color, -1 );
                dst.at< cv::Vec3b >( r,c ) = cv::Vec3b( color[0], color[1], color[2] );
            }
        }
    }
}


void MiscUtils::mask_overlay( cv::Mat& src, const cv::Mat& mask, cv::Scalar color )
{
    if( src.empty() || mask.empty() ) {
        cout << __FILE__ << ":" << __LINE__ << "Either of src or mask is not allocated\n";
        throw 32;
    }

    if( src.rows == mask.rows && src.cols == mask.cols ) {
        ; // ok
    }
    else
    {
        cout << __FILE__ << ":" << __LINE__ << "[ MiscUtils::mask_overlay] invalid input\n";
        cout << "src :" << MiscUtils::cvmat_info( src ) << endl;
        cout << "mask :" << MiscUtils::cvmat_info( mask ) << endl;
        throw 32;
    }


    if( src.channels() != 3 ) {
        cout << __FILE__ << ":" << __LINE__ << " src.channels need to be 3 for this version of mask_overlay, you may use other overload instance of this otherwise\n";
        throw 32;
    }


    for( int r=0 ; r<src.rows ; r++ )
    {
        for( int c=0 ; c<src.cols ; c++ )
        {
            if( mask.at<uchar>(r,c) > 0 )
            {
                // auto pt = cv::Point2f( (float)c, (float)r );
                // cv::circle( dst, pt, 2, color, -1 );
                src.at< cv::Vec3b >( r,c ) = cv::Vec3b( color[0], color[1], color[2] );
            }
        }
    }
}

// append a status image . ';' separated
void MiscUtils::append_status_image( cv::Mat& im, const string& msg, float txt_size, cv::Scalar bg_color, cv::Scalar txt_color, float line_thinkness )
{
    bool is_single_channel = (im.channels()==1)?true:false;
    txt_size = (txt_size<0.1 || txt_size>2)?0.4:txt_size;

    std::vector<std::string> msg_tokens = split(msg, ';');
    const int height_per_line = 30 * txt_size;
    const int top_padding = height_per_line;
    int status_im_height = top_padding+height_per_line*(int)msg_tokens.size();

    cv::Mat status;
    if( is_single_channel )
        status = cv::Mat(status_im_height, im.cols, CV_8UC1, cv::Scalar(0,0,0) );
    else
        status = cv::Mat(status_im_height, im.cols, CV_8UC3, bg_color );


    for( int h=0 ; h<msg_tokens.size() ; h++ )
        cv::putText( status, msg_tokens[h].c_str(), cv::Point(10,height_per_line*h+height_per_line),
                cv::FONT_HERSHEY_SIMPLEX,
                txt_size, txt_color, line_thinkness );


    cv::vconcat( im, status, im );


}

bool MiscUtils::side_by_side( const cv::Mat& A, const cv::Mat& B, cv::Mat& dst )
{
    if( A.rows == B.rows && A.channels() == B.channels() ) {
        cv::hconcat(A, B, dst);
        return true;
    }
    else {
        dst = cv::Mat();
        return false;
    }
}

bool MiscUtils::vertical_side_by_side( const cv::Mat& A, const cv::Mat& B, cv::Mat& dst )
{
    if( A.cols == B.cols && A.channels() == B.channels() ) {
        cv::vconcat(A, B, dst);
        return true;
    }
    else {
        dst = cv::Mat();
        return false;
    }
}

double MiscUtils::Slope(int x0, int y0, int x1, int y1){
     return (double)(y1-y0)/(x1-x0);
}

void MiscUtils::draw_fullLine(cv::Mat& img, cv::Point2f a, cv::Point2f b, cv::Scalar color){
     double slope = MiscUtils::Slope(a.x, a.y, b.x, b.y);

     cv::Point2f p(0,0), q(img.cols,img.rows);

     p.y = -(a.x - p.x) * slope + a.y;
     q.y = -(b.x - q.x) * slope + b.y;

     cv::line(img,p,q,color,1,8,0);
}

// draw line on the image, given a line equation in homogeneous co-ordinates. l = (a,b,c) for ax+by+c = 0
void MiscUtils::draw_line( const Vector3d l, cv::Mat& im, cv::Scalar color )
{
    // C++: void line(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness=1, int lineType=8, int shift=0)
    if( l(0) == 0 ) {
        // plot y = -c/b
        cv::Point2f a(0.0, -l(2)/l(1) );
        cv::Point2f a_(10.0, -l(2)/l(1) );
        MiscUtils::draw_fullLine( im, a, a_, color );
        return;
    }

    if( l(1) == 0 ) {
        // plot x = -c/a
        cv::Point2f b(-l(2)/l(0), 0.0 );
        cv::Point2f b_(-l(2)/l(0), 10.0 );
        MiscUtils::draw_fullLine( im, b, b_, color );
        return;
    }

    cv::Point2f a(0.0, -l(2)/l(1) );
    cv::Point2f b(-l(2)/l(0), 0.0 );
    // cout << a << "<--->" << b << endl;
    // cv::line( im, a, b, cv::Scalar(255,255,255) );
    MiscUtils::draw_fullLine( im, a, b, color );
}

// mark point on the image, pt is in homogeneous co-ordinate.
void MiscUtils::draw_point( const Vector3d pt, cv::Mat& im, cv::Scalar color  )
{
    // C++: void circle(Mat& img, Point center, int radius, const Scalar& color, int thickness=1, int lineType=8, int shift=0)
    cv::circle( im, cv::Point2f( pt(0)/pt(2), pt(1)/pt(2) ), 2, color, -1   );

}

// mark point on image
void MiscUtils::draw_point( const Vector2d pt, cv::Mat& im, cv::Scalar color  )
{
    cv::circle( im, cv::Point2f( pt(0), pt(1) ), 2, color, -1   );
}
