#include "SurfelMap.h"


SurfelXMap::SurfelXMap( camodocal::CameraPtr _camera)
{
    clear_data();
    m_camera = _camera;
    surfel_mutex = new std::mutex();

}

bool SurfelXMap::clear_data()
{
    camIdx.clear();
    w_T_c.clear();
    sp_cX.clear();
    sp_uv.clear();

    S__wX = MatrixXd::Zero( 4, 10000 );
    S__normal = MatrixXd::Zero( 4, 10000 );
    S__size = 0;
    n_fused.clear();
    n_unstable.clear();
}

// consider passing reference of the local pointcloud TODO
#define __FUSE__( msg ) msg;
// #define __FUSE__( msg ) ;


// debug for part-1
// #define __FUSE_A__debug( msg ) msg;
#define __FUSE_A__debug( msg ) ;


// debug for part-2
// #define __FUSE__debug( msg ) msg;
#define __FUSE__debug( msg ) ;

bool SurfelXMap::fuse_with( int i, Matrix4d __wTc, MatrixXd __sp_cX, MatrixXd __sp_uv,
    cv::Mat& image_i, cv::Mat& depth_i )
{
    cout << TermColor::iGREEN() << "[SurfelXMap::fuse_with] Starts\n" << TermColor::RESET();

    cout << "---Collect Data\n";
    camIdx.push_back( i );
    w_T_c.push_back( __wTc );
    sp_cX.push_back( __sp_cX );
    sp_uv.push_back( __sp_uv );


    cout << "Existing sizes: ";
    cout << "w_T_c=" << w_T_c.size() << "\t";
    cout << "sp_cX=" << sp_cX.size() << "\t";
    cout << "sp_uv=" << sp_uv.size() << "\t";
    cout << "\nsp_cX[0]=" << sp_cX[0].rows() << " x " << sp_cX[0].cols() << "\t";
    cout << "sp_uv[0]=" << sp_uv[0].rows() << " x " << sp_uv[0].cols() << "\t";
    cout << endl;

    assert( __sp_cX.rows() == 4 && __sp_cX.cols() == __sp_uv.cols() && __sp_uv.cols() > 0 );
    assert( image_i.rows == depth_i.rows && image_i.rows > 0 );
    assert( image_i.cols == depth_i.cols && image_i.cols > 0 );


    // if this was the first node in the series, init new surfels at each superpixel
    MatrixXd __sp_wX = __wTc  *  __sp_cX; //currrent superpixel in world frame-of-ref
    if(  w_T_c.size() == 1  )
    {
        cout << "initialize new surfels for each current superpixel\n";


        for( int i=0 ; i<__sp_cX.cols() ; i++ ) {
            if( __sp_cX(2,i) < 0.5 ||  __sp_cX(2,i) > 20  ) //dont initialize surfels if the depth values are not in the stereo range
                continue;

            {
                std::lock_guard<std::mutex> lk(*surfel_mutex);
                S__wX.col(S__size) = __sp_wX.col(i);
                S__normal.col( S__size ) = Vector4d( 1.0, 1.0, 1.0, 0.0 ); //TODO
                n_fused.push_back(0);
                n_unstable.push_back(0);
                S__size++;
            }
        }
        cout << "[SurfelXMap::fuse_with]Added " << __sp_cX.cols() << " new surfels, current surfel count = " << S__size << endl;
        cout << TermColor::iGREEN() << "[SurfelXMap::fuse_with] ENDS\n" << TermColor::RESET();
        return true;

    }





    //if this was not the first,
    if( w_T_c.size() > 1 )
    {
        //--------------------------------------------------------------------------------------//
        //------- reproject existing surfels on this view and -------//
        //------- update the surfels 3d position if possible from the current depth map. -------//
        //--------------------------------------------------------------------------------------//
        #if 1
        const double fuse_near = .5, fuse_far = 30.0 ;
        cout << "###\n### reproject all existing surfels (count=" << S__size << ") on this view  and merge if appropriate\n###\n";
        cout << "uv_c" << i << " := PI( w_T_c" << i << ".inv() * wX )\n";
        int n_surfs_not_in_view_frustum = 0;
        int n_surfs_not_visible_in_current_view = 0;
        int n_surfs_invalid_reproj_depth = 0;
        int n_surfs_consistent_with_depth_at_reprojection_on_current_view = 0;
        int n_surfs_inconsistent_with_depth_at_reprojection_on_current_view = 0;

        MatrixXd __ciX = __wTc.inverse() * S__wX.leftCols( S__size );

        //loop over all surfels and perspective_proj each surfel-point on this view, for those surfels visible in this view try to update the 3d positions
        assert( __ciX.rows() == 4 && __ciX.cols() > 0 ); assert( m_camera );
        for( int i=0 ; i<__ciX.cols() ; i++ )
        {
            __FUSE_A__debug(
            cout << TermColor::YELLOW() << "----global_surfel#" << i << TermColor::RESET() << endl;
            cout << "(3dpt) :: S__wX.col(i)=" << S__wX.col(i).transpose() << "\t";
            cout << "__ciX.col(i)=" << __ciX.col(i).transpose() << endl; )
            if( __ciX(2,i) < fuse_near || __ciX(2,i) > fuse_far ) {
                n_surfs_not_in_view_frustum++;
                __FUSE_A__debug( cout << "this surfels depth wrt current camera is either too far or behind me, so skip this surfel\n"; )
                continue;
            }


            //perspective project
            Vector2d c_p; //< projection of the surfel in this camera view
            m_camera->spaceToPlane( __ciX.col(i).topRows(3), c_p );
            __FUSE_A__debug( cout << "this surfel got projected on current view at c_p=" << c_p.transpose() << endl; )
            if( c_p(0) <= 0 || c_p(0) > image_i.cols || c_p(1) <= 0 || c_p(1) > image_i.rows  ) {
                n_surfs_not_visible_in_current_view++;
                __FUSE_A__debug( cout << "\tthis surfel is not visible in the current view image, so skip it\n"; )
                continue;
            }

            __FUSE_A__debug( cout << TermColor::GREEN() << "this point is visible in current image\n" << TermColor::RESET() ; )

            // look at the depth of c_p in current view, make sure it matches to the surfels depth
            float depth_val;
            if( depth_i.type() == CV_16UC1 )
                depth_val = (float) depth_i.at<uint16_t>(  (int)c_p(1), (int)c_p(0)  );
            else if( depth_i.type() == CV_32FC1 )
                depth_val = (float) depth_i.at<float>( (int)c_p(1), (int)c_p(0)  );
            else {
                cout << "[SurfelXMap::fuse_with]depth type is neighter of CV_16UC1 or CV_32FC1\n";
                assert( false );
                exit(1);
            }

            if( depth_val <= 1e-5 || depth_val > 20. ) {
                n_surfs_invalid_reproj_depth++;
                __FUSE_A__debug( cout << TermColor::RED() << "at this point current_depth has no value (NAN), so skip this surfel\n" << TermColor::RESET(); )
                continue;
            }

            __FUSE_A__debug(
            cout << "depth_val in the current_depth at this point was: " << depth_val << "\t";
            cout << "comparing this to depth of the surfel in this view (ie. __ciX(2,i) = " << __ciX(2,i) << ")\n"; )



            if( abs(depth_val - __ciX(2,i))  < 1.0 )
            {
                // this `depth_val` can be used to further improve the surfel's depth value
                n_surfs_consistent_with_depth_at_reprojection_on_current_view++;
                __FUSE_A__debug(
                cout << TermColor::GREEN() << "   diff = " << abs(depth_val - __ciX(2,i)) << ", better than the threshold\n";)

                #if 1
                // update surfel's 3d position with current depth_val.
                if( depth_val > 0.5 && depth_val < 10 ) { //if depth_val is in normal range, i am more confident about depth_val,
                     Vector3d c_PPP;
                     m_camera->liftProjective( c_p, c_PPP );
                     c_PPP *= depth_val; // the new 3d position of this surfel (in current camera frame-of-ref)
                     Vector4d updated_ciX = .7 * Vector4d(c_PPP(0),c_PPP(1),c_PPP(2),1) + .3 * __ciX.col(i);
                     {
                     std::lock_guard<std::mutex> lk(*surfel_mutex);
                     S__wX.col(i) = __wTc * updated_ciX;
                     }
                }
                else {
                    Vector3d c_PPP;
                    m_camera->liftProjective( c_p, c_PPP );
                    c_PPP *= depth_val; // the new 3d position of this surfel (in current camera frame-of-ref)
                    Vector4d updated_ciX = .3 * Vector4d(c_PPP(0),c_PPP(1),c_PPP(2),1) + .7 * __ciX.col(i);
                    {
                    std::lock_guard<std::mutex> lk(*surfel_mutex);
                    S__wX.col(i) = __wTc * updated_ciX;
                    }

                }
                __FUSE_A__debug(
                cout << "   updated surfel #" << i << "'s 3d position to : " << S__wX.col(i).transpose() << endl;
                )
                #endif
                n_fused[i]++;

            } else
            {
                // if the surfels depth is repeatingly not agreeing to current depth, then it is a sign that the depth estimate of the surfel is bad.
                n_surfs_inconsistent_with_depth_at_reprojection_on_current_view++;
                __FUSE_A__debug(
                cout << TermColor::RED() << "\tthis is not agreeing to the surfels depth. this usually means the depth value of the surfel was bad or the current depth value was bad.\n" << TermColor::RESET(); )
                n_unstable[i]++;
            }

        }


        cout << TermColor::CYAN();
        cout <<  "n_surfs_not_in_view_frustum = " << n_surfs_not_in_view_frustum << "\t";
        cout <<  "n_surfs_not_visible_in_current_view = " << n_surfs_not_visible_in_current_view << "\t";
        cout <<  "n_surfs_invalid_reproj_depth = " << n_surfs_invalid_reproj_depth << "\t";
        cout << endl;
        cout <<  "n_surfs_consistent_with_depth_at_reprojection_on_current_view = " << n_surfs_consistent_with_depth_at_reprojection_on_current_view << "\t";
        cout << endl;
        cout <<  "n_surfs_inconsistent_with_depth_at_reprojection_on_current_view = " << n_surfs_inconsistent_with_depth_at_reprojection_on_current_view << "\t";
        cout << TermColor::RESET() << endl;

        #endif


        //-------------------------------------------------------------------------------------------------//
        //------- loop over current super-pixels and see if any new surfels can be added to the map -------//
        //-------------------------------------------------------------------------------------------------//
        #if 1
        cout << "###\n### loop over current super-pixels (n=" << __sp_wX.cols() << ") and see if any new surfels can be added to the map (map has " << S__size << " surfels)\n###\n";
        const double radius = 0.2;
        int n_new_surfels = 0;
        for( int i=0 ; i<__sp_wX.cols() ; i++ )
        {
            __FUSE__debug(
            cout << TermColor::YELLOW() << "-----superpixel#" << i << TermColor::RESET() << endl;
            cout << "__sp_wX.col(i) = " << __sp_wX.col(i).transpose() << endl;
            )
            vector<int> radiusIdx;
            find_nn_of_b_in_A( S__wX.leftCols( S__size ), __sp_wX.col(i) , radius, radiusIdx  );

            __FUSE__debug(
            cout << "found " << radiusIdx.size() << " surfels (total surfels in the map=" << S__size<< ") within radius=" << radius << " of this superpixel\n";
            if( radiusIdx.size() > 0 ) {
            cout << "idx = ";
            for( int k=0 ; k< (int)radiusIdx.size(); k++ )
                cout << radiusIdx[k] << ", ";
            cout << endl;
            }
            )


            if( radiusIdx.size() == 0 ) {
                // attempt to add new surfel
                {
                    std::lock_guard<std::mutex> lk(*surfel_mutex);
                    __FUSE__debug(
                    cout << TermColor::CYAN() <<  "Add new surfel to the map\n" << TermColor::RESET();)
                    S__wX.col(S__size) = __sp_wX.col(i);
                    S__normal.col( S__size ) = Vector4d( 1.0, 1.0, 1.0, 0.0 ); //TODO
                    n_fused.push_back(0);
                    n_unstable.push_back(0);
                    S__size++;
                }
                n_new_surfels++;
            } else {
                // update an existing superpixel
                __FUSE__debug(
                cout << TermColor::GREEN() << "this superpixel looks like already exist in the map as a surfel, so just make note of this and ignore this superpixel\n" << TermColor::RESET();)
            }
        }
        cout << TermColor::CYAN();
        cout << "I looped through " << __sp_wX.cols() << " superpixels";
        cout << "\tadded " << n_new_surfels << " new surfels to the map";
        cout << "\tnow the map has " << S__size << " surfels\n" ;
        cout << TermColor::RESET();
        #endif
    }


    // TODO:
    // print means and sigmas of the current surfels.


    cout << TermColor::iGREEN() << "[SurfelXMap::fuse_with] ENDS\n" << TermColor::RESET();
    return true;
}



//************************** Retrive Data *******************************//
int SurfelXMap::surfelSize() const
{
    std::lock_guard<std::mutex> lk(*surfel_mutex);
    return S__size;
}

Vector4d SurfelXMap::surfelWorldPosition(int i) const //returns i'th 3d pt
{
    std::lock_guard<std::mutex> lk(*surfel_mutex);
    assert( i >=0 && i<S__size );
    return S__wX.col(i);
}


MatrixXd SurfelXMap::surfelWorldPosition() const //returns all 3d points in db
{
    std::lock_guard<std::mutex> lk(*surfel_mutex);
    return S__wX.leftCols( S__size  );
}


//*************************** HELPERS *************************************//
void SurfelXMap::perspectiveProject3DPoints( const MatrixXd& c_V, MatrixXd& c_v )
{
    assert( c_V.rows() == 4 && c_V.cols() > 0 );

    // cout << "c_V: " << c_V.rows() << "," << c_V.cols() << "\t";
    // cout << "c_v: " << c_v.rows() << "," << c_v.cols() << "\n";
    // return ;
    //c_V : 4xN
    if( !m_camera ) {
        cout << TermColor::RED() << "[SurfelXMap::perspectiveProject3DPoints] FATAL ERROR The cameras was not set...you need to set the camera to this class before calling the perspective function\n" << TermColor::RESET();
        exit(2);
    }

    c_v = MatrixXd::Zero( 3, c_V.cols() );
    for( int i=0 ; i<c_V.cols() ; i++ ) {
        Vector2d p;
        m_camera->spaceToPlane( c_V.col(i).topRows(3), p );
        c_v.col(i) << p, 1.0;
    }



}



bool SurfelXMap::find_nn_of_b_in_A( const MatrixXd& A, const VectorXd& b,
    const double radius, vector<int>& to_retidx ) const
{
    assert( A.rows() == b.rows() && (b.rows() > 0 ) );
    assert( A.cols() > 0 );
    assert( radius > 0 );
    to_retidx.clear();

    // brute force implementation
    for( int i=0 ; i<A.cols() ; i++ ) {
        double dist =  (A.col(i) - b).norm();
        if( dist < radius )
            to_retidx.push_back( i );
    }

    // TODO: use some flann library to have the same effect, maybe nanoflann?
}
