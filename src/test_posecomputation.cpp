#include "PoseComputation.h"

#include "utils/RawFileIO.h"

#if 0
int main()
{
    cout << "Gello world\n";

    // Load Data
    MatrixXd aX, bX;
    RawFileIO::read_eigen_matrix( "/app/catkin_ws/src/gmm_pointcloud_align/resources/pointsets/731-2903__sa_c0_X.txt", aX );
    RawFileIO::read_eigen_matrix( "/app/catkin_ws/src/gmm_pointcloud_align/resources/pointsets/731-2903__sb_c0_X.txt", bX );

    // Close form computation
    Matrix4d aTb;
    PoseComputation::closedFormSVD( aX, bX, bTb );
}
#endif

#if 0
int main()
{
    cout << "Gwello World\n";

    // Load Data
    MatrixXd aX, bX;
    RawFileIO::read_eigen_matrix( "/app/catkin_ws/src/gmm_pointcloud_align/resources/pointsets/741-2953__dst0.txt", aX );
    RawFileIO::read_eigen_matrix( "/app/catkin_ws/src/gmm_pointcloud_align/resources/pointsets/741-2953__dst1.txt", bX );

    Matrix4d aTb;
    PoseComputation::align3D3DWithRefinement( aX, bX, aTb );
}
#endif

template <typename Derived>
void trip_nans_and_infs( MatrixBase<Derived>& X )
{
    for( int c = 0 ; c<X.cols() ; c++ )
        for( int r=0 ; r<X.rows() ; r++ )
            if( X(r,c) != X(r,c) )
                X(r,c) = 0;
}

int main()
{
    cout << "Gwello World\n";

    // Load Data
    VectorXd dst_d_a, dst_d_b, dst_sf;
    MatrixXd dst_aX_sans_depth, dst_bX_sans_depth;
    RawFileIO::read_eigen_matrix( "/app/catkin_ws/src/gmm_pointcloud_align/resources/pointsets/dst_d_a.txt", dst_d_a );
    RawFileIO::read_eigen_matrix( "/app/catkin_ws/src/gmm_pointcloud_align/resources/pointsets/dst_d_b.txt", dst_d_b );
    RawFileIO::read_eigen_matrix( "/app/catkin_ws/src/gmm_pointcloud_align/resources/pointsets/dst_sf.txt", dst_sf );
    RawFileIO::read_eigen_matrix( "/app/catkin_ws/src/gmm_pointcloud_align/resources/pointsets/dst_aX_sans_depth.txt", dst_aX_sans_depth );
    RawFileIO::read_eigen_matrix( "/app/catkin_ws/src/gmm_pointcloud_align/resources/pointsets/dst_bX_sans_depth.txt", dst_bX_sans_depth );




    Matrix4d a_T_b;
    PoseComputation::closedFormSVD( dst_aX_sans_depth, dst_d_a, dst_bX_sans_depth, dst_d_b,  dst_sf, a_T_b  );


}
