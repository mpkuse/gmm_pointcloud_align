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
