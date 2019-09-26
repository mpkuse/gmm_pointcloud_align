#include "PoseComputation.h"

#include "utils/RawFileIO.h"

int main()
{
    cout << "Gello world\n";

    // Load Data
    MatrixXd aX, bX;
    RawFileIO::read_eigen_matrix( "/app/catkin_ws/src/gmm_pointcloud_align/resources/pointsets/731-2903__sa_c0_X.txt", aX );
    RawFileIO::read_eigen_matrix( "/app/catkin_ws/src/gmm_pointcloud_align/resources/pointsets/731-2903__sb_c0_X.txt", bX );

    // Close form computation
    Matrix4d bTa;
    PoseComputation::closedFormSVD( aX, bX, bTa );
}
