#include "Triangulation.h"



#if 0
/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
Mat_<double> IterativeLinearLSTriangulation(Point3d u,    //homogenous image point (u,v,1)
                                            Matx34d P,          //camera 1 matrix
                                            Point3d u1,         //homogenous image point in 2nd camera
                                            Matx34d P1          //camera 2 matrix
                                            ) {
    double wi = 1, wi1 = 1;
    Mat_<double> X(4,1);
    for (int i=0; i<10; i++) { //Hartley suggests 10 iterations at most
        Mat_<double> X_ = LinearLSTriangulation(u,P,u1,P1);
        X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X_(3) = 1.0;

        //recalculate weights
        double p2x = Mat_<double>(Mat_<double>(P).row(2)*X)(0);
        double p2x1 = Mat_<double>(Mat_<double>(P1).row(2)*X)(0);

        //breaking point
        if(fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON) break;

        wi = p2x;
        wi1 = p2x1;

        //reweight equations and solve
        Matx43d A((u.x*P(2,0)-P(0,0))/wi,       (u.x*P(2,1)-P(0,1))/wi,         (u.x*P(2,2)-P(0,2))/wi,
                  (u.y*P(2,0)-P(1,0))/wi,       (u.y*P(2,1)-P(1,1))/wi,         (u.y*P(2,2)-P(1,2))/wi,
                  (u1.x*P1(2,0)-P1(0,0))/wi1,   (u1.x*P1(2,1)-P1(0,1))/wi1,     (u1.x*P1(2,2)-P1(0,2))/wi1,
                  (u1.y*P1(2,0)-P1(1,0))/wi1,   (u1.y*P1(2,1)-P1(1,1))/wi1,     (u1.y*P1(2,2)-P1(1,2))/wi1
                  );
        Mat_<double> B = (Mat_<double>(4,1) <<    -(u.x*P(2,3)    -P(0,3))/wi,
                          -(u.y*P(2,3)  -P(1,3))/wi,
                          -(u1.x*P1(2,3)    -P1(0,3))/wi1,
                          -(u1.y*P1(2,3)    -P1(1,3))/wi1
                          );

        solve(A,B,X_,DECOMP_SVD);
        X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X_(3) = 1.0;
    }
    return X;
}
#endif


double Triangulation::IterativeLinearLSTriangulation(
    const Vector3d& u,  const Matrix4d& P,
    const Vector3d& u1,    const Matrix4d& P1   )
{
    // I am not too sure of this implementation check in the book.

    double wi = 1, wi1 = 1;
    const double EPSILON = 1e-7;
    // Mat_<double> X(4,1);
    Vector4d X;
    LinearLSTriangulation(u,P,u1,P1, X);
    for (int i=0; i<10; i++) { //Hartley suggests 10 iterations at most

        //recalculate weights
        double p2x = P.row(2) * X; // Mat_<double>(Mat_<double>(P).row(2)*X)(0);
        double p2x1 =P1.row(2) * X; //  Mat_<double>(Mat_<double>(P1).row(2)*X)(0);

        //breaking point
        if(fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON) break;

        wi = p2x;
        wi1 = p2x1;

        #if 0
        //reweight equations and solve
        Matx43d A((u.x*P(2,0)-P(0,0))/wi,       (u.x*P(2,1)-P(0,1))/wi,         (u.x*P(2,2)-P(0,2))/wi,
                  (u.y*P(2,0)-P(1,0))/wi,       (u.y*P(2,1)-P(1,1))/wi,         (u.y*P(2,2)-P(1,2))/wi,
                  (u1.x*P1(2,0)-P1(0,0))/wi1,   (u1.x*P1(2,1)-P1(0,1))/wi1,     (u1.x*P1(2,2)-P1(0,2))/wi1,
                  (u1.y*P1(2,0)-P1(1,0))/wi1,   (u1.y*P1(2,1)-P1(1,1))/wi1,     (u1.y*P1(2,2)-P1(1,2))/wi1
                  );
        Mat_<double> B = (Mat_<double>(4,1) <<    -(u.x*P(2,3)    -P(0,3))/wi,
                          -(u.y*P(2,3)  -P(1,3))/wi,
                          -(u1.x*P1(2,3)    -P1(0,3))/wi1,
                          -(u1.y*P1(2,3)    -P1(1,3))/wi1
                          );

        solve(A,B,X_,DECOMP_SVD);
        X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X_(3) = 1.0;
        #endif
        MatrixXd A = MatrixXd::Zero(4,3);
        A.row(0) << (u(0)*P(2,0)-P(0,0))/wi,       (u(0)*P(2,1)-P(0,1))/wi,         (u(0)*P(2,2)-P(0,2))/wi;
        A.row(1) << (u(1)*P(2,0)-P(1,0))/wi,       (u(1)*P(2,1)-P(1,1))/wi,         (u(1)*P(2,2)-P(1,2))/wi;
        A.row(2) << (u1(0)*P1(2,0)-P1(0,0))/wi1,   (u1(0)*P1(2,1)-P1(0,1))/wi1,     (u1(0)*P1(2,2)-P1(0,2))/wi1;
        A.row(3) << (u1(1)*P1(2,0)-P1(1,0))/wi1,   (u1(1)*P1(2,1)-P1(1,1))/wi1,     (u1(1)*P1(2,2)-P1(1,2))/wi1;

        Vector4d b;
        b <<  -(u(0)*P(2,3)    -P(0,3))/wi,
            -(u(1)*P(2,3)  -P(1,3))/wi,
            -(u1(0)*P1(2,3)    -P1(0,3))/wi1,
            -(u1(1)*P1(2,3)    -P1(1,3))/wi1;

        VectorXd tmp = A.bdcSvd(ComputeThinU | ComputeThinV).solve(b);
        X(0) = tmp(0);
        X(1) = tmp(1);
        X(2) = tmp(2);
        X(3) =  1.0;

    }
    // return X;
}



/**
 From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
 */
 void Triangulation::LinearLSTriangulation( const Vector3d& u,  const Matrix4d& P,
                                     const Vector3d& u1,    const Matrix4d& P1, Vector4d& result_X   )
{
    //build matrix A for homogenous equation system Ax = 0
    //assume X = (x,y,z,1), for Linear-LS method
    //which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
    // Matx43d A(u.x*P(2,0)-P(0,0),    u.x*P(2,1)-P(0,1),      u.x*P(2,2)-P(0,2),
    //       u.y*P(2,0)-P(1,0),    u.y*P(2,1)-P(1,1),      u.y*P(2,2)-P(1,2),
    //       u1.x*P1(2,0)-P1(0,0), u1.x*P1(2,1)-P1(0,1),   u1.x*P1(2,2)-P1(0,2),
    //       u1.y*P1(2,0)-P1(1,0), u1.y*P1(2,1)-P1(1,1),   u1.y*P1(2,2)-P1(1,2)
    //           );

    MatrixXd A = MatrixXd::Zero(4,3);
    A.row(0) << u(0)*P(2,0)-P(0,0), u(0)*P(2,1)-P(0,1), u(0)*P(2,2)-P(0,2);
    A.row(1) << u(1)*P(2,0)-P(1,0),    u(1)*P(2,1)-P(1,1),      u(1)*P(2,2)-P(1,2);
    A.row(2) << u1(0)*P1(2,0)-P1(0,0), u1(0)*P1(2,1)-P1(0,1),   u1(0)*P1(2,2)-P1(0,2);
    A.row(3) << u1(1)*P1(2,0)-P1(1,0), u1(1)*P1(2,1)-P1(1,1),   u1(1)*P1(2,2)-P1(1,2);

    // Mat_ B = (Mat_(4,1) <<    -(u.x*P(2,3)    -P(0,3)),
    //                   -(u.y*P(2,3)  -P(1,3)),
    //                   -(u1.x*P1(2,3)    -P1(0,3)),
    //                   -(u1.y*P1(2,3)    -P1(1,3)));
    Vector4d b;
    b <<  -(u(0)*P(2,3)   -P(0,3)),
           -(u(1)*P(2,3)   -P(1,3)),
           -(u1(0)*P1(2,3) -P1(0,3)),
           -(u1(1)*P1(2,3) -P1(1,3)) ;


    // Mat_ X;
    // solve(A,B,X,DECOMP_SVD);
    VectorXd X = A.bdcSvd(ComputeThinU | ComputeThinV).solve(b);
    result_X(0) = X(0);
    result_X(1) = X(1);
    result_X(2) = X(2);
    result_X(3) = 1.0;
}




// #define __Triangulation__MultiViewLinearLSTriangulation_(msg) msg;
#define __Triangulation__MultiViewLinearLSTriangulation_(msg) ;
bool Triangulation::MultiViewLinearLSTriangulation( const Vector3d& base_u,
    const vector<Vector3d>& tracked_u,   const vector<Matrix4d>& p_T_base,
    Vector4d& result_X,  const vector<bool>status )
{
    int N = tracked_u.size();
    assert( N>0 );
    assert( p_T_base.size() == N );

    bool use_all = false;
    if( status.size() == 0 )
        use_all = true;
    else {
        use_all = false;
        assert( status.size() == N );
    }

    int N_true = total_true( status );
    // cout << "[Triangulation::MultiViewLinearLSTriangulation]this u is visible in " << N_true << " adjacent images out of total " << N << " tracked images\n";

    // Matrix<double, Dynamic, 3>  A = Matrix<double, Dynamic, 3>::Zero( 2*N_true + 2, 3  ); // N_true x 3
    MatrixXd A = MatrixXd::Zero( 2*N_true + 2, 3);
    VectorXd b = VectorXd::Zero( 2*N_true + 2 );


    // 2 equations per tracking
    // note: R = [ r_1; r_2; r_3 ], r1 is 1st row of R;    t = [tx; ty; tz]
    // [ u r_3 - r_1 ] [ x ]        [tx - u*tz]
    // [ v r_3 - r_2 ] [ y ]    =   [ty - v*tz]
    //                 [ z ]        []
    int c=0;
    for( int i=0 ; i<N ; i++ )
    {
        if( status[i] == false )
            continue;

        #if 1
        Matrix3d p_R_base = p_T_base[i].topLeftCorner(3,3);
        Vector3d p_t_base = p_T_base[i].col(3).topRows(3);
        A.row( 2*c + 0 ) = p_R_base.row(2) * tracked_u[i](0) - p_R_base.row(0);
        b(2*c + 0) = p_t_base(0) - tracked_u[i](0) * p_t_base(2);

        A.row( 2*c + 1 ) = p_R_base.row(2) * tracked_u[i](1) - p_R_base.row(1);
        b(2*c + 1) = p_t_base(1) - tracked_u[i](1) * p_t_base(2);
        #endif
        c++;
    }

    // equation for base
    A.row(2*N_true) << -1.0, 0.0, base_u(0);
    b(2*N_true) = 0.0;
    A.row(2*N_true+1) << 0.0, -1.0, base_u(1);
    b(2*N_true+1) = 0.0;


    // Solve minimize_x ||Ax - b||
    __Triangulation__MultiViewLinearLSTriangulation_(
    cout << "N_true=" << N_true << "\t";
    cout << "A: " << A.rows() << "x" << A.cols() << "\t" << "b: " << b.rows() << "x" << b.cols() << endl;)
    VectorXd X = A.bdcSvd(ComputeThinU | ComputeThinV).solve(b);
    // VectorXd X = A.bdcSvd().solve(b);
    // VectorXd X = A.FullPivHouseholderQR().solve(b);
    result_X(0) = X(0);
    result_X(1) = X(1);
    result_X(2) = X(2);
    result_X(3) = 1.0;


    __Triangulation__MultiViewLinearLSTriangulation_(
    cout << "||AX-b|| = " << (A * X - b).squaredNorm() << endl; )


}
