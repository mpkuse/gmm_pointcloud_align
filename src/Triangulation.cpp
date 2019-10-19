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
