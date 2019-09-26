#include "PoseComputation.h"

// #define _PoseComputation__closedFormSVD_(msg) msg;
#define _PoseComputation__closedFormSVD_(msg) ;
bool PoseComputation::closedFormSVD( const MatrixXd& aX, const MatrixXd& bX, Matrix4d& a_T_b )
{
    assert( aX.rows() == 4 && bX.rows() == 4 && aX.cols() == bX.cols() && aX.cols() > 0 );
    ElapsedTime _t; _t.tic();
    cout << TermColor::GREEN() << "=== PoseComputation::closedFormSVD input size = " << aX.rows() << "x" << aX.cols() << TermColor::RESET() << endl;

    // centroids
    VectorXd cen_aX = aX.rowwise().mean();
    VectorXd cen_bX = bX.rowwise().mean();
    _PoseComputation__closedFormSVD_(
    cout << "centroids computed: ";
    cout << "cen_aX = " << cen_aX.rows() << "x" << cen_aX.cols() << "\t" << cen_aX.transpose() << "\n";
    cout << "cen_bX = " << cen_bX.rows() << "x" << cen_bX.cols() << "\t" << cen_bX.transpose() << "\n";
    )


    // aX_cap:= aX - cen_aX
    // bX_cap:= bX - cen_bX
    // H := aX_cap * bX_cap.transpose()
    Matrix3d H = (bX.colwise() - cen_bX).topRows(3) * (aX.colwise() - cen_aX).topRows(3).transpose();
    _PoseComputation__closedFormSVD_(cout << "H=" << H.rows() << "x" << H.cols() << endl;)

    // U,S,Vt = svd( H )
    JacobiSVD<Matrix3d> svd( H, ComputeFullU | ComputeFullV);

    _PoseComputation__closedFormSVD_(
    cout << "Singular=\n" << svd.singularValues() << endl;
    cout << "U=\n" << svd.matrixU()  << endl;
    cout << "V=\n" << svd.matrixV()  << endl;
    )

    // R := V * Ut. if det(R) is -1, then use R = [v1, v2, -v3] * Ut
    Matrix3d R = svd.matrixV() * svd.matrixU().transpose();
    _PoseComputation__closedFormSVD_(
    cout << "R=\n" << R << endl;
    cout << "R.det=" << R.determinant() << endl;)
    assert( abs(R.determinant()-1.0)<1e-6 );

    // translation : mean(aX) - R mean(bX)
    Vector3d tr = aX.rowwise().mean().topRows(3) - R * bX.rowwise().mean().topRows(3);
    _PoseComputation__closedFormSVD_(cout << "Translation=" << tr << endl;)


    a_T_b = Matrix4d::Identity();
    a_T_b.topLeftCorner(3,3) = R;
    a_T_b.col(3).topRows(3) = tr;
    cout << "a_T_b=\n" << a_T_b << endl;
    cout << TermColor::BLUE() << "computation done in ms=" << _t.toc_milli() << TermColor::RESET() << endl;
    return true;
}
