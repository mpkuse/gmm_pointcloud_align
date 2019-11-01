#pragma once
// Form a local bundle and compte the relative pose between 2 sequences
//     a0---a1---a2----...... ---an
//
//      b0--b1--b2--...bm
//  Also have correspondences and depths at several randomly picked pairs



#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <queue>
#include <ostream>
#include <memory> //for std::shared_ptr
#include <map>
using namespace std;

// My utilities (mpkuse)
#include "utils/TermColor.h"
#include "utils/ElapsedTime.h"
#include "utils/PoseManipUtils.h"
#include "utils/RawFileIO.h"

// JSON
#include "utils/nlohmann/json.hpp"
using json = nlohmann::json;

// Eigen
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
using namespace Eigen;
#include <opencv2/core/eigen.hpp>

// Ceres
#include <ceres/ceres.h>
using namespace ceres;




class LocalBundle
{
public:
    LocalBundle();

    //-----------------//
    //----- Input -----//
    //-----------------//

    // The Odometry data for each frame in sequence-x. x can be a, b, c....
    void inputOdometry( int seqJ, vector<Matrix4d> _x0_T_c );

    // The initial guess between the sequences
    void inputInitialGuess( int seqa, int seqb, Matrix4d a0_T_b0 );


    void inputFeatureMatches( int seq_a, int seq_b,
        const vector<MatrixXd> all_normed_uv_a, const vector<MatrixXd> all_normed_uv_b );
    void inputFeatureMatchesDepths( int seq_a, int seq_b,
        const vector<VectorXd> all_d_a, const vector<VectorXd> all_d_b, const vector<VectorXd> all_sf );
    void inputFeatureMatchesPoses( int seq_a, int seq_b,
        const vector<Matrix4d> all_a0_T_a, const vector<Matrix4d> all_b0_T_b );


    // only for debug
    void inputFeatureMatchesImIdx( int seq_a, int seq_b, vector< std::pair<int,int> > all_pair_idx );
    void inputOdometryImIdx( int seqJ, vector<int> odom_seqJ_idx );

    void print_inputs_info() const;

    // save state to json
    json odomSeqJ_toJSON( int j ) const ;
    json matches_SeqPair_toJSON( int seq_a, int seq_b ) const ;
    void toJSON( const string BASE) const;


    // read state from json
    bool odomSeqJ_fromJSON(const string BASE, int j);
    bool matches_SeqPair_fromJSON(const string BASE, int seqa, int seqb);
    void fromJSON( const string BASE ) ;


    //------------------//
    //----- Solver -----//
    //------------------//
    void solve();




    //--------------------//
    //----- Retrive ------//
    //-------------------//



private:
    map< int , vector<Matrix4d> > x0_T_c; //pose of camera wrt its 1st frame (ie. 0th frame)

    map< std::pair<int,int> ,  Matrix4d > a0_T_b0; //initial guess of the relative pose between frame-0 of 2 sequences.

    map< std::pair<int,int>,  vector<MatrixXd>   > normed_uv_a;
    map< std::pair<int,int>,  vector<VectorXd>   > d_a;
    map< std::pair<int,int>,  vector<Matrix4d>   > a0_T_a;

    map< std::pair<int,int>,  vector<MatrixXd>   > normed_uv_b;
    map< std::pair<int,int>,  vector<VectorXd>   > d_b;
    map< std::pair<int,int>,  vector<Matrix4d>   > b0_T_b;

    // debug
    map< int, vector<int> > seq_x_idx;
    map< std::pair<int,int> , vector< std::pair<int,int> >  >all_pair_idx;

};
