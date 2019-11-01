#include "LocalBundle.h"


LocalBundle::LocalBundle()
{

}

//----------------------------------------------------------------------------//
//              INPUT
//----------------------------------------------------------------------------//


void LocalBundle::inputOdometry( int seqJ, vector<Matrix4d> _x0_T_c )
{
    if( this->x0_T_c.count( seqJ) != 0 ) {
        cout << TermColor::RED() << "[LocalBundle::inputOdometry] ERROR, the seqID=" << seqJ << ", which was the input already exisit. This mean for this seq you have already input the odometry.\n" << TermColor::RESET();
        cout << "....exit....\n";
        exit(2);
    }

    cout << "[LocalBundle::inputOdometry] set odometry for seqID=" << seqJ << " this seq has " << _x0_T_c.size() << " frames" <<  endl;
    x0_T_c[ seqJ ] = _x0_T_c;
}


void LocalBundle::inputInitialGuess(  int seqa, int seqb, Matrix4d ___a0_T_b0 )
{
    auto p = std::make_pair( seqa, seqb );
    if( this->a0_T_b0.count(p) != 0 ) {
        cout << TermColor::RED() << "[LocalBundle::inputInitialGuess] ERROR the initial guess between 0th frames of seq= " << seqa << " and seq=" << seqb << " already exist. You are trying to set it again. This is not the intended purpose.\n" << TermColor::RESET();
        cout << "....exit....\n";
        exit(2);
    }

    cout << "[LocalBundle::inputInitialGuess] set initial guess for a0_T_b0={" << seqa << "}0_T_{" << seqb<< "}0" << endl;
    this->a0_T_b0[ p ] = ___a0_T_b0;
}



void LocalBundle::inputFeatureMatches( int seq_a, int seq_b,
    const vector<MatrixXd> all_normed_uv_a, const vector<MatrixXd> all_normed_uv_b )
{
    assert( all_normed_uv_a.size() == all_normed_uv_b.size() && all_normed_uv_a.size() > 0 );
    for( int i=0 ; i<(int)all_normed_uv_a.size() ; i++ )
    {
        assert( all_normed_uv_a[i].cols() == all_normed_uv_b[i].cols() );
        assert( all_normed_uv_a[i].rows() == all_normed_uv_b[i].rows() && (all_normed_uv_b[i].rows() == 2 || all_normed_uv_b[i].rows()==3) );
        // cout << "[LocalBundle::inputFeatureMatches]image-pair#" << i << " has " << all_normed_uv_a[i].cols() << " feature correspondences\n";
    }

    auto p = std::make_pair( seq_a, seq_b );
    if( this->normed_uv_a.count(p) !=0 || this->normed_uv_b.count(p) != 0 ) {
        cout << TermColor::RED() << "[LocalBundle::inputFeatureMatches] ERROR this pair " << seq_a << "," << seq_b << " already exists\n" << TermColor::RESET();
        cout << "exit...\n";
        exit(2);
    }

    cout << "[LocalBundle::inputFeatureMatches] Set correspondences for " << all_normed_uv_a.size() << " image-pairs in seqa=" << seq_a << ", seqb=" << seq_b << endl;
    for( int i=0 ; i<(int)all_normed_uv_a.size() ; i++ )
    {
        cout << ">>>> [LocalBundle::inputFeatureMatches]image-pair#" << i << " has " << all_normed_uv_a[i].cols() << " feature correspondences\n";
    }
    normed_uv_a[p] = all_normed_uv_a;
    normed_uv_b[p] = all_normed_uv_b;
}


void LocalBundle::inputFeatureMatchesDepths( int seq_a, int seq_b,
    const vector<VectorXd> all_d_a, const vector<VectorXd> all_d_b, const vector<VectorXd> all_sf )
{
    assert( all_d_a.size() == all_d_b.size() && all_d_a.size() > 0 );
    for( int i=0 ; i<(int)all_d_a.size() ; i++ )
    {
        assert( all_d_a[i].size() == all_d_b[i].size() && all_d_a[i].size() > 0 );
        cout << "[LocalBundle::inputFeatureMatchesDepths]image-pair#" << i << " has " << all_d_a[i].size() << " depth values\n";
    }

    auto p = std::make_pair( seq_a, seq_b );
    if( this->d_a.count(p) !=0 || this->d_b.count(p) != 0 ) {
        cout << TermColor::RED() << "[LocalBundle::inputFeatureMatchesDepths] ERROR this pair " << seq_a << "," << seq_b << " already exists\n" << TermColor::RESET();
        cout << "exit...\n";
        exit(2);
    }

    cout << "[LocalBundle::inputFeatureMatchesDepths] Set depths for " << all_d_a.size() << " image-pairs in seqa=" << seq_a << ", seqb=" << seq_b << endl;
    for( int i=0 ; i<(int)all_d_a.size() ; i++ )
    {
        cout << ">>>> [LocalBundle::inputFeatureMatchesDepths]image-pair#" << i << " has " << all_d_a[i].size() << " depth values\n";
    }

    d_a[p] = all_d_a;
    d_b[p] = all_d_b;
    // TODO save all_sf???
}


void LocalBundle::inputFeatureMatchesPoses( int seq_a, int seq_b,
    const vector<Matrix4d> all_a0_T_a, const vector<Matrix4d> all_b0_T_b )
{
    assert( all_a0_T_a.size() == all_b0_T_b.size() && all_a0_T_a.size() > 0 );

    auto p = std::make_pair( seq_a, seq_b );
    if( this->a0_T_a.count(p) !=0 || this->b0_T_b.count(p) != 0 ) {
        cout << TermColor::RED() << "[LocalBundle::inputFeatureMatchesPoses] ERROR this pair " << seq_a << "," << seq_b << " already exists\n" << TermColor::RESET();
        cout << "exit...\n";
        exit(2);
    }

    cout << "[LocalBundle::inputFeatureMatchesPoses] Set poses (a0_T_a and b0_T_b) for " << all_a0_T_a.size() << " image-pairs in seqa=" << seq_a << ", seqb=" << seq_b << endl;
    a0_T_a[p] = all_a0_T_a;
    b0_T_b[p] = all_b0_T_b;

}


void LocalBundle::inputFeatureMatchesImIdx( int seq_a, int seq_b, vector< std::pair<int,int> > all_pair_idx )
{
    auto p = std::make_pair( seq_a, seq_b );
    this->all_pair_idx[ p ] = all_pair_idx;
}

void LocalBundle::inputOdometryImIdx( int seqJ, vector<int> odom_seqJ_idx )
{
    seq_x_idx[ seqJ ] = odom_seqJ_idx;
}



//----------------------------------------------------------------------------//
//             END INPUT
//----------------------------------------------------------------------------//




//----------------------------------------------------------------------------//
//      print, json IO
//----------------------------------------------------------------------------//

json LocalBundle::odomSeqJ_toJSON( int j ) const
{
    json obj;
    obj["seqID"] = j;

    assert( x0_T_c.count(j) != 0 && seq_x_idx.count(j) != 0 );


    // save `x0_T_c[j]`
    // save `seq_x_idx[j]`
    obj["data"] = json();
    auto odom_poses = x0_T_c.at( j );
    auto odom_idx = seq_x_idx.at( j );


    assert( odom_poses.size() == odom_idx.size() && odom_poses.size() > 0 );
    for( int i=0 ; i<(int)odom_poses.size() ; i++ )
    {
        json tmp;
        tmp["c0_T_c"] = RawFileIO::write_eigen_matrix_tojson( odom_poses.at(i)  );
        tmp["idx"] = odom_idx.at(i);

        obj["data"].push_back( tmp );
    }

    return obj;
}

json LocalBundle::matches_SeqPair_toJSON( int seq_a, int seq_b ) const
{
    json obj;
    obj["seq_a"] = seq_a;
    obj["seq_b"] = seq_b;
    auto p = std::make_pair( seq_a, seq_b );
    assert( this->a0_T_b0.count(p) > 0 );


    // save `a0_T_b0`
    obj["initial_guess____a0_T_b0"] = RawFileIO::write_eigen_matrix_tojson( a0_T_b0.at(p) );

    assert( this->normed_uv_a.count(p) > 0 );
    assert( this->d_a.count(p) > 0 );
    assert( this->a0_T_a.count(p) > 0 );

    assert( this->normed_uv_a.count(p) > 0 );
    assert( this->d_a.count(p) > 0 );
    assert( this->a0_T_a.count(p) > 0 );

    assert( this->all_pair_idx.count(p) > 0 );


    obj["data"] = json();

    int N_im_pairs = (int)this->a0_T_a.at(p).size();
    for( int im_pair_i = 0 ; im_pair_i < N_im_pairs ; im_pair_i++ )
    {
        json tmp;
        tmp["idx_a"] = all_pair_idx.at(p).at( im_pair_i ).first;
        tmp["idx_b"] = all_pair_idx.at(p).at( im_pair_i ).second;
        tmp["npoint_feature_matches"] = normed_uv_a.at(p).at( im_pair_i ).cols();


        assert( normed_uv_a.at(p).at( im_pair_i ).cols() == normed_uv_b.at(p).at( im_pair_i ).cols() );
        assert( d_a.at(p).at( im_pair_i ).size() == d_a.at(p).at( im_pair_i ).size() );
        // uv_a <---> uv_b
        tmp["normed_uv_a"] = RawFileIO::write_eigen_matrix_tojson(  normed_uv_a.at(p).at( im_pair_i ) );
        tmp["normed_uv_b"] = RawFileIO::write_eigen_matrix_tojson(  normed_uv_b.at(p).at( im_pair_i ) );

        // d_a <--> d_b
        tmp["d_a"] = RawFileIO::write_eigen_matrix_tojson(  d_a.at(p).at( im_pair_i ) );
        tmp["d_b"] = RawFileIO::write_eigen_matrix_tojson(  d_b.at(p).at( im_pair_i ) );


        // a0_T_a <---> b0_T_b
        tmp["a0_T_a"] = RawFileIO::write_eigen_matrix_tojson(  a0_T_a.at(p).at( im_pair_i ) );
        tmp["b0_T_b"] = RawFileIO::write_eigen_matrix_tojson(  b0_T_b.at(p).at( im_pair_i ) );


        obj["data"].push_back( tmp );
    }

    return obj;



}

void LocalBundle::toJSON(const string BASE) const
{
    // const string BASE = "/app/catkin_ws/src/gmm_pointcloud_align/resources/local_bundle/";

    // Save Sequences
    for( auto it = x0_T_c.begin() ; it != x0_T_c.end() ; it++ )
    {
        json tmp_i = odomSeqJ_toJSON( it->first );

        string fname = BASE+ "/odomSeq" + to_string(it->first) + ".json";
        cout << TermColor::iGREEN() << "Open File: " << fname << TermColor::RESET() << endl;
        std::ofstream o(fname);
        o << std::setw(4) << tmp_i << std::endl;
    }


    // Save matches
    for( auto it = all_pair_idx.begin() ; it != all_pair_idx.end() ; it++ )
    {
        auto p = it->first;
        json tmp_p = matches_SeqPair_toJSON( p.first, p.second );

        string fname = BASE+ "/seqPair_" + to_string(p.first) + "_" + to_string(p.second) + ".json";
        cout << TermColor::iGREEN() << "Open File: " << fname << TermColor::RESET() << endl;
        std::ofstream o(fname);
        o << std::setw(4) << tmp_p << std::endl;
    }
}

//---

bool LocalBundle::odomSeqJ_fromJSON( const string BASE, int j)
{
    // const string BASE = "/app/catkin_ws/src/gmm_pointcloud_align/resources/local_bundle/";
    // int j = 0;

    string fname = BASE + "/odomSeq" + std::to_string( j ) + ".json";
    cout << "Load: " << fname << endl;

    // read json
    std::ifstream json_file(fname);
    json obj;
    json_file >> obj;


    int seqID = obj["seqID"];
    cout << "seqID = " << seqID << endl;
    assert( seqID == j );

    int N = (int)obj["data"].size();
    cout << "There are " << N << " seq items\n";
    this->x0_T_c[seqID] = vector<Matrix4d>();
    this->seq_x_idx[seqID] = vector<int>();
    for( int i=0 ; i<N ; i++ )
    {
        json tmp_pose = obj["data"][i]["c0_T_c"];
        Matrix4d c0_T_c;
        RawFileIO::read_eigen_matrix4d_fromjson( tmp_pose, c0_T_c );

        int idx = obj["data"][i]["idx"];
        cout << idx << "\t";
        cout << PoseManipUtils::prettyprintMatrix4d( c0_T_c ) << endl;
        cout << c0_T_c << endl;

        // set this data in `this`
        this->x0_T_c[seqID].push_back( c0_T_c );
        this->seq_x_idx[seqID].push_back( idx );

    }

    return true;
}

bool LocalBundle::matches_SeqPair_fromJSON( const string BASE, int seqa, int seqb )
{
    // const string BASE = "/app/catkin_ws/src/gmm_pointcloud_align/resources/local_bundle/";
    // int seqa = 0;
    // int seqb = 1;

    string fname = BASE + "/seqPair_" + std::to_string( seqa ) + "_" + std::to_string( seqb ) + ".json";
    cout << "Load: " << fname << endl;

    // read json
    std::ifstream json_file(fname);
    json obj;
    json_file >> obj;

    int json_seqa = obj["seq_a"];
    int json_seqb = obj["seq_b"];
    assert( seqa == json_seqa && seqb == json_seqb );
    auto pyp = std::make_pair( seqa, seqb );
    cout << "json_seqa=" << json_seqa<< "\tjson_seqb=" << json_seqb << endl;


    json tmp = obj["initial_guess____a0_T_b0"];
    a0_T_b0[pyp] = Matrix4d::Identity();
    RawFileIO::read_eigen_matrix4d_fromjson( tmp, a0_T_b0[pyp] );
    cout << "a0_T_b0[(" << pyp.first << "," << pyp.second  << ")]=" << a0_T_b0[pyp] << endl;


    //
    int N = (int)obj["data"].size();
    cout << "There are " << N << "data items\n";
    normed_uv_a[pyp] =  vector<MatrixXd> ();
    d_a[pyp] = vector<VectorXd> ();
    a0_T_a[pyp] = vector<Matrix4d>();

    normed_uv_b[pyp] =  vector<MatrixXd> ();
    d_b[pyp] = vector<VectorXd> ();
    b0_T_b[pyp] = vector<Matrix4d>();

    all_pair_idx[pyp] =  vector< std::pair<int,int> >();

    for( int i=0 ; i<N ; i++ )
    {
        json tmp_data = obj["data"][i];

        Matrix4d tmp_a0_T_a;
        RawFileIO::read_eigen_matrix4d_fromjson( tmp_data["a0_T_a"], tmp_a0_T_a );

        Matrix4d tmp_b0_T_b;
        RawFileIO::read_eigen_matrix4d_fromjson( tmp_data["b0_T_b"], tmp_b0_T_b );

        int idx_a = tmp_data["idx_a"];
        int idx_b = tmp_data["idx_b"];


        VectorXd tmp_d_a;
        RawFileIO::read_eigen_vector_fromjson( tmp_data["d_a"], tmp_d_a );

        VectorXd tmp_d_b;
        RawFileIO::read_eigen_vector_fromjson( tmp_data["d_b"], tmp_d_b );


        MatrixXd tmp_normed_uv_a;
        RawFileIO::read_eigen_matrix_fromjson( tmp_data["normed_uv_a"], tmp_normed_uv_a );

        MatrixXd tmp_normed_uv_b;
        RawFileIO::read_eigen_matrix_fromjson( tmp_data["normed_uv_b"], tmp_normed_uv_b );


        cout << "i=" << i << "\t";
        cout << idx_a << "<--->" << idx_b << "\t";
        cout << "d_a, d_b = " << tmp_d_a.size() << ", " << tmp_d_b.size() << "\t";
        cout << "uv_a=" << tmp_normed_uv_a.rows() << "x" << tmp_normed_uv_a.cols() << "\t";
        cout << "uv_b=" << tmp_normed_uv_a.rows() << "x" << tmp_normed_uv_a.cols() << "\t";
        cout << endl;


        // set this data in `this`
        all_pair_idx[pyp].push_back( make_pair( idx_a, idx_b) );

        normed_uv_a[pyp].push_back( tmp_normed_uv_a );
        d_a[pyp].push_back( tmp_d_a );
        a0_T_a[pyp].push_back( tmp_a0_T_a );

        normed_uv_b[pyp].push_back( tmp_normed_uv_b );
        d_b[pyp].push_back( tmp_d_b );
        b0_T_b[pyp].push_back( tmp_b0_T_b );


    }

}

void LocalBundle::fromJSON( const string BASE )
{
    // TODO ideally should scan this directory, but itz ok.
    odomSeqJ_fromJSON(BASE, 0);
    odomSeqJ_fromJSON(BASE, 1);
    matches_SeqPair_fromJSON(BASE, 0, 1);
}
//---

void LocalBundle::print_inputs_info() const
{
    cout << TermColor::GREEN() << "-----------------------------------\n---LocalBundle::print_inputs_info---\n---------------------------------------\n";
    cout << "There are " << x0_T_c.size() << " sequences\n";
    vector<char> char_list = { 'a', 'b', 'c', 'd', 'e', 'f' };

    int i=0;
    for( auto it=x0_T_c.begin() ; it!=x0_T_c.end() ; it++ )
    {
        cout << "Seq#" << it->first  << ":\n";
        cout << char_list[i] << 0 << "  , " << char_list[i] << 1 << " ... " << char_list[i] << it->second.size() << endl;
        cout << *( seq_x_idx.at( it->first ).begin() ) << "," << *( seq_x_idx.at( it->first ).begin()+1 ) << " ... " << *( seq_x_idx.at( it->first ).rbegin() ) << endl;
        cout << endl;
        i++;
    }


    #if 0
    // too much details
    for( auto it=seq_x_idx.begin() ; it!=seq_x_idx.end() ; it++ )
    {
        auto Y = it->second;
        cout << "\nSeq#" << it->first << ": ";
        for( auto ity = Y.begin() ; ity!= Y.end() ; ity++ )
            cout << *ity << "\t";
        cout << endl;

    }
    #endif


    cout << "There are " << all_pair_idx.size() << " seq-pairs\n";
    for( auto it = all_pair_idx.begin() ; it != all_pair_idx.end() ; it++ )
    {
        cout << "seq-pair#" << std::distance( it , all_pair_idx.begin() ) << " between seq#" << it->first.first << " and seq#" << it->first.second << endl; ;

        #if 0
        auto y = it->second;
        for( auto itz=y.begin() ; itz!=y.end() ; itz++ )
            cout << itz->first << "<--->" << itz->second << "\n";
        cout << endl;
        #endif

        auto y = it->second;
        auto pyp = it->first;
        for( int k=0 ; k<y.size() ; k++ ) {
            cout << "\t#" << k << "\t";
            cout << all_pair_idx.at(pyp).at( k ).first << "<--->" << all_pair_idx.at(pyp).at( k ).first<< "\t";
            cout << normed_uv_a.at(pyp).at( k ).rows() << "x" << normed_uv_a.at(pyp).at( k ).cols() << "\t";
            cout << normed_uv_b.at(pyp).at( k ).rows() << "x" << normed_uv_b.at(pyp).at( k ).cols() << "\t";
            cout << d_a.at(pyp).at(k).size() << "\t" <<  d_b.at(pyp).at(k).size() << "\t";
            a0_T_a.at(pyp).at(k);
            b0_T_b.at(pyp).at(k);

            cout << endl;
        }


    }

    cout << TermColor::GREEN() << "-----------------------------------\n---END LocalBundle::print_inputs_info---\n---------------------------------------\n" << TermColor::RESET();
}



//----------------------------------------------------------------------------//
//      END print, json IO
//----------------------------------------------------------------------------//





void LocalBundle::solve()
{

}
