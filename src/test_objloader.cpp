#include <iostream>
using namespace std;

#include "utils/MeshObject.h"

int main()
{
    cout << "Helloworld\n";

    MeshObject obj( "bunny.obj", 1.0 );

    auto w_X = obj.getVertices();
    cout << "vertices dims: " << w_X.rows() << "x" << w_X.cols() << endl;
    cout << w_X.leftCols(10) << endl;
    return 0;
}
