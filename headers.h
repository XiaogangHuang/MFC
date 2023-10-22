#include <limits.h>
#include <Eigen/Dense>
#include "nanoflann.hpp"
#include "cyw_timer.h"
#include "DisjSet.h"
#include "fileIO.h"

using namespace std;
using namespace Eigen;

double sqr_distance(vector<float>& v1, vector<float>& v2);

void printMatrix(MatrixXd& dataSets);

int vectorsIntersection(vector<int> v1, vector<int> v2);

