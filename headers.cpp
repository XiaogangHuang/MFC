#include "headers.h"

double sqr_distance(vector<float>& v1, vector<float>& v2)
{
    int dim = v1.size();
    double sum = 0.0;
    for(int i = 0; i < dim;i++)
    {
        double d1 =v1[i] - v2[i];
        d1 *= d1;
        sum = sum+d1;
    }
    return sum;
}

void printMatrix(MatrixXd& dataSets)
{
    int dim = dataSets.cols();
    for (size_t i = 0; i < 20; i++)
    {
        for (size_t j = 0; j < dim; j++)
        {
            printf("%.2f ", dataSets(i,j));
        }
        printf("\n");
    }
}

int vectorsIntersection(vector<int> v1, vector<int> v2) {
    vector<int> v;
    sort(v1.begin(), v1.end());
    sort(v2.begin(), v2.end());
    set_intersection(v1.begin(), v1.end(), v2.begin(), v2.end(), back_inserter(v));//Çó½»¼¯ 
    return v.size();
}
