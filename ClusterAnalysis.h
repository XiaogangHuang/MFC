#pragma once
#include "headers.h"

struct State {
    double density;
    double kdist;
    double density_mean;
    double density_std;
    int pioneer;
    int clusterId;
    int clusterID;
    /*double naive;
    double naive_mean;
    double naive_std;
    double refined;*/
};

//Cluster analysis type.
class ClusterAnalysis
{
private:
    MatrixXd dataSets;
    nanoflann::KDTreeEigenMatrixAdaptor< MatrixXd >* node_data_kd;
    //vector<vector<vector<int>>> Border_of_Clusters;
    vector< vector< double > > SaddlePoints;
    vector< vector< double > > ClusterTree;
    vector< vector< int > > PNEI;
    vector< vector< int > > RNN;
    vector< vector< int > > MNN;
    vector< vector< int > > AdjClust;
    vector< vector< int > > AdjPts;
    vector< vector< int > > MERGE;
    vector< double >  out_dists_sqr;
    vector< size_t > ret_indexes;
    vector< int > ClusterCenters;
    State* state;
    /*double minPhi;
    double maxPhi;*/
    double preference;
    int dataNum;
    int dataDim;
    int K;
    int clusterID;
    int MinSize;
    int clu_num;
    int lambda;
    int alg;
    int o_tree;
public:
    ClusterAnalysis() {}            //Default constructor.
	~ClusterAnalysis() {
        free(state);
        delete node_data_kd;							
	}                              // Destructor

    bool Init(char* fileName, int K, int alg, int clu_num, int r, int o_tree, double pre);    //Initialization operation
    void Running();
    void Normalization();
    double Density(vector< double >& out_dists_sqr);
    void SetArrivalPoints();
    void FindMNN();
    int IntersectionCount(vector<int>& mark, int i);
    /*double IntersectionCount(vector<int>& mark, int i, int j);
    double IntersectionCount(vector<int>& mark, vector<double>& dist_vec, int i, int j);*/
    /*void CalculateNaive();
    void CalculateRefined();*/
    bool FindPioneers();
    bool FindPioneers_MNN();
    void GenerateCusters();
    void GenerateCusters_MNN();
    void GenerateCusters_SNNDPC();
    //double CalculateRefined(vector<double>& pt, vector<int>& mark);
    double CalculateDensity(vector< double >& pt);
    void FindSaddlePoints();
    void CalculateConsolidation(vector< double >& scores);
    int ConstructClusterTree();

    void SetArrivalPoints_SNN();
    double EuclideanDist(vector<double>& neis, int pt);
    double SNNLocalDensity(vector<double>& coord, vector<int>& neis, vector<int>& mark);
    double SNNLocalDensity(vector<int>& neis, vector<int>& mark, int j);
    void FindSaddlePoints_SNN();
    double CalculateDensity_SNN(vector<double>& pt);

    bool Init(char* fileName, int nc);    //Initialization operation
    void Running_LDP_MST();
    void Parameters_LDP_MST();
    bool FindPioneers_LDP_MST();
    void GenerateCusters_LDP_MST();
    void CalculateSND(vector< vector< int > >& MST);
    int iscontain(vector<int>& q, int front, int rear, int x);
    void ExtractClusters();

    bool WriteToFile();    //save results
};
