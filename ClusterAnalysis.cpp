#pragma warning(disable:4996)
#include "ClusterAnalysis.h"


bool ClusterAnalysis::Init(char* fileName, int K, int alg, int clu_num, int r, int o_tree, double pre) {
    this->preference = pre;
    this->K = K;                            //set the K
    this->clu_num = clu_num;
    this->alg = alg;
    this->o_tree = o_tree;

    out_dists_sqr.resize(K);
    ret_indexes.resize(K);

    int dim;
    int data_size;
    cout<<"reading data..." << endl;
    double *raw_data=read_data((char*)fileName,(char*)" ",&dim, &data_size);
    dataNum = data_size;
    dataDim = dim - 1;
    dataSets.resize(dataNum, dataDim);
    for (size_t i = 0; i < dataNum; i++) {
        for (size_t j = 0; j < dataDim; j++)
            dataSets(i, j) = raw_data[dim * i + j];
    }
    free(raw_data);
    if (r != 0)
    {
        Normalization();
    }

    state = (State*)malloc(data_size * sizeof(State));
    if (state != NULL)
    {
        for (size_t i = 0; i < dataNum; i++)
        {
            state[i].pioneer = -2;
            state[i].clusterId = -1;
            state[i].clusterID = -1;
        }
    }
    //printMatrix(dataSets);

    CYW_TIMER build_timer;
    build_timer.start_my_timer();
    cout << "building the trees...\n";
    node_data_kd = new nanoflann::KDTreeEigenMatrixAdaptor<MatrixXd>(dataDim, dataSets, 10);
    build_timer.stop_my_timer();

    //printMatrix(dataSets);
    printf("n = %d  dim = %d\n", dataNum, dataDim);
    printf("kd-tree build time = %.4f\n", build_timer.get_my_timer());
    MinSize = floor(dataNum * 0.16 / pow(log10(dataNum), 2));
    //MinSize = 0;
    //MinSize = dataNum * 0.03;
    printf("MinSize: %d\n", MinSize);
    return true;
}

void ClusterAnalysis::Running(){
    if (alg == 0)
    {
        SetArrivalPoints();             // Find point's epsilon neighborhood.
        FindPioneers();                 // Find point's pioneer.
        GenerateCusters();              // Form initial clusters.
        if (clusterID > 1)
        {
            FindSaddlePoints();             // Find saddle points.
            ConstructClusterTree();         // Form cluster tree.
        }
    }
    else if (alg == 1)
    {
        SetArrivalPoints();             // Find point's epsilon neighborhood.
        FindPioneers_MNN();             // Find point's pioneer.
        GenerateCusters_MNN();          // Form initial clusters.
        //GenerateCusters_SNNDPC();     // Form initial clusters.
        FindSaddlePoints();             // Find saddle points.
        ConstructClusterTree();         // Form cluster tree.
    }
    else
    {
        SetArrivalPoints_SNN();         // Find point's epsilon neighborhood.
        FindPioneers();                 // Find point's pioneer.
        GenerateCusters();              // Form initial clusters.
        FindSaddlePoints_SNN();             // Find saddle points.
        ConstructClusterTree();         // Form cluster tree.
    }
}

void ClusterAnalysis::Normalization()
{
    vector<double> minCoord(dataDim);
    vector<double> maxCoord(dataDim);
    for (size_t i = 0; i < dataDim; i++)
    {
        minCoord[i] = dataSets(0, i);
        maxCoord[i] = dataSets(0, i);
    }

    for (size_t i = 1; i < dataNum; i++)
    {
        for (size_t j = 0; j < dataDim; j++)
        {
            double temp = dataSets(i, j);
            if (temp > maxCoord[j])
            {
                maxCoord[j] = temp;
            }
            else if (temp < minCoord[j])
            {
                minCoord[j] = temp;
            }
        }
    }
    for (size_t i = 0; i < dataNum; i++)
    {
        for (size_t j = 0; j < dataDim; j++)
        {
            if (maxCoord[j] - minCoord[j] == 0)
            {
                dataSets(i, j) = 0;
            }
            else
            {
                dataSets(i, j) = (dataSets(i, j) - minCoord[j]) / (maxCoord[j] - minCoord[j]);
            }
        }
    }
}

double ClusterAnalysis::Density(vector< double >&  out_dists_sqr) {
    double density = 0.0;
    for (size_t jt = 0; jt < K; jt++)
    {
        density += sqrt(out_dists_sqr[jt]);
    }
    return K / density;
}

void ClusterAnalysis::SetArrivalPoints(){
    vector<double> pt(dataDim);
    for (int it = 0; it < dataNum; it++)
    {
        for (int jt = 0; jt < dataDim; jt++)
        {
            pt[jt] = dataSets(it, jt);
        }
        nanoflann::KNNResultSet<double> resultSet(K);
        resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
        node_data_kd->index_->findNeighbors(resultSet, &pt[0]);

        vector<int> nei;
        nei.reserve(K);
        for(int i = 0; i < K; i++)
        {
            nei.push_back(ret_indexes[i]);
        }
        PNEI.push_back(move(nei));
        state[it].density = Density(out_dists_sqr);
        state[it].kdist = sqrt(out_dists_sqr[K - 1]);
    }
    FindMNN();

    for (size_t i = 0; i < dataNum; i++)
    {
        double sum = 0.0, std = 0.0;
        for (size_t j = 0; j < PNEI[i].size(); j++)
        {
            int nei = PNEI[i][j];
            double den = state[nei].density;
            sum += den;
            std += den * den;
        }
        state[i].density_mean = sum / K;
        state[i].density_std = sqrt((std - sum * state[i].density_mean) / (K - 1));
    }

    //CalculateNaive();
    //CalculateRefined();
}

void ClusterAnalysis::FindMNN()
{
    RNN.resize(dataNum);
    for (size_t i = 0; i < dataNum; i++)
    {
        vector<int>& knn = PNEI[i];
        for (size_t j = 0; j < knn.size(); j++)
        {
            RNN[knn[j]].push_back(i);
        }
    }

    vector<int> mark(dataNum, 0);
    for (size_t i = 0; i < dataNum; i++)
    {
        vector<int> vec;
        for (size_t j = 0; j < RNN[i].size(); j++)
        {
            mark[RNN[i][j]] = 1;
        }
        for (size_t j = 0; j < PNEI[i].size(); j++)
        {
            if (mark[PNEI[i][j]] == 1)
            {
                vec.push_back(PNEI[i][j]);
            }
        }
        MNN.push_back(move(vec));
        for (size_t j = 0; j < RNN[i].size(); j++)
        {
            mark[RNN[i][j]] = 0;
        }
    }
}

int ClusterAnalysis::IntersectionCount(vector<int>& mark, int i)
{
    int count = 0;
    for (size_t it = 0; it < PNEI[i].size(); it++)
    {
        if (mark[PNEI[i][it]] != 0)
        {
            count++;
        }
    }
    return count;
}

//double ClusterAnalysis::IntersectionCount(vector<int>& mark, int i, int j)
//{
//    double x = 0.0;
//    int count = 0;
//    for (size_t it = 0; it < PNEI[i].size(); it++)
//    {
//        int pt = PNEI[i][it];
//        if (mark[pt] > 0)
//        {
//            count++;
//            x += (dataSets.row(i) - dataSets.row(pt)).norm() + (dataSets.row(j) - dataSets.row(pt)).norm();
//        }
//    }
//    if (count > 0)
//    {
//        return x / count;
//    }
//    return 2 * (dataSets.row(i) - dataSets.row(j)).norm();
//}
//
//double ClusterAnalysis::IntersectionCount(vector<int>& mark, vector<double>& dist_vec, int i, int j)
//{
//    double x = 0.0;
//    int count = 0;
//    for (size_t it = 0; it < PNEI[i].size(); it++)
//    {
//        int pt = PNEI[i][it];
//        if (mark[pt] > 0)
//        {
//            count++;
//            x += (dataSets.row(i) - dataSets.row(pt)).norm() + sqrt(dist_vec[mark[pt]]);
//        }
//    }
//    if (count > 0)
//    {
//        return x / count;
//    }
//    return 2 * sqrt(dist_vec[j]);
//}

//void ClusterAnalysis::CalculateNaive()
//{
//    vector<int> mark(dataNum, 0);
//    for (size_t i = 0; i < dataNum; i++)
//    {
//        for (size_t j = 0; j < PNEI[i].size(); j++)
//        {
//            mark[PNEI[i][j]] = 1;
//        }
//        double density = 0.0;
//        for (size_t j = 0; j < PNEI[i].size(); j++)
//        {
//            int nei = PNEI[i][j];
//            density += IntersectionCount(mark, nei, i) * (dataSets.row(i) - dataSets.row(nei)).norm();
//        }
//        state[i].naive = density / state[i].sumdist;
//        for (size_t j = 0; j < PNEI[i].size(); j++)
//        {
//            mark[PNEI[i][j]] = 0;
//        }
//    }
//}
//
//void ClusterAnalysis::CalculateRefined()
//{
//    for (size_t i = 0; i < dataNum; i++)
//    {
//        double mean = 0.0, std = 0.0;
//        for (size_t j = 0; j < PNEI[i].size(); j++)
//        {
//            int nei = PNEI[i][j];
//            mean += state[nei].naive;
//            std += state[nei].naive * state[nei].naive;
//        }
//        state[i].naive_mean = mean / K;
//        state[i].naive_std = sqrt((std - mean * state[i].naive_mean) / (K - 1));
//    }
//
//    vector<double> Tau(dataNum);
//    vector<double> Phi(dataNum);
//    vector<int> mark(dataNum, 0);
//    for (size_t i = 0; i < dataNum; i++)
//    {
//        for (size_t j = 0; j < PNEI[i].size(); j++)
//        {
//            mark[PNEI[i][j]] = 1;
//        }
//        int count = 0;
//        for (size_t j = 0; j < PNEI[i].size(); j++)
//        {
//            int nei = PNEI[i][j];
//            if (abs(state[i].naive - state[nei].naive_mean) <= 3 * state[nei].naive_std && 
//                abs(state[nei].naive - state[i].naive_mean) <= 3 * state[i].naive_std &&
//                IntersectionCount(mark, nei) >= K / 2)
//            {
//                count++;
//            }
//        }
//        Tau[i] = 1.0 * count / K;
//        Phi[i] = 1.0 * K * count / state[i].sumdist;
//        for (size_t j = 0; j < PNEI[i].size(); j++)
//        {
//            mark[PNEI[i][j]] = 0;
//        }
//    }
//    minPhi = *min_element(Phi.begin(), Phi.end());
//    maxPhi = *max_element(Phi.begin(), Phi.end()) - minPhi;
//    for (size_t i = 0; i < dataNum; i++)
//    {
//        state[i].refined = 1 / (state[i].naive * exp(-Tau[i] * (Phi[i] - minPhi) / maxPhi));
//    }
//}

bool ClusterAnalysis::FindPioneers(){
    /*for (int it = 0; it < dataNum; it++)
    {
        double mindist = DBL_MAX;
        int pioneer = -1;
        for  (int jt = 0; jt < PNEI[it].size(); jt++)
        {
            int neiId = PNEI[it][jt];
            if (state[it].density < state[neiId].density)
            {
                double dist = (dataSets.row(it) - dataSets.row(neiId)).norm();
                if (mindist > dist)
                {
                    mindist = dist;
                    pioneer = neiId;
                }
            }
        }
        state[it].pioneer = pioneer;
    }*/

    for (int it = 0; it < dataNum; it++)
    {
        int pioneer = -1;
        for (int jt = 0; jt < PNEI[it].size(); jt++)
        {
            int neiId = PNEI[it][jt];
            if ((state[it].density < state[neiId].density || (state[it].density == state[neiId].density &&
                state[neiId].pioneer == -1/* && (dataSets.row(it) - dataSets.row(neiId)).norm() == 0*/)))
            {
                pioneer = neiId;
                break;
            }
        }
        state[it].pioneer = pioneer;
    }
    return true;

    /*for (int it = 0; it < dataNum; it++)
    {
        double max = state[it].density;
        state[it].pioneer = -1;
        for (int jt = 0; jt < PNEI[it].size(); jt++)
        {
            int neiId = PNEI[it][jt];
            if (max < state[neiId].density)
            {
                max = state[neiId].density;
                state[it].pioneer = neiId;
            }
        }
    }
    return true;*/
}

bool ClusterAnalysis::FindPioneers_MNN() {
    for (int it = 0; it < dataNum; it++)
    {
        int pioneer = -1;
        for (int jt = 0; jt < MNN[it].size(); jt++)
        {
            int neiId = MNN[it][jt];
            if (state[it].density < state[neiId].density || (state[it].density == state[neiId].density && 
                state[neiId].pioneer == -1/* && (dataSets.row(it) - dataSets.row(neiId)).norm() == 0*/))
            {
                pioneer = neiId;
                break;
            }
        }
        state[it].pioneer = pioneer;
    }

    return true;
}

void ClusterAnalysis::GenerateCusters(){
    // Point are traversed in descending order according to their densities.
    vector<int> order(dataNum);
    for (int it = 0; it < dataNum; it++)
        order[it] = it;
    State* temp = state;
    sort(order.begin(), order.end(),
        [temp](int a, int b) {return temp[a].density > temp[b].density; });
    clusterID = 0;
    for (int it = 0; it < dataNum; it++)
    {
        int processedPt = order[it];
        int temp = processedPt;
        if (state[processedPt].pioneer != -1)
        {
            temp = state[processedPt].pioneer;
        }
        if (state[temp].clusterId == -1)
        {
            ClusterCenters.push_back(temp);
            state[temp].clusterId = clusterID++;
        }
        state[processedPt].clusterId = state[temp].clusterId;
    }

    // Searching for unstable areas
    vector<vector<int>> votes(dataNum, vector<int>(clusterID, 0));
    vector<int> mark1(dataNum, 0);
    vector<int> unstableSet;
    for (int it = 0; it < dataNum; it++)
    {
        if (state[it].pioneer != -1)
        {
            for (size_t j = 0; j < PNEI[it].size(); j++)
            {
                int nei = PNEI[it][j];
                votes[it][state[nei].clusterId]++;
            }
            if (2 * votes[it][state[it].clusterId] <= K)
            {
                unstableSet.push_back(it);
                mark1[it] = 1;
            }
        }
    }

    /*vector< vector< int > > PCLUSTER;
    PCLUSTER.resize(clusterID);
    for (int it = 0; it < dataNum; it++)
    {
        int clust = state[it].clusterId;
        PCLUSTER[clust].push_back(it);
    }
    vector< int > mark(dataNum, 0);
    for (int it = 0; it < clusterID; it++)
    {
        for (size_t i = 0; i < PCLUSTER[it].size(); i++)
        {
            int pnt = PCLUSTER[it][i];
            for (size_t jt = 0; jt < PNEI[pnt].size(); jt++)
            {
                mark[PNEI[pnt][jt]] = 1;
            }
            printf("%3d %3d:", pnt, mark1[pnt]);
            vector<int>& temp = PNEI[pnt];
            for (size_t jt = 0; jt < temp.size(); jt++)
            {
                printf("  %3d ", temp[jt]);
            }
            printf("\n        ");
            for (size_t jt = 0; jt < temp.size(); jt++)
            {
                printf("  %3d ", state[temp[jt]].clusterId);
            }
            printf("\n        ");
            for (size_t jt = 0; jt < temp.size(); jt++)
            {
                printf("  %3d ", state[temp[jt]].pioneer);
            }
            printf("\n        ");
            for (size_t jt = 0; jt < temp.size(); jt++)
            {
                printf("  %3d ", IntersectionCount(mark, temp[jt]));
            }
            printf("\n        ");
            for (size_t jt = 0; jt < temp.size(); jt++)
            {
                printf(" %.2lf ", state[temp[jt]].density);
            }
            printf("\n");
            for (size_t jt = 0; jt < PNEI[pnt].size(); jt++)
            {
                mark[PNEI[pnt][jt]] = 0;
            }
        }
    }*/

    for (size_t i = 0; i < unstableSet.size(); i++)
    {
        int processedPt = unstableSet[i];
        vector<double> votes(clusterID, 0);
        for (size_t j = 0; j < PNEI[processedPt].size(); j++)
        {
            int pt = PNEI[processedPt][j];
            if (pt == processedPt)
            {
                votes[state[pt].clusterId] += 0.5;
            }
            else if (mark1[pt] != 1)
            {
                votes[state[pt].clusterId] += 1;
            }
        }
        int maxVote = max_element(votes.begin(), votes.end()) - votes.begin();
        if (votes[maxVote] > 0 && maxVote != state[processedPt].clusterId)
        {
            for (size_t j = 0; j < PNEI[processedPt].size(); j++)
            {
                int pt = PNEI[processedPt][j];
                if (state[pt].clusterId == maxVote)
                {
                    state[processedPt].pioneer = pt;
                    break;
                }
            }
            state[processedPt].clusterId = maxVote;
        }
        mark1[processedPt] = 0;
    }

    for (size_t i = 0; i < dataNum; i++)
    {
        int pioneer = state[i].pioneer;
        if (pioneer != -1 && state[i].clusterId != state[pioneer].clusterId)
        {
            for (size_t j = 0; j < PNEI[i].size(); j++)
            {
                int pt = PNEI[i][j];
                if (pt != i && state[pt].clusterId == state[i].clusterId)
                {
                    state[i].pioneer = pt;
                    break;
                }
            }
        }
    }

    printf("There are %d initial clusters.\n", clusterID);
}

void ClusterAnalysis::GenerateCusters_MNN() {
    // Point are traversed in descending order according to their densities.
    vector<int> order(dataNum);
    for (int it = 0; it < dataNum; it++)
        order[it] = it;
    State* temp = state;
    sort(order.begin(), order.end(),
        [temp](int a, int b) {return temp[a].density > temp[b].density; });
    clusterID = 0;
    for (int it = 0; it < dataNum; it++)
    {
        int processedPt = order[it];
        int temp = processedPt;
        if (state[processedPt].pioneer != -1)
        {
            temp = state[processedPt].pioneer;
        }
        if (state[temp].clusterId == -1)
        {
            ClusterCenters.push_back(temp);
            state[temp].clusterId = clusterID++;
        }
        state[processedPt].clusterId = state[temp].clusterId;
    }

    // Searching for unstable areas
    vector<vector<int>> votes(dataNum, vector<int>(clusterID, 0));
    vector<int> mark1(dataNum, 0);
    vector<int> unstableSet;
    for (int it = 0; it < dataNum; it++)
    {
        for (size_t j = 0; j < PNEI[it].size(); j++)
        {
            int nei = PNEI[it][j];
            votes[it][state[nei].clusterId]++;
        }
        if (2 * votes[it][state[it].clusterId] <= K)
        {
            unstableSet.push_back(it);
            mark1[it] = 1;
        }
    }

    /*vector< vector< int > > PCLUSTER;
    PCLUSTER.resize(clusterID);
    for (int it = 0; it < dataNum; it++)
    {
        int clust = state[it].clusterId;
        PCLUSTER[clust].push_back(it);
    }
    vector< int > mark(dataNum, 0);
    for (int it = 0; it < clusterID; it++)
    {
        for (size_t i = 0; i < PCLUSTER[it].size(); i++)
        {
            int pnt = PCLUSTER[it][i];
            for (size_t jt = 0; jt < PNEI[pnt].size(); jt++)
            {
                mark[PNEI[pnt][jt]] = 1;
            }
            printf("%3d %3d:", pnt, mark1[pnt]);
            vector<int>& temp = PNEI[pnt];
            for (size_t jt = 0; jt < temp.size(); jt++)
            {
                printf("  %3d ", temp[jt]);
            }
            printf("\n        ");
            for (size_t jt = 0; jt < temp.size(); jt++)
            {
                printf("  %3d ", state[temp[jt]].clusterId);
            }
            printf("\n        ");
            for (size_t jt = 0; jt < temp.size(); jt++)
            {
                printf("  %3d ", state[temp[jt]].pioneer);
            }
            printf("\n        ");
            for (size_t jt = 0; jt < temp.size(); jt++)
            {
                printf("  %3d ", IntersectionCount(mark, temp[jt]));
            }
            printf("\n        ");
            for (size_t jt = 0; jt < temp.size(); jt++)
            {
                printf(" %.2lf ", state[temp[jt]].density);
            }
            printf("\n");
            for (size_t jt = 0; jt < PNEI[pnt].size(); jt++)
            {
                mark[PNEI[pnt][jt]] = 0;
            }
        }
    }*/

    for (size_t i = 0; i < unstableSet.size(); i++)
    {
        int processedPt = unstableSet[i];
        vector<double> votes(clusterID, 0);
        for (size_t j = 0; j < PNEI[processedPt].size(); j++)
        {
            int pt = PNEI[processedPt][j];
            if (pt == processedPt)
            {
                votes[state[pt].clusterId] += 0.5;
            }
            else if (mark1[pt] != 1 &&
                abs(state[pt].density_mean - state[processedPt].density) <= 3 * state[pt].density_std)
            {
                votes[state[pt].clusterId] += 1;
            }
        }
        int maxVote = max_element(votes.begin(), votes.end()) - votes.begin();
        if (votes[maxVote] > 0)
        {
            state[processedPt].clusterId = maxVote;
        }
        mark1[processedPt] = 0;
    }

    printf("There are %d initial clusters.\n", clusterID);
}

void ClusterAnalysis::GenerateCusters_SNNDPC() {
    // Point are traversed in descending order according to their densities.
    clusterID = 0;
    vector<int> queue;
    for (int it = 0; it < dataNum; it++)
    {
        if (state[it].pioneer == -1)
        {
            ClusterCenters.push_back(it);
            queue.push_back(it);
            state[it].clusterId = clusterID++;
        }
    }
    vector<int> mark(dataNum, 0);
    int pos = 0;
    while (pos != queue.size())
    {
        int temp = queue[pos++];
        for (size_t i = 0; i < PNEI[temp].size(); i++)
        {
            mark[PNEI[temp][i]] = 1;
        }
        for (size_t i = 0; i < PNEI[temp].size(); i++)
        {
            int nei = PNEI[temp][i];
            if (state[nei].clusterId == -1 && IntersectionCount(mark, nei) >= K / 2)
            {
                state[nei].clusterId = state[temp].clusterId;
                queue.push_back(nei);
            }
        }
        for (size_t i = 0; i < PNEI[temp].size(); i++)
        {
            mark[PNEI[temp][i]] = 0;
        }
    }

    int cond = 1;
    while (cond == 1)
    {
        int temp = 0;
        for (size_t i = 0; i < dataNum; i++)
        {
            if (state[i].clusterId == -1)
            {
                vector<int> votes(clusterID, 0);
                for (auto qt : PNEI[i])
                {
                    if (state[qt].clusterId > 0)
                    {
                        votes[state[qt].clusterId]++;
                    }
                }
                int maxVote = max_element(votes.begin(), votes.end()) - votes.begin();
                if (state[i].clusterId != maxVote)
                {
                    state[i].clusterId = maxVote;
                    temp = 1;
                }
            }
        }
        cond = temp;
    }


    printf("There are %d initial clusters.\n", clusterID);
}

//double ClusterAnalysis::CalculateRefined(vector< double >& pt, vector< int >& mark) {
//    nanoflann::KNNResultSet<double> resultSet(K);
//    resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
//    node_data_kd->index_->findNeighbors(resultSet, &pt[0]);
//
//    for (size_t j = 0; j < K; j++)
//    {
//        mark[ret_indexes[j]] = j;
//    }
//    double density = 0.0;
//    double sum = 0.0;
//    for (size_t j = 0; j < K; j++)
//    {
//        int nei = ret_indexes[j];
//        density += IntersectionCount(mark, out_dists_sqr, nei, j) * sqrt(out_dists_sqr[j]);
//        sum += sqrt(out_dists_sqr[j]);
//    }
//    double naive = density / sum;
//
//    double naive_mean = 0.0, naive_std = 0.0;
//    for (size_t j = 0; j < K; j++)
//    {
//        int nei = ret_indexes[j];
//        naive_mean += state[nei].naive;
//        naive_std += state[nei].naive * state[nei].naive;
//    }
//    naive_mean = naive_mean / K;
//    naive_std = sqrt((naive_std - naive_mean * naive_mean / K) / (K - 1));
//
//    int count = 0;
//    for (size_t j = 0; j < K; j++)
//    {
//        int nei = ret_indexes[j];
//        if (abs(naive- state[nei].naive_mean) <= 3 * state[nei].naive_std &&
//            abs(state[nei].naive - naive_mean) <= 3 * naive_std &&
//            IntersectionCount(mark, nei) >= K / 2)
//        {
//            count++;
//        }
//    }
//    double Tau = 1.0 * count / K;
//    double Phi = 1.0 * K * count / sum;
//
//    for (size_t j = 0; j < K; j++)
//    {
//        mark[ret_indexes[j]] = 0;
//    }
//    return 1 / (naive * exp(-Tau * (Phi - minPhi) / maxPhi));
//}

double ClusterAnalysis::CalculateDensity(vector< double >& pt){
    nanoflann::KNNResultSet<double> resultSet(K);
    resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
    node_data_kd->index_->findNeighbors(resultSet, &pt[0]);

    return Density(out_dists_sqr);
}

void ClusterAnalysis::FindSaddlePoints(){
    vector< vector< int > > PCLUSTER;
    PCLUSTER.resize(clusterID);
    for (int it = 0; it < dataNum; it++)
    {
        int clust = state[it].clusterId;
        PCLUSTER[clust].push_back(it);
    }

    vector< vector< int > > MinDist(dataNum, vector< int >(clusterID, -1));
    //vector< vector< int > > AdjPts;
 
    // 找边界点
    for (int it = 0; it < dataNum; it++)
    {
        int clust = state[it].clusterId;
        //vector<int> adjpts;
        for (int jt = 0; jt < K; jt++)
        {
            int adjpoint = PNEI[it][jt];
            int adj_cluster = state[adjpoint].clusterId;
            // 检查邻居数据点是否属于不同的簇并且是到这个簇中的点中距离最小的
            if (adj_cluster != clust && MinDist[it][adj_cluster] == -1)
            {
                // 将边界上的点(属于簇adj_cluster的it的近邻中离it最近的点)保存到MinDist中
                MinDist[it][adj_cluster] = adjpoint;
                //adjpts.push_back(adjpoint);
                //printf("%d ", adjpoint);
            }
        }
        //AdjPts.push_back(adjpts);
    }

    vector< vector< pair<int, double> > > SADDLE;
    vector< vector< vector< int > > > Borders;
    for (int it = 0; it < PCLUSTER.size(); it++)
    {
        // 找到相邻簇的边界点中密度最高的点
        vector< pair<int, double> > border(clusterID, {-1, 0.0});
        for (int jt = 0; jt < PCLUSTER[it].size(); jt++)
        {
            int pt = PCLUSTER[it][jt];
            for (int zt = 0; zt < clusterID; zt++)
            {
                int adjpt = MinDist[pt][zt];
                if (adjpt != -1 && state[adjpt].density > border[zt].second)
                {
                    border[zt].first = adjpt;
                    border[zt].second = state[adjpt].density;
                }
            }
        }
        SADDLE.push_back(move(border));
    }

    for (int it = 0; it < SADDLE.size(); it++)
    {
        for (int jt = 0; jt < SADDLE[it].size(); jt++)
        {
            int pt = SADDLE[it][jt].first;
            if (pt != -1)
            {
                // it簇与jt簇相邻，并且pt是jt簇中与it簇的边界点
                // pt是it簇中某一点的k近邻
                // 下面找it簇中距离pt最近的点，将其与pt的均值点作为it簇与jt簇的鞍点
                int adjpt = -1;
                for (size_t i = 0; i < PNEI[pt].size(); i++)
                {
                    int nei = PNEI[pt][i];
                    if (state[nei].clusterId == it)
                    {
                        adjpt = nei;
                        break;
                    }
                }
                if (adjpt == -1)
                {
                    double mindist = DBL_MAX;
                    for (int i = 0; i < PCLUSTER[it].size(); i++)
                    {
                        double dist = (dataSets.row(pt) - dataSets.row(PCLUSTER[it][i])).norm();
                        if (dist < mindist)
                        {
                            mindist = dist;
                            adjpt = PCLUSTER[it][i];
                        }
                    }
                    //AdjPts[pt].push_back(adjpt);
                }
                //printf("%d %d\n", pt, adjpt);
                // 计算鞍点的坐标和密度值
                vector< double > saddle;
                saddle.reserve(dataDim + 1);
                for (int i = 0; i < dataDim; i++)
                {
                    saddle.push_back((dataSets(pt, i) + dataSets(adjpt, i)) / 2);
                }
                /*printf("C1: %2d %3d %.3lf; C2: %2d %3d %.3lf; border: %3d %.3lf; adjpt: %3d %.3lf; %.3lf\n",
                    it, ClusterCenters[it], state[ClusterCenters[it]].density,
                    jt, ClusterCenters[jt], state[ClusterCenters[jt]].density,
                    pt, state[pt].density, adjpt, state[adjpt].density, CalculateDensity(saddle));*/
                //printf(" %d %d\n", adjpt, pt);
                saddle.push_back(CalculateDensity(saddle));
                SaddlePoints.push_back(move(saddle));
                AdjClust.push_back({ it, jt });
                AdjPts.push_back({ adjpt, pt });
            }
        }
    }
    //printf("\n\n");

    //Border_of_Clusters.resize(clusterID);
    //for (size_t i = 0; i < clusterID; i++)
    //{
    //    Border_of_Clusters[i].resize(clusterID);
    //}
    //for (size_t i = 0; i < clusterID; i++)
    //{
    //    for (size_t j = 0; j < PCLUSTER[i].size(); j++)
    //    {
    //        int curPoint = PCLUSTER[i][j];
    //        for (size_t it = 0; it < AdjPts[curPoint].size(); it++)
    //        {
    //            int adj_Clust = state[AdjPts[curPoint][it]].clusterId;
    //            Border_of_Clusters[i][adj_Clust].push_back(AdjPts[curPoint][it]);
    //        }
    //    }
    //    for (size_t j = 0; j < clusterID; j++)
    //    {
    //        sort(Border_of_Clusters[i][j].begin(), Border_of_Clusters[i][j].end());
    //        auto last = unique(Border_of_Clusters[i][j].begin(), Border_of_Clusters[i][j].end());
    //        Border_of_Clusters[i][j].erase(last, Border_of_Clusters[i][j].end());
    //    }
    //    /*printf("%d:\n", i);
    //    for (size_t j = 0; j < clusterID; j++)
    //    {
    //        if (Border_of_Clusters[i][j].size() > 0)
    //        {
    //            printf("%d: ", j);
    //        }
    //        for (size_t it = 0; it < Border_of_Clusters[i][j].size(); it++)
    //        {
    //            printf("%d ", Border_of_Clusters[i][j][it]);
    //        }
    //        if (Border_of_Clusters[i][j].size()>0)
    //        {
    //            printf("\n");
    //        }
    //    }*/
    //}

    /*ofstream of1("data\\edge.txt");
    for (size_t i = 0; i < clusterID; i++)
    {
        for (size_t j = 0; j < clusterID; j++)
        {
            for (size_t it = 0; it < Border_of_Clusters[i][j].size(); it++)
            {
                int adjPoint = Border_of_Clusters[i][j][it];
                int adj_Clust = state[adjPoint].clusterId;
                for (size_t jt = 0; jt < K; jt++)
                {
                    int nei = PNEI[adjPoint][jt];
                    if (state[nei].clusterId == i || state[nei].clusterId == adj_Clust)
                    {
                        of1 << adjPoint << " " << nei << endl;
                    }
                }
            }
        }
    }
    of1.close();*/

    /*for (int it = 0; it < dataNum; it++)
    {
        for (int jt = 0; jt < AdjPts[it].size(); jt++)
        {
            printf("%d %d\n", it, AdjPts[it][jt]);
        }
    }*/

    /*for (int it = 0; it < dataNum; it++)
    {
        printf("%d %d: ", it, state[it].clusterId);
        for (int jt = 0; jt < AdjPts[it].size(); jt++)
        {
            printf("%d %d, ", AdjPts[it][jt], state[AdjPts[it][jt]].clusterId);
        }
        printf("\n");
    }*/
}

void ClusterAnalysis::CalculateConsolidation(vector< double >& scores)
{
    // 根据边界上边的分布计算connectivity
    vector<double> connectivity(AdjPts.size());
    for (int it = 0; it < AdjPts.size(); it++)
    {
        int pt_it = AdjPts[it][0], pt_jt = AdjPts[it][1], pos1 = 2 * K, pos2 = 2 * K;
        for (size_t jt = 0; jt < PNEI[pt_it].size(); jt++)
        {
            if (PNEI[pt_it][jt] == pt_jt)
            {
                pos2 = jt + 1;
                break;
            }
        }
        
        for (size_t jt = 0; jt < PNEI[pt_jt].size(); jt++)
        {
            if (PNEI[pt_jt][jt] == pt_it)
            {
                pos1 = jt + 1;
                break;
            }
        }

        connectivity[it] = 2 * sqrt(pos1 * pos2) / 
            ((pos1 + pos2) * (dataSets.row(pt_it) - dataSets.row(pt_jt)).norm());
        /*connectivity[it] = 2 * sqrt(pos1 * pos2) /
            ((pos1 + pos2) * (dataSets.row(state[pt_it].pioneer) - dataSets.row(state[pt_jt].pioneer)).norm());*/
        /*connectivity[it] = (state[pt_jt].density + state[pt_it].density) * sqrt(pos1 * pos2) /
            ((pos1 + pos2) * (dataSets.row(pt_it) - dataSets.row(pt_jt)).norm());*/

        /*vector<int> closestPts;
        closestPts.reserve(Border_of_Clusters[cluster_2][cluster_1].size());
        double crossEdges = 0.0, sumDensity = 0.0, distMin = DBL_MAX;
        for (auto pt : Border_of_Clusters[cluster_2][cluster_1])
        {
            double mindist = DBL_MAX;
            int closestPt = -1;
            for (auto qt : Border_of_Clusters[cluster_1][cluster_2])
            {
                double dist = (dataSets.row(pt) - dataSets.row(qt)).norm();
                if (dist < mindist)
                {
                    mindist = dist;
                    closestPt = qt;
                }
            }
            if (mindist < distMin)
            {
                distMin = mindist;
            }
            crossEdges += mindist;
            sumDensity += state[pt].density;
            closestPts.push_back(closestPt);
        }

        int crossEdgesCount = Border_of_Clusters[cluster_2][cluster_1].size();
        for (auto pt : Border_of_Clusters[cluster_1][cluster_2])
        {
            double mindist = DBL_MAX;
            int closestPos = -1;
            for (size_t jt = 0; jt < Border_of_Clusters[cluster_2][cluster_1].size(); jt++)
            {
                int qt = Border_of_Clusters[cluster_2][cluster_1][jt];
                double dist = (dataSets.row(pt) - dataSets.row(qt)).norm();
                if (dist < mindist)
                {
                    mindist = dist;
                    closestPos = jt;
                }
            }
            sumDensity += state[pt].density;
            if (closestPts[closestPos] != pt)
            {
                crossEdges += mindist;
                crossEdgesCount++;
            }
        }*/

        /*connectivity[it] = crossEdgesCount / (distMin * distMin * crossEdges);
        printf("%d %d %d %d %5.2f %5.2lf %5.2lf %5.2lf %5.2lf\n", cluster_1, cluster_2, crossEdgesCount,
            Border_of_Clusters[cluster_2][cluster_1].size() + Border_of_Clusters[cluster_1][cluster_2].size(),
            crossEdges, 1 / crossEdges, distMin, 1 / (distMin * distMin), sumDensity);*/
    }

    double minCon = *min_element(connectivity.begin(), connectivity.end()),
        diffCon = *max_element(connectivity.begin(), connectivity.end()) - minCon;
    for (int it = 0; it < AdjClust.size(); it++)
    {
        connectivity[it] = (connectivity[it] - minCon) / diffCon;
    }

    //Border_of_Clusters.clear();
    //vector<vector<vector<int>>>().swap(Border_of_Clusters);

    // 计算Cluster Crossover Degree (CCD)
    /*vector<vector<double>> CCD(clusterID, vector<double>(clusterID, 0.0));
    vector<vector<int>> CNT(clusterID, vector<int>(clusterID, 1));
    for (size_t i = 0; i < dataNum; i++)
    {
        vector<int> cnt(clusterID, 0);
        for (size_t j = 0; j < K; j++)
        {
            int nei = PNEI[i][j];
            cnt[state[nei].clusterId]++;
        }
        int cur = state[i].clusterId;
        for (size_t j = 0; j < clusterID; j++)
        {
            double temp = 2 * sqrt(cnt[cur] * cnt[j]) / (cnt[cur] + cnt[j]);
            CCD[cur][j] += temp;
            CCD[j][cur] += temp;
            CNT[cur][j]++;
            CNT[j][cur]++;
        }
    }
    for (size_t i = 0; i < clusterID; i++)
    {
        for (size_t j = 0; j < clusterID; j++)
        {
            if (CNT[i][j] > 0)
            {
                CCD[i][j] /= CNT[i][j];
            }
        }
    }*/

    // 根据SLDP(Shared neighbors between two local density peaks)计算connectivity
    /*vector<vector<double>> connectivity(clusterID, vector<double>(clusterID, 1.0));
    vector< vector< int > > PCLUSTER;
    PCLUSTER.resize(clusterID);
    for (int it = 0; it < dataNum; it++)
    {
        int clust = state[it].clusterId;
        PCLUSTER[clust].push_back(it);
    }

    vector< vector< int > > NLDP;
    vector<int> mark(dataNum, 0);
    for (size_t i = 0; i < PCLUSTER.size(); i++)
    {
        vector< int > nldp;
        for (size_t j = 0; j < PCLUSTER[i].size(); j++)
        {
            int pt = PCLUSTER[i][j];
            if (mark[pt] == 0)
            {
                mark[pt] = 1;
                nldp.push_back(pt);
            }
            for (size_t nei = 0; nei <= K / 2; nei++)
            {
                int nn = PNEI[pt][nei];
                if (mark[nn] == 0)
                {
                    mark[nn] = 1;
                    nldp.push_back(nn);
                }
            }
        }
        for (size_t j = 0; j < nldp.size(); j++)
        {
            mark[nldp[j]] = 0;
        }
        NLDP.push_back(nldp);
    }

    for (size_t i = 0; i < clusterID; i++)
    {
        for (size_t j = 0; j < NLDP[i].size(); j++)
        {
            mark[NLDP[i][j]] = 1;
        }
        for (size_t j = i + 1; j < clusterID; j++)
        {
            int count = 0;
            double sum = 0.0;
            for (size_t jt = 0; jt < NLDP[j].size(); jt++)
            {
                if (mark[NLDP[j][jt]] > 0)
                {
                    //printf("%d ", NLDP[j][jt] + 1);
                    count++;
                    sum += state[NLDP[j][jt]].density;
                }
            }
            connectivity[i][j] = sum / count;
            connectivity[j][i] = sum / count;
        }
        for (size_t j = 0; j < NLDP[i].size(); j++)
        {
            mark[NLDP[i][j]] = 0;
        }
    }
    //printf("\n\n");*/


    // 计算鞍部点的密度与最高峰的密度的比值
    vector< double > macro_scores(AdjClust.size());
    for (int it = 0; it < AdjClust.size(); it++)
    {
        double peak_1 = state[ClusterCenters[AdjClust[it][0]]].density;
        double peak_2 = state[ClusterCenters[AdjClust[it][1]]].density;
        if (peak_1 < peak_2)
            macro_scores[it] = SaddlePoints[it][dataDim] / peak_2;
        else
            macro_scores[it] = SaddlePoints[it][dataDim] / peak_1;
    }

    // 综合合并系数
    for (int it = 0; it < AdjClust.size(); it++)
    {
        scores[it] = pow(macro_scores[it], preference) * pow(connectivity[it], 1 - preference);
        /*printf("%4d %4d  %.2lf   %.2lf  %.2lf\n", AdjClust[it][0], AdjClust[it][1], macro_scores[it], 
            connectivity[it], scores[it]);*/
    }
    /*for (int it = 0; it < AdjClust.size(); it++)
    {
        int cluster_1 = AdjClust[it][0];
        int cluster_2 = AdjClust[it][1];
        scores[it] = pow(macro_scores[it], preference) * pow(connectivity[cluster_1][cluster_2], 1 - preference);
    }*/
}

int ClusterAnalysis::ConstructClusterTree() {
    DisjSet disjSet(clusterID);
    vector< double > merge_scores(AdjClust.size());
    CalculateConsolidation(merge_scores);
    vector<int> order(AdjClust.size());
    for (int it = 0; it < AdjClust.size(); it++)
        order[it] = it;

    sort(order.begin(), order.end(),
        [merge_scores](int a, int b) { return merge_scores[a] > merge_scores[b]; });
    vector< vector< int > > MST;

    vector< double > MST_scores;
    vector< vector< int > > IND(clusterID, vector<int>(clusterID, 0));
    for (int it = 0; it < AdjClust.size(); it++)
    {
        int cluster_1 = AdjClust[order[it]][0];
        int cluster_2 = AdjClust[order[it]][1];
        if (disjSet.Is_same(cluster_1, cluster_2) == false)
        {
            disjSet.Union(cluster_1, cluster_2);
            MST.push_back({ cluster_1, cluster_2 });
            MST_scores.push_back(merge_scores[order[it]]);
            IND[cluster_1][cluster_2] = 1;
            IND[cluster_2][cluster_1] = 1;
            if (disjSet.Size() == 1)
                break;
        }
    }

    {
        vector< int > cluster(clusterID, -1);
        int ncl = 0;
        for (size_t it = 0; it < clusterID; it++)
        {
            if (cluster[it] == -1)
            {
                cluster[it] = ncl;
                vector < int > queue(clusterID, 0);
                int front = 0, rear = 0;
                queue[rear] = it;
                rear += 1;
                while (front != rear)
                {
                    int p = queue[front++];
                    for (size_t j = 0; j < clusterID; j++)
                    {
                        if (IND[p][j] + IND[j][p] > 0 && cluster[j] == -1)
                        {
                            cluster[j] = ncl;
                            queue[rear] = j;
                            rear++;
                        }
                    }
                }
                ncl++;
            }
        }
        printf("Minimum number of clusters: %d\n", ncl);
        if (ncl == clu_num)
        {
            printf("Obtain the desired clusters: %d\n", 0);
            for (size_t it = 0; it < dataNum; it++)
            {
                state[it].clusterID = cluster[state[it].clusterId];
            }
            if (o_tree == 0)
            {
                return 0;
            }
        }
        if (ncl > clu_num)
        {
            printf("Minimum number of clusters is greater than desired number of clusters\n");
            return 0;
        }
        MERGE.push_back(move(cluster)); 
    }

    vector< int > sz(clusterID, 0);
    for (int it = 0; it < dataNum; it++)
    {
        int clust = state[it].clusterId;
        sz[clust]++;
    }
    int i = MST.size();
    while (i > 0)
    {
        int s1 = 0, s2 = 0;
        int p, q;
        while (s1 < MinSize || s2 < MinSize)
        {
            s1 = 0; s2 = 0;
            i--;
            if (i < 0)
            {
                break;
            }
            p = MST[i][0];
            q = MST[i][1];
            vector < int > queue(clusterID, 0);
            int front = 0, rear = 0;
            queue[rear] = p;
            rear = rear + 1;
            vector<int> visited(clusterID, 0);
            visited[p] = 1;
            while (front != rear)
            {
                int temp = queue[front];
                s1 += sz[temp];
                front++;
                for (size_t j = 0; j < clusterID; j++)
                {
                    if (IND[temp][j] + IND[j][temp] > 0 && j != q && visited[j] == 0)
                    {
                        visited[j] = 1;
                        queue[rear] = j;
                        rear += 1;
                    }
                }
            }
            front = 0;
            rear = 0;
            queue[rear] = q;
            rear++;
            visited[q] = 1;
            while (front != rear)
            {
                int temp = queue[front];
                s2 += sz[temp];
                front++;
                for (size_t j = 0; j < clusterID; j++)
                {
                    if (IND[temp][j] + IND[j][temp] > 0 && j != p && visited[j] == 0)
                    {
                        visited[j] = 1;
                        queue[rear] = j;
                        rear += 1;
                    }
                }
            }
        }
        if (i < 0)
        {
            break;
        }
        IND[p][q] = 0;
        IND[q][p] = 0;
        ClusterTree.push_back({ (double)p, (double)q, MST_scores[i] });
        vector< int > cluster(clusterID, -1);
        int ncl = 0;
        for (size_t it = 0; it < clusterID; it++)
        {
            if (cluster[it] == -1)
            {
                cluster[it] = ncl;
                vector < int > queue(clusterID, 0);
                int front = 0, rear = 0;
                queue[rear] = it;
                rear += 1;
                int clustersize = sz[it];
                while (front != rear)
                {
                    int p = queue[front++];
                    for (size_t j = 0; j < clusterID; j++)
                    {
                        if (IND[p][j] + IND[j][p] > 0 && cluster[j] == -1)
                        {
                            clustersize += sz[j];
                            cluster[j] = ncl;
                            queue[rear] = j;
                            rear++;
                        }
                    }
                }
                if (clustersize > 0)
                {
                    ncl++;
                }
            }
        }
        if (ncl == clu_num)
        {
            printf("Obtain the desired clusters: %d\n", MERGE.size());
            for (size_t it = 0; it < dataNum; it++)
            {
                state[it].clusterID = cluster[state[it].clusterId];
            }
            if (o_tree == 0)
            {
                return 0;
            }
        }
        MERGE.push_back(move(cluster));
    }

    return 1;
    //sort(order.begin(), order.end(),
    //    [merge_scores](int a, int b) { return merge_scores[a] > merge_scores[b]; });
    //for (int it = 0; it < AdjClust.size(); it++)
    //{
    //    int cluster_1 = AdjClust[order[it]][0];
    //    int cluster_2 = AdjClust[order[it]][1];
    //    //printf("%d %d %f\n", cluster_1, cluster_2, merge_scores[order[it]]);
    //    if (disjSet.Is_same(cluster_1, cluster_2) == false)
    //    {
    //        disjSet.Union(cluster_1, cluster_2);
    //        //printf("%d %d\n", ClusterCenters[cluster_1] + 1, ClusterCenters[cluster_2] + 1);
    //        ClusterTree.push_back({ (double)cluster_1, (double)cluster_2, merge_scores[order[it]] });
    //        if (disjSet.Size() > 0)
    //        {
    //            vector< int > cluster(clusterID, -1);
    //            int clust = 1;
    //            for (int i = 0; i < clusterID; i++)
    //            {
    //                int par = disjSet.Find(i);
    //                if (cluster[par] == -1)
    //                {
    //                    cluster[par] = clust++;
    //                }
    //                cluster[i] = cluster[par];
    //            }
    //            MERGE.push_back(move(cluster));
    //        }
    //    }
    //}
}


void ClusterAnalysis::SetArrivalPoints_SNN() {
    vector<double> pt(dataDim);
    for (int it = 0; it < dataNum; it++)
    {
        for (int jt = 0; jt < dataDim; jt++)
        {
            pt[jt] = dataSets(it, jt);
        }
        nanoflann::KNNResultSet<double> resultSet(K);
        resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
        node_data_kd->index_->findNeighbors(resultSet, &pt[0]);

        vector<int> nei;
        nei.reserve(K);
        for (int i = 0; i < K; i++)
        {
            nei.push_back(ret_indexes[i]);
        }
        PNEI.push_back(move(nei));
        state[it].kdist = sqrt(out_dists_sqr[K - 1]);
    }
    FindMNN();

    vector<int> mark(dataNum, 0);
    for (size_t i = 0; i < dataNum; i++)
    {
        for (size_t j = 0; j < PNEI[i].size(); j++)
        {
            mark[PNEI[i][j]] = 1;
        }
        state[i].density = SNNLocalDensity(MNN[i], mark, i);
        for (size_t j = 0; j < PNEI[i].size(); j++)
        {
            mark[PNEI[i][j]] = 0;
        }
    }
}

double ClusterAnalysis::EuclideanDist(vector<double>& coord, int pt)
{
    double sum = 0.0;
    for (size_t i = 0; i < dataDim; i++)
    {
        double temp = coord[i] - dataSets(pt, i);
        temp *= temp;
        sum += temp;
    }
    return sqrt(sum);
}

double ClusterAnalysis::SNNLocalDensity(vector<double>& coord, vector<int>& neis, vector<int>& mark) {
    double density = 0.0;
    for (size_t jt = 0; jt < neis.size(); jt++)
    {
        int count = 0;
        int i = neis[jt];
        double x = 0.0;
        for (size_t it = 0; it < K; it++)
        {
            int pt = PNEI[i][it];
            if (mark[pt] > 0)
            {
                count++;
                x += (dataSets.row(i) - dataSets.row(pt)).norm() + EuclideanDist(coord, pt);
            }
        }
        density += count * count / x;
    }
    return density;
}

double ClusterAnalysis::SNNLocalDensity(vector<int>& neis, vector<int>& mark, int j) {
    double density = 0.0;
    for (size_t jt = 0; jt < neis.size(); jt++)
    {
        double x = 0.0;
        int count = 0;
        int i = neis[jt];
        for (size_t it = 0; it < PNEI[i].size(); it++)
        {
            int pt = PNEI[i][it];
            if (mark[pt] > 0)
            {
                count++;
                x += (dataSets.row(i) - dataSets.row(pt)).norm() + (dataSets.row(j) - dataSets.row(pt)).norm();
            }
        }
        density += count * count / x;
    }
    return density;
}

void ClusterAnalysis::FindSaddlePoints_SNN() {
    vector< vector< int > > PCLUSTER;
    PCLUSTER.resize(clusterID);
    for (int it = 0; it < dataNum; it++)
    {
        int clust = state[it].clusterId;
        PCLUSTER[clust].push_back(it);
    }

    vector< vector< int > > MinDist(dataNum, vector< int >(clusterID, -1));
    // 找边界点
    for (int it = 0; it < dataNum; it++)
    {
        int clust = state[it].clusterId;
        //vector<int> adjpts;
        for (int jt = 0; jt < K; jt++)
        {
            int adjpoint = PNEI[it][jt];
            int adj_cluster = state[adjpoint].clusterId;
            // 检查邻居数据点是否属于不同的簇并且是到这个簇中的点中距离最小的
            if (adj_cluster != clust && MinDist[it][adj_cluster] == -1)
            {
                // 将边界上的点(属于簇adj_cluster的it的近邻中离it最近的点)保存到MinDist中
                MinDist[it][adj_cluster] = adjpoint;
            }
        }
    }

    vector< vector< pair<int, double> > > SADDLE;
    vector< vector< vector< int > > > Borders;
    for (int it = 0; it < PCLUSTER.size(); it++)
    {
        // 找到相邻簇的边界点中密度最高的点
        vector< pair<int, double> > border(clusterID, { -1, 0.0 });
        for (int jt = 0; jt < PCLUSTER[it].size(); jt++)
        {
            int pt = PCLUSTER[it][jt];
            for (int zt = 0; zt < clusterID; zt++)
            {
                int adjpt = MinDist[pt][zt];
                if (adjpt != -1 && state[adjpt].density > border[zt].second)
                {
                    border[zt].first = adjpt;
                    border[zt].second = state[adjpt].density;
                }
            }
        }
        SADDLE.push_back(move(border));
    }

    for (int it = 0; it < SADDLE.size(); it++)
    {
        for (int jt = 0; jt < SADDLE[it].size(); jt++)
        {
            int pt = SADDLE[it][jt].first;
            if (pt != -1)
            {
                // it簇与jt簇相邻，并且pt是jt簇中与it簇的边界点
                // pt是it簇中某一点的k近邻
                // 下面找it簇中距离pt最近的点，将其与pt的均值点作为it簇与jt簇的鞍点
                int adjpt = -1;
                for (size_t i = 0; i < PNEI[pt].size(); i++)
                {
                    int nei = PNEI[pt][i];
                    if (state[nei].clusterId == it)
                    {
                        adjpt = nei;
                        break;
                    }
                }
                if (adjpt == -1)
                {
                    double mindist = DBL_MAX;
                    for (int i = 0; i < PCLUSTER[it].size(); i++)
                    {
                        double dist = (dataSets.row(pt) - dataSets.row(PCLUSTER[it][i])).norm();
                        if (dist < mindist)
                        {
                            mindist = dist;
                            adjpt = PCLUSTER[it][i];
                        }
                    }
                }
                // 计算鞍点的坐标和密度值
                vector< double > saddle;
                saddle.reserve(dataDim + 1);
                for (int i = 0; i < dataDim; i++)
                {
                    saddle.push_back((dataSets(pt, i) + dataSets(adjpt, i)) / 2);
                }
                saddle.push_back(CalculateDensity_SNN(saddle));
                SaddlePoints.push_back(move(saddle));
                AdjClust.push_back({ it, jt });
                AdjPts.push_back({ adjpt, pt });
            }
        }
    }
}

double ClusterAnalysis::CalculateDensity_SNN(vector< double >& pt) {
    nanoflann::KNNResultSet<double> resultSet(K);
    resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
    node_data_kd->index_->findNeighbors(resultSet, &pt[0]);

    vector<int> mark(dataNum, 0);
    vector<int> neis;
    for (size_t i = 0; i < K; i++)
    {
        int nei = ret_indexes[i];
        mark[nei] = 1;
        if (state[nei].kdist >= sqrt(out_dists_sqr[i]))
        {
            neis.push_back(nei);
        }
    }

    return SNNLocalDensity(pt, neis, mark);
}


bool ClusterAnalysis::Init(char* fileName, int nc) {
    this->clu_num = nc;

    int dim;
    int data_size;
    cout << "reading data..." << endl;
    double* raw_data = read_data((char*)fileName, (char*)" ", &dim, &data_size);
    dataNum = data_size;
    dataDim = dim - 1;
    dataSets.resize(dataNum, dataDim);
    for (size_t i = 0; i < dataNum; i++) {
        for (size_t j = 0; j < dataDim; j++)
            dataSets(i, j) = raw_data[dim * i + j];
    }
    free(raw_data);

    state = (State*)malloc(data_size * sizeof(State));
    if (state != NULL)
    {
        for (size_t i = 0; i < dataNum; i++)
        {
            state[i].pioneer = -1;
            state[i].clusterId = -1;
        }
    }
    //printMatrix(dataSets);

    CYW_TIMER build_timer;
    build_timer.start_my_timer();
    cout << "building the trees...\n";
    node_data_kd = new nanoflann::KDTreeEigenMatrixAdaptor<MatrixXd>(dataDim, dataSets, 10);
    build_timer.stop_my_timer();

    //printMatrix(dataSets);
    printf("n = %d  dim = %d\n", dataNum, dataDim);
    printf("kd-tree build time = %.4f\n", build_timer.get_my_timer());
    return true;
}

void ClusterAnalysis::Running_LDP_MST() {
    Parameters_LDP_MST();
    FindPioneers_LDP_MST();           // Find point's pioneer.
    GenerateCusters_LDP_MST();        // Form initial clusters.
    ExtractClusters();                // Extract clustering results.
}

void ClusterAnalysis::Parameters_LDP_MST()
{
    int temp = 100;
    vector< double >  out_dists_sqr1(temp);
    vector< size_t > ret_indexes1(temp);
    vector< vector< int > > KNN1;
    vector<double> pt(dataDim);
    for (int it = 0; it < dataNum; it++)
    {
        for (int jt = 0; jt < dataDim; jt++)
        {
            pt[jt] = dataSets(it, jt);
        }
        nanoflann::KNNResultSet<double> resultSet(temp);
        resultSet.init(&ret_indexes1[0], &out_dists_sqr1[0]);
        node_data_kd->index_->findNeighbors(resultSet, &pt[0]);

        vector<int> nei;
        nei.reserve(temp);
        for (int i = 0; i < temp; i++)
        {
            if (ret_indexes1[i] != it)
            {
                nei.push_back(ret_indexes1[i]);
            }
        }
        KNN1.push_back(move(nei));
    }

    int r = 0;
    int flag = 0;
    vector<int> nb(dataNum, 0);
    int count = 0, count1 = dataNum, count2 = dataNum;
    while (flag == 0)
    {
        for (size_t i = 0; i < dataNum; i++)
        {
            int k = KNN1[i][r];
            nb[k] += 1;
            if (nb[k] == 1)
            {
                count2 -= 1;
            }
        }
        r += 1;
        if (count1 == count2)
        {
            count += 1;
        }
        else
        {
            count = 1;
        }
        if (count2 == 0 || (r > 1 && count >= 2))
        {
            flag = 1;
        }
        count1 = count2;
    }
    lambda = r;
    K = *max_element(nb.begin(), nb.end());
    printf("K: %d, lambda: %d\n", K, lambda);

    for (size_t i = 0; i < dataNum; i++)
    {
        vector<int> nei;
        nei.reserve(K);
        double sumdist = 0.0;
        for (size_t j = 0; j < K; j++)
        {
            nei.push_back(KNN1[i][j]);
            sumdist += (dataSets.row(i) - dataSets.row(KNN1[i][j])).norm();
        }
        state[i].density = nb[i] / sumdist;
        PNEI.push_back(nei);
    }
}

bool ClusterAnalysis::FindPioneers_LDP_MST() {
    for (int it = 0; it < dataNum; it++)
    {
        double max = state[it].density;
        for (int jt = 0; jt < lambda; jt++)
        {
            int neiId = PNEI[it][jt];
            if (max < state[neiId].density)
            {
                max = state[neiId].density;
                state[it].pioneer = neiId;
            }
        }
    }
    return true;
}

void ClusterAnalysis::GenerateCusters_LDP_MST() {
    // Point are traversed in descending order according to their densities.
    vector<int> order(dataNum);
    for (int it = 0; it < dataNum; it++)
        order[it] = it;
    State* temp = state;
    sort(order.begin(), order.end(),
        [temp](int a, int b) {return temp[a].density > temp[b].density; });
    clusterID = 0;
    for (int it = 0; it < dataNum; it++)
    {
        int processedPt = order[it];
        int pioneer = state[processedPt].pioneer;
        if(pioneer == -1)
        {
            ClusterCenters.push_back(processedPt);
            state[processedPt].clusterId = clusterID++;
        }
        else
        {
            state[processedPt].clusterId = state[pioneer].clusterId;
        }
    }

    printf("There are %d initial clusters.\n", clusterID);
}

void ClusterAnalysis::CalculateSND(vector< vector< int > >& MST)
{
    vector< vector< int > > PCLUSTER;
    PCLUSTER.resize(clusterID);
    for (int it = 0; it < dataNum; it++)
    {
        int clust = state[it].clusterId;
        PCLUSTER[clust].push_back(it);
    }

    vector< vector< int > > NLDP;
    vector<int> mark(dataNum, 0);
    for (size_t i = 0; i < PCLUSTER.size(); i++)
    {
        vector< int > nldp;
        for (size_t j = 0; j < PCLUSTER[i].size(); j++)
        {
            int pt = PCLUSTER[i][j];
            if (mark[pt] == 0)
            {
                mark[pt] = 1;
                nldp.push_back(pt);
            }
            for (size_t nei = 0; nei < lambda; nei++)
            {
                int nn = PNEI[pt][nei];
                if (mark[nn] == 0)
                {
                    mark[nn] = 1;
                    nldp.push_back(nn);
                }
            }
        }
        for (size_t j = 0; j < nldp.size(); j++)
        {
            mark[nldp[j]] = 0;
        }
        NLDP.push_back(nldp);
    }

    double maxd = 0.0;
    for (size_t i = 0; i < clusterID; i++)
    {
        for (size_t j = i + 1; j < clusterID; j++)
        {
            double dist = (dataSets.row(ClusterCenters[i]) - dataSets.row(ClusterCenters[j])).norm();
            if (dist > maxd)
            {
                maxd = dist;
            }
        }
    }
    vector< double > SND;
    for (size_t i = 0; i < clusterID; i++)
    {
        for (size_t j = 0; j < NLDP[i].size(); j++)
        {
            mark[NLDP[i][j]] = 1;
        }
        for (size_t j = i + 1; j < clusterID; j++)
        {
            AdjClust.push_back({ (int)i, (int)j });
            int count = 0;
            double sum = 0.0;
            for (size_t jt = 0; jt < NLDP[j].size(); jt++)
            {
                if (mark[NLDP[j][jt]] > 0)
                {
                    //printf("%d ", NLDP[j][jt]);
                    count += 1;
                    sum += state[NLDP[j][jt]].density;
                }
            }
            if (count > 0)
            {
                SND.push_back((dataSets.row(ClusterCenters[i]) - dataSets.row(ClusterCenters[j])).norm()
                    / (count * sum));
            }
            else
            {
                SND.push_back(maxd * (1 + (dataSets.row(ClusterCenters[i]) - dataSets.row(ClusterCenters[j])).norm()));
            }
        }
        for (size_t j = 0; j < NLDP[i].size(); j++)
        {
            mark[NLDP[i][j]] = 0;
        }
    }
    //printf("\n\n");
    vector<int> order;
    order.resize(AdjClust.size());
    for (int it = 0; it < AdjClust.size(); it++)
        order[it] = it;
    vector<double>& temp = SND;
    sort(order.begin(), order.end(),
        [temp](int a, int b) { return temp[a] < temp[b]; });

    DisjSet disjSet(clusterID);
    for (int it = 0; it < AdjClust.size(); it++)
    {
        int cluster_1 = AdjClust[order[it]][0];
        int cluster_2 = AdjClust[order[it]][1];
        if (disjSet.Is_same(cluster_1, cluster_2) == false)
        {
            disjSet.Union(cluster_1, cluster_2);
            MST.push_back({ cluster_1, cluster_2 });
            if (disjSet.Size() == 1)
                break;
        }
    }

}

int ClusterAnalysis::iscontain(vector<int>& q, int front, int rear, int x)
{
    for (size_t i = front; i < rear; i++)
    {
        if (q[i] == x)
        {
            return 1;
        }
    }
    return 0;
}

void ClusterAnalysis::ExtractClusters() {
    /*ofstream of1("data\\result1.txt");
    for (int i = 0; i < dataNum; i++)
    {
        of1 << state[i].clusterId << " " << state[i].density << endl;
    }
    of1.close();*/
    vector< vector< int > > MST;
    CalculateSND(MST);
    int minsize = ceil(0.018 * dataNum);
    vector< vector< int > > IND(clusterID, vector<int>(clusterID, 0));

    for (size_t i = 0; i < MST.size(); i++)
    {
        int cluster_1 = MST[i][0];
        int cluster_2 = MST[i][1]; 
        IND[cluster_1][cluster_2] = 1;
        IND[cluster_2][cluster_1] = 1;
    }
    vector< int > sz(clusterID, 0);
    for (int it = 0; it < dataNum; it++)
    {
        int clust = state[it].clusterId;
        sz[clust]++;
    }

    int k = 1, i = MST.size();
    while (k < clu_num)
    {
        int s1 = 0, s2 = 0;
        int p = MST[i - 1][0];
        int q = MST[i - 1][1];
        while (s1 < minsize || s2 < minsize)
        {
            s1 = 0; s2 = 0;
            vector<int> visited(clusterID,0);
            i -= 1;
            p = MST[i][0];
            q = MST[i][1]; 
            vector < int > queue(clusterID, 0);
            int front = 0, rear = 0; 
            queue[rear] = p;
            rear = rear + 1;
            while (front != rear)
            {
                int temp = queue[front];
                s1 = s1 + sz[temp];
                visited[temp] = 1;
                front += 1;
                for (size_t j = 0; j < clusterID; j++)
                {
                    if (IND[temp][j] + IND[j][temp] > 0 && j!= q && visited[j] == 0 && 
                        iscontain(queue, front, rear,j) == 0)
                    {
                        queue[rear] = j;
                        rear += 1;
                    }
                }
            }
            front = 0;
            rear = 0;
            queue[rear] = q;
            rear = rear + 1;
            while (front != rear)
            {
                int temp = queue[front];
                s2 = s2 + sz[temp];
                visited[temp] = 1;
                front += 1;
                for (size_t j = 0; j < clusterID; j++)
                {
                    if (IND[temp][j] + IND[j][temp] > 0 && j != p && visited[j] == 0 &&
                        iscontain(queue, front, rear, j) == 0)
                    {
                        queue[rear] = j;
                        rear += 1;
                    }
                }
            }
        }
        IND[p][q] = 0;
        IND[q][p] = 0;
        k++;
    }
    vector< int > cluster(clusterID, -1);
    int ncl = -1;
    vector< int > sumedge(clu_num, 1);
    for (size_t i = 0; i < clusterID; i++)
    {
        if (cluster[i] == -1)
        {
            ncl = ncl + 1;
            vector < int > queue(clusterID, 0);
            vector < int > visited(clusterID, 0);
            int front = 0,rear = 0; 
            queue[rear] = i;
            visited[i] = 1;
            rear += 1;
            sumedge[ncl] = 0;
            while (front != rear)
            {
                int p = queue[front];
                front += 1;
                cluster[p] = ncl;
                for (size_t j = 0; j < clusterID; j++)
                {
                    if (IND[p][j] + IND[j][p] > 0 && cluster[j] == -1 && visited[j] == 0)
                    {
                        visited[j] = 1;
                        queue[rear] = j;
                        rear += 1;
                    }
                }
            }
        }
    }
    for (size_t i = 0; i < dataNum; i++)
    {
        int clu = state[i].clusterId;
        state[i].clusterId = cluster[clu];
    }
}


bool ClusterAnalysis::WriteToFile(){
    printf("Saving the results...\n");
    ofstream of1("data\\result.txt");
    for (int i = 0; i < dataNum; i++)
    {
        of1 << state[i].clusterId << " " << state[i].density << " " << state[i].pioneer << endl;
    }
    of1.close();

    if (alg != 2)
    {
        ofstream of1("data\\kresult.txt");
        for (int i = 0; i < dataNum; i++)
        {
            of1 << state[i].clusterID << endl;
        }
        of1.close();

        ofstream of2("data\\saddle.txt");
        for (int it = 0; it < SaddlePoints.size(); it++)
        {
            for (int jt = 0; jt < SaddlePoints[it].size(); jt++)
            {
                of2 << SaddlePoints[it][jt] << " ";
            }
            of2 << endl;
        }
        of2.close();

        if (o_tree != 0)
        {
            ofstream of3("data\\clustertree.txt");
            for (int it = 0; it < ClusterTree.size(); it++)
            {
                for (int jt = 0; jt < ClusterTree[it].size(); jt++)
                {
                    of3 << ClusterTree[it][jt] << " ";
                }
                of3 << endl;
            }
            of3.close();

            ofstream of4("data\\merge.txt");
            for (int it = 0; it < MERGE.size(); it++)
            {
                for (int jt = 0; jt < MERGE[it].size(); jt++)
                {
                    of4 << MERGE[it][jt] << " ";
                }
                of4 << endl;
            }
            of4.close();
        }
    }
    printf("Done.\n");
    return true;
}
