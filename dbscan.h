#pragma once
#include "headers.h"

class DBSCAN
{
private:
    MatrixXd dataSets;  // 数据矩阵（行=样本，列=维度）
    nanoflann::KDTreeEigenMatrixAdaptor<MatrixXd>* node_data_kd;  // KD树
    vector<size_t> ret_indexes;  // KDTree搜索返回索引
    vector<int> labels;           // 样本的簇标签，-1表示噪声
    vector<int> visited;           // 样本的簇标签，-1表示噪声
    double radius;               // 邻域半径 eps
    int dataNum;                 // 样本数
    int dataDim;                 // 特征维数
    int minPts;                  // 最小邻域点数

public:
    DBSCAN() : node_data_kd(nullptr), radius(0), dataNum(0), dataDim(0), minPts(0) {}
    ~DBSCAN() {
        if (node_data_kd) delete node_data_kd;
    }

    // 初始化数据和KD树
    bool Init(char* fileName, int K, double eps)
    {
        this->radius = eps;
        this->minPts = K;
        ret_indexes.resize(K);

        cout << "radius: " << radius << " minPts: " << minPts << endl;
        int dim;
        int data_size;
        cout << "reading data..." << endl;
        double* raw_data = read_data(fileName, (char*)" ", &dim, &data_size);

        dataNum = data_size;
        dataDim = dim - 1;  // 假设最后一列是标签或者忽略
        dataSets.resize(dataNum, dataDim);
        for (size_t i = 0; i < dataNum; i++) {
            for (size_t j = 0; j < dataDim; j++)
                dataSets(i, j) = raw_data[dim * i + j];
        }
        free(raw_data);

        cout << "building the KD-tree...\n";
        CYW_TIMER build_timer;
        build_timer.start_my_timer();
        node_data_kd = new nanoflann::KDTreeEigenMatrixAdaptor<MatrixXd>(dataDim, dataSets, 10);
        build_timer.stop_my_timer();

        printf("n = %d  dim = %d\n", dataNum, dataDim);
        printf("kd-tree build time = %.4f\n", build_timer.get_my_timer());

        labels.resize(dataNum, -1);  // 初始化所有样本为噪声
        visited.resize(dataNum, 0);  // 初始化所有样本为噪声
        return true;
    }

    // DBSCAN 核心聚类逻辑
    void Running()
    {
        cout << "Running DBSCAN..." << endl;
        int cluster_id = 0;

        for (int i = 0; i < dataNum; i++) {
            if (labels[i] != -1) continue;  // 已经访问过
            vector<size_t> neighbor_indices;
            radiusSearch(i, neighbor_indices);
            visited[i] = 1;
            if (neighbor_indices.size() < minPts) {
                labels[i] = -1;  // 标记为噪声
            }
            else {
                expandCluster(i, neighbor_indices, cluster_id);
                cluster_id++;
            }
        }

        cout << "DBSCAN finished. Found " << cluster_id << " clusters." << endl;
    }

    // 获取聚类结果
    const vector<int>& getLabels() const { return labels; }

private:
    // KDTree邻域搜索
    void radiusSearch(int idx, vector<size_t>& neighbor_indices)
    {
        nanoflann::SearchParameters params;
        Eigen::VectorXd query = dataSets.row(idx);
        std::vector<nanoflann::ResultItem<size_t, double>> radiusIdxs;
        nanoflann::RadiusResultSet<double, size_t> radiusResults(radius * radius, radiusIdxs);
        radiusResults.init();
        nanoflann::SearchParameters searchParams;
        searchParams.sorted = false;
        node_data_kd->index_->findNeighbors(radiusResults, query.data(), searchParams);
        for (size_t i = 0; i < radiusIdxs.size(); i++)
        {
            neighbor_indices.push_back(radiusIdxs[i].first);
        }
    }

    // 扩展簇
    void expandCluster(int idx, vector<size_t>& neighbor_indices, int cluster_id)
    {
        labels[idx] = cluster_id;

        size_t k = 0;
        while (k < neighbor_indices.size()) {
            int n_idx = neighbor_indices[k];
            if (labels[n_idx] == -1) labels[n_idx] = cluster_id; // 将噪声点加入簇
            if (visited[n_idx] == 1) {
                k++;
                continue;  // 已经属于其他簇的点跳过
            }

            labels[n_idx] = cluster_id;

            // 扩展邻域
            vector<size_t> n_neighbors;
            radiusSearch(n_idx, n_neighbors);
            visited[n_idx] = 1;
            if (n_neighbors.size() >= minPts) {
                neighbor_indices.insert(neighbor_indices.end(), n_neighbors.begin(), n_neighbors.end());
            }
            k++;
        }
    }
};
