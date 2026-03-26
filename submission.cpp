#include <bits/stdc++.h>
#define BINS 256
#define ll long long int
#define MIN_COORD -10000
#define MAX_COORD 10000
#define COORD_RANGE 20000

using namespace std;

void parse_input(string filename, int*& h_px, int*& h_py, int*& h_pz, int*& h_intensity, int& N, int& K, int& T) {
    ifstream inFile(filename);

    if (!inFile.is_open()) {
        cout << "could not open input file\n";
        return;
    }

    inFile >> N >> K >> T;
    h_px        = (int*)malloc(N * sizeof(int));
    h_py        = (int*)malloc(N * sizeof(int));
    h_pz        = (int*)malloc(N * sizeof(int));
    h_intensity = (int*)malloc(N * sizeof(int));

    for (int i = 0; i < N; i++) {
        inFile >> h_px[i] >> h_py[i] >> h_pz[i] >> h_intensity[i];
    }

    inFile.close();
}

void writeoutput(int* h_px, int* h_py, int* h_pz, int* h_intensity, int N, string filename) {
    ofstream outFile(filename);

    if (!outFile.is_open()) {
        cout << "error in creating output file\n";
        return;
    }

    for (int i = 0; i < N; i++) {
        outFile << h_px[i] << " " << h_py[i] << " " << h_pz[i] << " " << h_intensity[i] << "\n";
    }

    outFile.close();
}

// For each query point, find its K nearest neighbors using a max-heap.
void KNN(int* px, int* py, int* pz, int* knn_index, int N, int K) {
    priority_queue<pair<ll, int>> heap;

    for (int q = 0; q < N; q++) {
        ll qx = px[q], qy = py[q], qz = pz[q];

        while (!heap.empty()) heap.pop();

        for (int j = 0; j < N; j++) {
            if (j == q) continue;
            ll dx   = (ll)px[j] - qx;
            ll dy   = (ll)py[j] - qy;
            ll dz   = (ll)pz[j] - qz;
            ll dist = dx * dx + dy * dy + dz * dz;

            if ((int)heap.size() < K) {
                heap.push({dist, j});
            } else if (dist < heap.top().first) {
                heap.pop();
                heap.push({dist, j});
            }
        }

        int k = (int)heap.size() - 1;
        while (!heap.empty()) {
            knn_index[q * K + k] = heap.top().second;
            heap.pop();
            k--;
        }
    }
}

static inline int clamp_cell(int value, int grid_dim) {
    return max(0, min(grid_dim - 1, value));
}

static inline void consider_neighbor(int q, int candidate, int* px, int* py, int* pz, vector<ll>& localDist, vector<int>& localIdx, int K) {
    if (candidate == q) return;

    ll dx   = (ll)px[candidate] - px[q];
    ll dy   = (ll)py[candidate] - py[q];
    ll dz   = (ll)pz[candidate] - pz[q];
    ll dist = dx * dx + dy * dy + dz * dz;

    int worstPos = 0;
    for (int j = 1; j < K; j++) {
        if (localDist[j] > localDist[worstPos]) {
            worstPos = j;
        }
    }

    if (dist < localDist[worstPos]) {
        localDist[worstPos] = dist;
        localIdx[worstPos]  = candidate;
    }
}

void APPROX_KNN(int* px, int* py, int* pz, int* knn_index, int N, int K) {
    int grid_dim = max(1, (int)cbrt(27.0 * N / (2.0 * K)) + 1);
    double cell_size = (double)COORD_RANGE / grid_dim;
    int total_cells = grid_dim * grid_dim * grid_dim;

    vector<int> cellid(N);
    vector<int> cell_count(total_cells, 0);
    vector<int> cell_start(total_cells, 0);
    vector<int> cell_offset(total_cells, 0);
    vector<int> sorted_points(N);

    for (int i = 0; i < N; i++) {
        int cx = clamp_cell((int)((px[i] - MIN_COORD) / cell_size), grid_dim);
        int cy = clamp_cell((int)((py[i] - MIN_COORD) / cell_size), grid_dim);
        int cz = clamp_cell((int)((pz[i] - MIN_COORD) / cell_size), grid_dim);
        cellid[i] = cx + cy * grid_dim + cz * grid_dim * grid_dim;
        cell_count[cellid[i]]++;
    }

    for (int cell = 1; cell < total_cells; cell++) {
        cell_start[cell] = cell_start[cell - 1] + cell_count[cell - 1];
    }

    for (int i = 0; i < N; i++) {
        int cell = cellid[i];
        sorted_points[cell_start[cell] + cell_offset[cell]++] = i;
    }

    for (int q = 0; q < N; q++) {
        vector<ll> localDist(K, LLONG_MAX);
        vector<int> localIdx(K, -1);

        int cell = cellid[q];
        int cx = cell % grid_dim;
        cell /= grid_dim;
        int cy = cell % grid_dim;
        cell /= grid_dim;
        int cz = cell % grid_dim;

        auto scan_shell = [&](int radius, bool full_cube) {
            for (int dx = -radius; dx <= radius; dx++) {
                for (int dy = -radius; dy <= radius; dy++) {
                    for (int dz = -radius; dz <= radius; dz++) {
                        if (!full_cube && max({abs(dx), abs(dy), abs(dz)}) != radius) continue;

                        int nx = cx + dx;
                        int ny = cy + dy;
                        int nz = cz + dz;

                        if (nx < 0 || nx >= grid_dim) continue;
                        if (ny < 0 || ny >= grid_dim) continue;
                        if (nz < 0 || nz >= grid_dim) continue;

                        int neighbor_cell = nx + ny * grid_dim + nz * grid_dim * grid_dim;
                        int start = cell_start[neighbor_cell];
                        int size = cell_count[neighbor_cell];

                        for (int j = 0; j < size; j++) {
                            consider_neighbor(q, sorted_points[start + j], px, py, pz, localDist, localIdx, K);
                        }
                    }
                }
            }
        };

        scan_shell(1, true);
        int found = 0;
        for (int idx : localIdx) found += (idx != -1);

        if (found < K) {
            scan_shell(2, false);
            found = 0;
            for (int idx : localIdx) found += (idx != -1);
        }

        // Fall back to the exact scan if the local search window is too sparse.
        if (found < K) {
            fill(localDist.begin(), localDist.end(), LLONG_MAX);
            fill(localIdx.begin(), localIdx.end(), -1);
            for (int j = 0; j < N; j++) {
                consider_neighbor(q, j, px, py, pz, localDist, localIdx, K);
            }
        }

        for (int j = 0; j < K; j++) {
            knn_index[q * K + j] = localIdx[j];
        }
    }
}

// KNN-based histogram equalization
void CDF(int* knn_index, int* intensity, int K, int N) {
    int* orig = (int*)malloc(N * sizeof(int));
    memcpy(orig, intensity, N * sizeof(int));

    for (int i = 0; i < N; i++) {
        int localBins[BINS];
        for (int j = 0; j < BINS; j++) localBins[j] = 0;

        for (int k = 0; k < K; k++) {
            int neighbor = knn_index[i * K + k];
            localBins[orig[neighbor]]++;
        }

        for (int j = 1; j < BINS; j++) {
            localBins[j] += localBins[j - 1];
        }

        int cmin = -1;
        for (int j = 0; j < BINS; j++) {
            if (localBins[j] > 0) {
                cmin = localBins[j];
                break;
            }
        }

        int Ii   = orig[i];
        int CiIi = localBins[Ii];

        intensity[i] = (cmin == K) ? Ii : ((CiIi - cmin) * 255) / (K - cmin);
    }

    free(orig);
}

// K-Means clustering + per-cluster histogram equalization
void KMEANS(int* px, int* py, int* pz, int* intensity, int N, int K, int T) {
    int* cluster = (int*)calloc(N, sizeof(int));
    int* centroid_px = (int*)malloc(K * sizeof(int));
    int* centroid_py = (int*)malloc(K * sizeof(int));
    int* centroid_pz = (int*)malloc(K * sizeof(int));

    // Initialize centroids as first K points
    for (int i = 0; i < K; i++) {
        centroid_px[i] = px[i];
        centroid_py[i] = py[i];
        centroid_pz[i] = pz[i];
    }

    int* sum_px = (int*)malloc(K * sizeof(int));
    int* sum_py = (int*)malloc(K * sizeof(int));
    int* sum_pz = (int*)malloc(K * sizeof(int));
    int* count  = (int*)malloc(K * sizeof(int));

    for (int iteration = 0; iteration < T; iteration++) {
        // Assign each point to nearest centroid
        int changed = 0;
        for (int i = 0; i < N; i++) {
            ll min_dist = LLONG_MAX;
            int best = -1;
            for (int k = 0; k < K; k++) {
                ll dx = (ll)px[i] - centroid_px[k];
                ll dy = (ll)py[i] - centroid_py[k];
                ll dz = (ll)pz[i] - centroid_pz[k];
                ll dist = dx * dx + dy * dy + dz * dz;
                if (dist < min_dist) {
                    min_dist = dist;
                    best = k;
                }
            }
            if (cluster[i] != best) {
                cluster[i] = best;
                changed = 1;
            }
        }

        if (!changed) break;

        // Update centroids
        for (int k = 0; k < K; k++) {
            sum_px[k] = sum_py[k] = sum_pz[k] = count[k] = 0;
        }
        for (int i = 0; i < N; i++) {
            int c = cluster[i];
            sum_px[c] += px[i];
            sum_py[c] += py[i];
            sum_pz[c] += pz[i];
            count[c]++;
        }
        for (int k = 0; k < K; k++) {
            if (count[k] > 0) {
                centroid_px[k] = sum_px[k] / count[k];
                centroid_py[k] = sum_py[k] / count[k];
                centroid_pz[k] = sum_pz[k] / count[k];
            }
        }
    }

    // Histogram equalization per cluster
    // Build histogram per cluster
    int* histogram = (int*)calloc(K * BINS, sizeof(int));
    for (int i = 0; i < N; i++) {
        histogram[cluster[i] * BINS + intensity[i]]++;
    }

    // CDF (prefix sum) per cluster
    for (int k = 0; k < K; k++) {
        for (int j = 1; j < BINS; j++) {
            histogram[k * BINS + j] += histogram[k * BINS + j - 1];
        }
    }

    // Remap intensities
    int* orig = (int*)malloc(N * sizeof(int));
    memcpy(orig, intensity, N * sizeof(int));

    for (int i = 0; i < N; i++) {
        int Ii = orig[i];
        int c = cluster[i];
        int CiIi = histogram[c * BINS + Ii];
        int m = histogram[c * BINS + 255];

        int cmin = 0;
        for (int j = 0; j < BINS; j++) {
            if (histogram[c * BINS + j] > 0) {
                cmin = histogram[c * BINS + j];
                break;
            }
        }

        intensity[i] = (cmin == m) ? Ii : ((CiIi - cmin) * 255) / (m - cmin);
    }

    free(orig);
    free(histogram);
    free(cluster);
    free(centroid_px); free(centroid_py); free(centroid_pz);
    free(sum_px); free(sum_py); free(sum_pz); free(count);
}

int main(int argc, char* argv[]) {
    auto total_start = chrono::high_resolution_clock::now();

    if (argc != 3) {
        cout << "Usage: ./seq input.txt <knn|approx_knn|kmeans>\n";
        return -1;
    }
    string filename = argv[1];
    string mode     = argv[2];

    int *h_px, *h_py, *h_pz, *h_intensity;
    int N, K, T;

    parse_input(filename, h_px, h_py, h_pz, h_intensity, N, K, T);

    if (mode == "knn") {
        int* knn_index = (int*)malloc(K * N * sizeof(int));

        auto knn_start = chrono::high_resolution_clock::now();
        KNN(h_px, h_py, h_pz, knn_index, N, K);
        auto knn_end = chrono::high_resolution_clock::now();
        cout << "KNN time: " << chrono::duration_cast<chrono::milliseconds>(knn_end - knn_start).count() << " ms\n";

        auto cdf_start = chrono::high_resolution_clock::now();
        CDF(knn_index, h_intensity, K, N);
        auto cdf_end = chrono::high_resolution_clock::now();
        cout << "CDF time: " << chrono::duration_cast<chrono::milliseconds>(cdf_end - cdf_start).count() << " ms\n";

        free(knn_index);
    } else if (mode == "approx_knn") {
        int* knn_index = (int*)malloc(K * N * sizeof(int));

        auto knn_start = chrono::high_resolution_clock::now();
        APPROX_KNN(h_px, h_py, h_pz, knn_index, N, K);
        auto knn_end = chrono::high_resolution_clock::now();
        cout << "Approx KNN time: " << chrono::duration_cast<chrono::milliseconds>(knn_end - knn_start).count() << " ms\n";

        auto cdf_start = chrono::high_resolution_clock::now();
        CDF(knn_index, h_intensity, K, N);
        auto cdf_end = chrono::high_resolution_clock::now();
        cout << "CDF time: " << chrono::duration_cast<chrono::milliseconds>(cdf_end - cdf_start).count() << " ms\n";

        free(knn_index);
    } else if (mode == "kmeans") {
        auto km_start = chrono::high_resolution_clock::now();
        KMEANS(h_px, h_py, h_pz, h_intensity, N, K, T);
        auto km_end = chrono::high_resolution_clock::now();
        cout << "KMeans time: " << chrono::duration_cast<chrono::milliseconds>(km_end - km_start).count() << " ms\n";
    } else {
        cout << "Unknown mode: " << mode << "\n";
        free(h_px); free(h_py); free(h_pz); free(h_intensity);
        return -1;
    }

    string out_file = mode + "_seq.txt";
    writeoutput(h_px, h_py, h_pz, h_intensity, N, out_file);

    auto total_end = chrono::high_resolution_clock::now();
    cout << "Total time: " << chrono::duration_cast<chrono::milliseconds>(total_end - total_start).count() << " ms\n";

    free(h_px); free(h_py); free(h_pz); free(h_intensity);
    return 0;
}
