#include <bits/stdc++.h>
#define TILE_SIZE 256
#define MAX_K 128
#define BINS 256
#define ll long long int
#define MIN_COORD -10000
#define MAX_COORD 10000
#define COORD_RANGE 20000
using namespace std;

__device__ inline bool lex_less_d(int ax, int ay, int az, int bx, int by, int bz) {
    if (ax != bx) return ax < bx;
    if (ay != by) return ay < by;
    return az < bz;
}

__host__ __device__ inline int clamp_cell(int value, int grid_dim) {
    if (value < 0) return 0;
    if (value >= grid_dim) return grid_dim - 1;
    return value;
}

void parse_input(string filename,int*& h_px, int*& h_py, int*& h_pz, int*& h_intensity, int& N, int& K, int& T){
    ifstream inFile(filename);
    
    if(!inFile.is_open()){
        cout<<"could not open input file\n";
        return;
    }

    inFile >> N >> K >> T ;
    h_px = (int*)malloc(N * sizeof(int));
    h_py = (int*)malloc(N * sizeof(int));
    h_pz = (int*)malloc(N * sizeof(int));
    h_intensity = (int*)malloc(N * sizeof(int));

    for(int i = 0;i<N;i++){
        inFile >> h_px[i] >> h_py[i] >> h_pz[i] >> h_intensity[i];
    }

    inFile.close();

}

void writeoutput(int* h_px, int* h_py, int* h_pz, int* h_intensity,int N, string filename){
    ofstream outFile(filename);

    if(!outFile.is_open()){
        cout<<"error in creating output file\n";
    }

    for(int i = 0;i<N;i++){
        outFile << h_px[i]<<" "<<h_py[i]<<" "<<h_pz[i]<<" "<<h_intensity[i]<<"\n";
    }

    outFile.close();
}

__global__ void KNN(int* d_px, int* d_py, int* d_pz, int* knn_index, int N, int K) {

    __shared__ int s_px[TILE_SIZE];
    __shared__ int s_py[TILE_SIZE];
    __shared__ int s_pz[TILE_SIZE];

    int q = blockDim.x * blockIdx.x + threadIdx.x;

    int qx = 0, qy = 0, qz = 0;
    if (q < N) {
        qx = d_px[q];
        qy = d_py[q];
        qz = d_pz[q];
    }

    ll localDist[MAX_K];
    int localIdx[MAX_K];
    for (int j = 0; j < K; j++) {
        localDist[j] = LLONG_MAX;
        localIdx[j]  = -1;
    }

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {

        int refIdx = t * TILE_SIZE + threadIdx.x;
        if (refIdx < N) {
            s_px[threadIdx.x] = d_px[refIdx];  
            s_py[threadIdx.x] = d_py[refIdx]; 
            s_pz[threadIdx.x] = d_pz[refIdx];  
        } else {
            s_px[threadIdx.x] = INT_MAX;
            s_py[threadIdx.x] = INT_MAX;
            s_pz[threadIdx.x] = INT_MAX;
        }
        __syncthreads();  

        if (q < N) {
            for (int i = 0; i < TILE_SIZE; i++) {
                int globalIdx = t * TILE_SIZE + i;
                if (globalIdx >= N) break;
                if (globalIdx == q) continue;
                ll dx = s_px[i] - qx;  
                ll dy = s_py[i] - qy;
                ll dz = s_pz[i] - qz;
                ll dist = dx*dx + dy*dy + dz*dz;

                int worstPos = 0;
                for (int j = 1; j < K; j++) {
                    if (localDist[j] > localDist[worstPos]) {
                        worstPos = j;
                    } else if (localDist[j] == localDist[worstPos] && localIdx[j] != -1 && localIdx[worstPos] != -1) {
                        int ji = localIdx[j], wi = localIdx[worstPos];
                        if (lex_less_d(d_px[wi], d_py[wi], d_pz[wi], d_px[ji], d_py[ji], d_pz[ji]))
                            worstPos = j;
                    }
                }

                if (dist < localDist[worstPos] ||
                    (dist == localDist[worstPos] && localIdx[worstPos] != -1 &&
                     lex_less_d(s_px[i], s_py[i], s_pz[i], d_px[localIdx[worstPos]], d_py[localIdx[worstPos]], d_pz[localIdx[worstPos]]))) {
                    localDist[worstPos] = dist;
                    localIdx[worstPos]  = globalIdx;
                }
            }
        }
        __syncthreads(); 
    }


    if (q < N) {
        for (int j = 0; j < K; j++) {
            knn_index[q*K + j]    = localIdx[j];
        }
    }
}

__constant__ int d_centroid_px[MAX_K];
__constant__ int d_centroid_py[MAX_K];
__constant__ int d_centroid_pz[MAX_K];

__global__ void KMEANS_assign_cluster(int* d_px, int* d_py, int* d_pz, int* d_cluster, int* d_changed, int N, int K){
    int i = blockDim.x * blockIdx.x + threadIdx.x ;
    if(i>=N)return;
    int px = d_px[i];
    int py = d_py[i];
    int pz = d_pz[i];

    int min_dist_centroid = -1;
    ll min_dist = LLONG_MAX;
    ll curr_dist = 0;
    for(int k=0;k<K;k++){
        curr_dist = (ll)(px - d_centroid_px[k])*(px - d_centroid_px[k]) + (ll)(py - d_centroid_py[k])*(py - d_centroid_py[k]) + (ll)(pz - d_centroid_pz[k])*(pz - d_centroid_pz[k]);
        if(curr_dist < min_dist) {
            min_dist = curr_dist;
            min_dist_centroid = k;
        } else if (curr_dist == min_dist) {
            if (lex_less_d(d_centroid_px[k], d_centroid_py[k], d_centroid_pz[k],
                           d_centroid_px[min_dist_centroid], d_centroid_py[min_dist_centroid], d_centroid_pz[min_dist_centroid]))
                min_dist_centroid = k;
        }
    }

    if(d_cluster[i] != min_dist_centroid){
        d_cluster[i] = min_dist_centroid;
        atomicExch(d_changed, 1);
    }
}   


__global__ void KMEANS_update_centroid(int* d_px, int* d_py, int* d_pz, int* d_cluster, int*d_sum_px, int*d_sum_py, int* d_sum_pz,int* d_count, int N){
    int i = blockDim.x * blockIdx.x +  threadIdx.x ;
    if(i>=N)return;
    int cluster_id = d_cluster[i];
    atomicAdd(&d_sum_px[cluster_id],d_px[i]);
    atomicAdd(&d_sum_py[cluster_id],d_py[i]);
    atomicAdd(&d_sum_pz[cluster_id],d_pz[i]);
    atomicAdd(&d_count[cluster_id],1);
}

__global__ void KMEANS_divide_centroid(int* d_sum_px, int* d_sum_py, int* d_sum_pz, int*d_count, int*d_new_centroid_px,int*d_new_centroid_py,int*d_new_centroid_pz,int K){
    int i = blockDim.x * blockIdx.x +  threadIdx.x ;
    if(i>=K)return;
    if(d_count[i] > 0){
        d_new_centroid_px[i] = d_sum_px[i] / d_count[i] ;
        d_new_centroid_py[i] = d_sum_py[i] / d_count[i] ;
        d_new_centroid_pz[i] = d_sum_pz[i] / d_count[i] ;
    } else {
        d_new_centroid_px[i] = d_centroid_px[i];
        d_new_centroid_py[i] = d_centroid_py[i];
        d_new_centroid_pz[i] = d_centroid_pz[i];
    }
}

__global__ void KMEANS_make_histogram(int* d_cluster, int* d_intensity, int* d_histogram, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i>=N)return;
    int cluster_id = d_cluster[i];
    int intensity = d_intensity[i];
    atomicAdd(&d_histogram[cluster_id * BINS + intensity],1);
}

__global__ void KMEANS_do_cdf(int* d_histogram,int K){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i>=K)return;
    for(int j = 1;j<BINS;j++){
        d_histogram[i*BINS + j] += d_histogram[i*BINS + j-1];
    }
    
}

__global__ void KMEANS_intensity_remap(int* d_cluster,  int* d_histogram, int *d_intensity, int* d_intensity_out, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x ; 
    if(i>=N)return;
    int Ii = d_intensity[i];
    int cluster_id = d_cluster[i];
    int CiIi = d_histogram[cluster_id * BINS + Ii];
    int m = d_histogram[cluster_id * BINS + 255];
    int cmin = 0;
    for(int j = 0;j<BINS;j++){
        if(d_histogram[cluster_id * BINS + j]>0){
            cmin = d_histogram[cluster_id * BINS + j];
            break;
        }
    }

    int val = (cmin == m) ? Ii : ((CiIi - cmin) * 255) / (m - cmin);
    d_intensity_out[i] = max(0, min(255, val));
}

__global__ void CDF(int* d_knnindex, int* d_intensity, int* d_intensity_out,  int K, int N){
    int i = blockDim.x * blockIdx.x +  threadIdx.x ;
    if(i>=N)return;
    int localBins[BINS];
    for(int j = 0; j < BINS; j++) localBins[j] = 0;

    localBins[d_intensity[i]]++;  // include the point itself

    for(int k = 0; k<K ;k++){
        int cur_intensity = d_intensity[d_knnindex[i*K + k]];
        localBins[cur_intensity]++;
    }

    for(int j = 1;j<BINS;j++){
        localBins[j] += localBins[j-1];
    }

    int cmin = -1;
    int total = K + 1;  // K neighbors + self
    for(int j = 0;j<BINS;j++){
        if(localBins[j]>0){
            cmin = localBins[j];
            break;
        }
    }
    int Ii = d_intensity[i];
    int CiIi = localBins[Ii];

    int val = (cmin == total) ? Ii : ((CiIi - cmin) * 255) / (total - cmin);
    d_intensity_out[i] = max(0, min(255, val)); 
    
}

__global__ void AKNN_assign_cellid(int* d_cellid, int* d_px, int* d_py, int* d_pz,float cell_size, int GRID_DIM, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x ; 
    if(i>=N)return;
    int cx = clamp_cell((int)((d_px[i] - MIN_COORD) / cell_size), GRID_DIM);
    int cy = clamp_cell((int)((d_py[i] - MIN_COORD) / cell_size), GRID_DIM);
    int cz = clamp_cell((int)((d_pz[i] - MIN_COORD) / cell_size), GRID_DIM);
    d_cellid[i] = cx + cy*GRID_DIM + cz*GRID_DIM*GRID_DIM; 
}


__global__ void AKNN_cell_count(int* d_cellid, int* d_cell_count, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i>=N)return; 
    int cellid = d_cellid[i];
    atomicAdd(&d_cell_count[cellid],1);
}   

__global__ void AKNN_prefix(int* d_cell_count, int* d_cell_start, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x ; 
    if(i>0)return;
    d_cell_start[0] = 0;
    for(int j = 1;j<N;j++){
        d_cell_start[j] = d_cell_start[j-1] + d_cell_count[j-1];
    }
}   

__global__ void AKNN_scatter(int* d_cellid, int* d_cell_start, int* d_counter,int* d_sorted, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int cellid = d_cellid[i];
        int slot = atomicAdd(&d_counter[cellid], 1); 
        d_sorted[d_cell_start[cellid] + slot] = i;
    }
}


__global__ void AKNN_getKNN(int* d_cellid, int* d_cell_start, int* d_cell_count, int* d_sorted, int* d_px, int* d_py, int* d_pz, int* d_knn_index, int N, int K, int GRID_DIM){
    int i = blockDim.x * blockIdx.x + threadIdx.x ;
    if(i>=N)return;
    int cellid = d_cellid[i];
    ll localDist[MAX_K];
    int localIdx[MAX_K];

    int cx = cellid % GRID_DIM;
    cellid = cellid / GRID_DIM;
    int cy = cellid % GRID_DIM;
    cellid = cellid / GRID_DIM;
    int cz = cellid % GRID_DIM;

    for (int j = 0; j < K; j++) {
        localDist[j] = LLONG_MAX;
        localIdx[j]  = -1;
    }

    int px = d_px[i];
    int py = d_py[i];
    int pz = d_pz[i];

    int total_points = 0;
    for (int ddx = -1; ddx <= 1; ddx++) {
        for (int ddy = -1; ddy <= 1; ddy++) {
            for (int ddz = -1; ddz <= 1; ddz++) {

                int nx = cx + ddx;
                int ny = cy + ddy;
                int nz = cz + ddz;

                if (nx < 0 || nx >= GRID_DIM) continue;
                if (ny < 0 || ny >= GRID_DIM) continue;
                if (nz < 0 || nz >= GRID_DIM) continue;

                int newcellid = nx + ny*GRID_DIM + nz*GRID_DIM*GRID_DIM;
                int startidx = d_cell_start[newcellid];
                int size = d_cell_count[newcellid];
                total_points += size;

                for(int j = 0; j < size; j++){
                    int newpt = d_sorted[startidx + j];
                    if(newpt == i) continue;

                    ll new_dist = (ll)(d_px[newpt] - px)*(d_px[newpt] - px) + (ll)(d_py[newpt] - py)*(d_py[newpt] - py) + (ll)(d_pz[newpt] - pz)*(d_pz[newpt] - pz);

                    int worstPos = 0;
                    for (int jj = 1; jj < K; jj++) {
                        if (localDist[jj] > localDist[worstPos]) {
                            worstPos = jj;
                        } else if (localDist[jj] == localDist[worstPos] && localIdx[jj] != -1 && localIdx[worstPos] != -1) {
                            int ji = localIdx[jj], wi = localIdx[worstPos];
                            if (lex_less_d(d_px[wi], d_py[wi], d_pz[wi], d_px[ji], d_py[ji], d_pz[ji]))
                                worstPos = jj;
                        }
                    }

                    if (new_dist < localDist[worstPos] ||
                        (new_dist == localDist[worstPos] && localIdx[worstPos] != -1 &&
                         lex_less_d(d_px[newpt], d_py[newpt], d_pz[newpt], d_px[localIdx[worstPos]], d_py[localIdx[worstPos]], d_pz[localIdx[worstPos]]))) {
                        localDist[worstPos] = new_dist;
                        localIdx[worstPos]  = newpt;
                    }
                }
            }
        }
    }

    if(total_points < K){
        for (int ddx = -2; ddx <= 2; ddx++) {
            for (int ddy = -2; ddy <= 2; ddy++) {
                for (int ddz = -2; ddz <= 2; ddz++) {
                    if (ddx >= -1 && ddx <= 1 && ddy >= -1 && ddy <= 1 && ddz >= -1 && ddz <= 1) continue;

                    int nx = cx + ddx;
                    int ny = cy + ddy;
                    int nz = cz + ddz;

                    if (nx < 0 || nx >= GRID_DIM) continue;
                    if (ny < 0 || ny >= GRID_DIM) continue;
                    if (nz < 0 || nz >= GRID_DIM) continue;

                    int newcellid = nx + ny*GRID_DIM + nz*GRID_DIM*GRID_DIM;
                    int startidx = d_cell_start[newcellid];
                    int size = d_cell_count[newcellid];

                    for(int j = 0; j < size; j++){
                        int newpt = d_sorted[startidx + j];
                        if(newpt == i) continue;

                        ll new_dist = (ll)(d_px[newpt] - px)*(d_px[newpt] - px) + (ll)(d_py[newpt] - py)*(d_py[newpt] - py) + (ll)(d_pz[newpt] - pz)*(d_pz[newpt] - pz);

                        int worstPos = 0;
                        for (int jj = 1; jj < K; jj++) {
                            if (localDist[jj] > localDist[worstPos]) {
                                worstPos = jj;
                            } else if (localDist[jj] == localDist[worstPos] && localIdx[jj] != -1 && localIdx[worstPos] != -1) {
                                int ji = localIdx[jj], wi = localIdx[worstPos];
                                if (lex_less_d(d_px[wi], d_py[wi], d_pz[wi], d_px[ji], d_py[ji], d_pz[ji]))
                                    worstPos = jj;
                            }
                        }

                        if (new_dist < localDist[worstPos] ||
                            (new_dist == localDist[worstPos] && localIdx[worstPos] != -1 &&
                             lex_less_d(d_px[newpt], d_py[newpt], d_pz[newpt], d_px[localIdx[worstPos]], d_py[localIdx[worstPos]], d_pz[localIdx[worstPos]]))) {
                            localDist[worstPos] = new_dist;
                            localIdx[worstPos]  = newpt;
                        }
                    }
                }
            }
        }
    }

    int found_neighbors = 0;
    for (int j = 0; j < K; j++) {
        found_neighbors += (localIdx[j] != -1);
    }

    if (found_neighbors < K) {
        for (int j = 0; j < K; j++) {
            localDist[j] = LLONG_MAX;
            localIdx[j] = -1;
        }

        for (int newpt = 0; newpt < N; newpt++) {
            if (newpt == i) continue;

            ll new_dist = (ll)(d_px[newpt] - px) * (d_px[newpt] - px) +
                          (ll)(d_py[newpt] - py) * (d_py[newpt] - py) +
                          (ll)(d_pz[newpt] - pz) * (d_pz[newpt] - pz);

            int worstPos = 0;
            for (int jj = 1; jj < K; jj++) {
                if (localDist[jj] > localDist[worstPos]) {
                    worstPos = jj;
                } else if (localDist[jj] == localDist[worstPos] && localIdx[jj] != -1 && localIdx[worstPos] != -1) {
                    int ji = localIdx[jj], wi = localIdx[worstPos];
                    if (lex_less_d(d_px[wi], d_py[wi], d_pz[wi], d_px[ji], d_py[ji], d_pz[ji]))
                        worstPos = jj;
                }
            }

            if (new_dist < localDist[worstPos] ||
                (new_dist == localDist[worstPos] && localIdx[worstPos] != -1 &&
                 lex_less_d(d_px[newpt], d_py[newpt], d_pz[newpt], d_px[localIdx[worstPos]], d_py[localIdx[worstPos]], d_pz[localIdx[worstPos]]))) {
                localDist[worstPos] = new_dist;
                localIdx[worstPos] = newpt;
            }
        }
    }

    for (int j = 0; j < K; j++) {
        d_knn_index[i*K + j] = localIdx[j];
    }
}
int main(int argc, char* argv[]){
    if(argc !=3){
        cout<<"Usage: ./a2 input.txt mode\n";
        return -1;
    }
    string filename = argv[1];
    string mode = argv[2];

    int *h_px, *h_py, *h_pz, *h_intensity;
    int N,K,T;

    
    parse_input(filename,h_px,h_py,h_pz,h_intensity,N,K,T);

    int *d_px, *d_py, *d_pz, *d_intensity, *d_intensity_out;

    cudaMalloc(&d_px, N*sizeof(int));
    cudaMalloc(&d_py, N*sizeof(int));
    cudaMalloc(&d_pz, N*sizeof(int));
    cudaMalloc(&d_intensity, N*sizeof(int));
    cudaMalloc(&d_intensity_out, N*sizeof(int));
    
    
    cudaMemcpy(d_px,h_px,N*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_py,h_py,N*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pz,h_pz,N*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_intensity,h_intensity,N*sizeof(int),cudaMemcpyHostToDevice);
    
    int blockdim = 256;
    int griddim = (N + blockdim - 1) / blockdim;
    
    if(mode == "knn"){
        int* d_knnindex;
        cudaMalloc(&d_knnindex, K*N*sizeof(int));
        KNN<<<griddim,blockdim>>>(d_px,d_py,d_pz,d_knnindex,N,K);
        cudaDeviceSynchronize();
        CDF<<<griddim,blockdim>>>(d_knnindex,d_intensity,d_intensity_out,K,N);
        cudaDeviceSynchronize();
        cudaMemcpy(h_intensity,d_intensity_out,N*sizeof(int),cudaMemcpyDeviceToHost);
        
    }else if(mode == "kmeans"){
        int* d_cluster, *d_sum_px, *d_sum_py,*d_sum_pz,*d_count, *d_new_centroid_px, *d_new_centroid_py, *d_new_centroid_pz;
        cudaMalloc(&d_cluster, N*sizeof(int));
        cudaMemset(d_cluster, 0xFF, N * sizeof(int));
        cudaMalloc(&d_sum_px, K*sizeof(int));
        cudaMalloc(&d_sum_py, K*sizeof(int));
        cudaMalloc(&d_sum_pz, K*sizeof(int));
        cudaMalloc(&d_count, K*sizeof(int));
        cudaMalloc(&d_new_centroid_px, K*sizeof(int));
        cudaMalloc(&d_new_centroid_py, K*sizeof(int));
        cudaMalloc(&d_new_centroid_pz, K*sizeof(int));
        int h_centroid_px[MAX_K],h_centroid_py[MAX_K],h_centroid_pz[MAX_K];
        for(int i = 0;i<K;i++){
            h_centroid_px[i] = h_px[i];
            h_centroid_py[i] = h_py[i];
            h_centroid_pz[i] = h_pz[i];
        }

        cudaMemcpyToSymbol(d_centroid_px,h_centroid_px,K*sizeof(int),0,cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(d_centroid_py,h_centroid_py,K*sizeof(int),0,cudaMemcpyHostToDevice);
        cudaMemcpyToSymbol(d_centroid_pz,h_centroid_pz,K*sizeof(int),0,cudaMemcpyHostToDevice);

        int* d_changed;
        cudaMalloc(&d_changed, sizeof(int));
        int h_changed = 0;

        for(int iteration = 0;iteration<T;iteration++){
            cudaMemset(d_changed, 0, sizeof(int));
            KMEANS_assign_cluster<<<griddim,blockdim>>>(d_px,d_py,d_pz,d_cluster,d_changed,N,K);
            cudaDeviceSynchronize();

            cudaMemcpy(&h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
            if(h_changed == 0) break;

            cudaMemset(d_sum_px, 0, K * sizeof(int));
            cudaMemset(d_sum_py, 0, K * sizeof(int));
            cudaMemset(d_sum_pz, 0, K * sizeof(int));
            cudaMemset(d_count,  0, K * sizeof(int));
            KMEANS_update_centroid<<<griddim,blockdim>>>(d_px,d_py,d_pz,d_cluster,d_sum_px, d_sum_py, d_sum_pz,d_count,N);
            cudaDeviceSynchronize();

            KMEANS_divide_centroid<<<1,blockdim>>>(d_sum_px,d_sum_py,d_sum_pz,d_count,d_new_centroid_px,d_new_centroid_py,d_new_centroid_pz,K);
            cudaDeviceSynchronize();

            cudaMemcpyToSymbol(d_centroid_px,d_new_centroid_px,K*sizeof(int),0,cudaMemcpyDeviceToDevice);
            cudaMemcpyToSymbol(d_centroid_py,d_new_centroid_py,K*sizeof(int),0,cudaMemcpyDeviceToDevice);
            cudaMemcpyToSymbol(d_centroid_pz,d_new_centroid_pz,K*sizeof(int),0,cudaMemcpyDeviceToDevice);
        }

        int* d_histogram;
        cudaMalloc(&d_histogram, K * BINS * sizeof(int));
        cudaMemset(d_histogram, 0, K * BINS * sizeof(int));

        KMEANS_make_histogram<<<griddim,blockdim>>>(d_cluster, d_intensity, d_histogram, N);
        cudaDeviceSynchronize();

        KMEANS_do_cdf<<<1, K>>>(d_histogram, K);
        cudaDeviceSynchronize();

        KMEANS_intensity_remap<<<griddim,blockdim>>>(d_cluster, d_histogram, d_intensity, d_intensity_out, N);
        cudaDeviceSynchronize();

        cudaMemcpy(h_intensity, d_intensity_out, N*sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(d_changed);
        cudaFree(d_histogram);
        cudaFree(d_cluster);
        cudaFree(d_sum_px);
        cudaFree(d_sum_py);
        cudaFree(d_sum_pz);
        cudaFree(d_count);
        cudaFree(d_new_centroid_px);
        cudaFree(d_new_centroid_py);
        cudaFree(d_new_centroid_pz);
    }else if(mode == "approx_knn"){
        int GRID_DIM = (int)cbrt(27.0f * N / (2.0f * K)) + 1;
        float cell_size = (float)COORD_RANGE / GRID_DIM;
        int TOTAL_CELLS = GRID_DIM * GRID_DIM * GRID_DIM;

        int *d_cellid, *d_cell_count, *d_cell_start, *d_counter, *d_sorted, *d_knnindex;
        cudaMalloc(&d_cellid, N * sizeof(int));
        cudaMalloc(&d_cell_count, TOTAL_CELLS * sizeof(int));
        cudaMalloc(&d_cell_start, TOTAL_CELLS * sizeof(int));
        cudaMalloc(&d_counter, TOTAL_CELLS * sizeof(int));
        cudaMalloc(&d_sorted, N * sizeof(int));
        cudaMalloc(&d_knnindex, K * N * sizeof(int));

        cudaMemset(d_cell_count, 0, TOTAL_CELLS * sizeof(int));
        cudaMemset(d_counter, 0, TOTAL_CELLS * sizeof(int));

        AKNN_assign_cellid<<<griddim, blockdim>>>(d_cellid, d_px, d_py, d_pz, cell_size, GRID_DIM, N);
        cudaDeviceSynchronize();

        AKNN_cell_count<<<griddim, blockdim>>>(d_cellid, d_cell_count, N);
        cudaDeviceSynchronize();

        AKNN_prefix<<<1, 1>>>(d_cell_count, d_cell_start, TOTAL_CELLS);
        cudaDeviceSynchronize();

        AKNN_scatter<<<griddim, blockdim>>>(d_cellid, d_cell_start, d_counter, d_sorted, N);
        cudaDeviceSynchronize();

        AKNN_getKNN<<<griddim, blockdim>>>(d_cellid, d_cell_start, d_cell_count, d_sorted, d_px, d_py, d_pz, d_knnindex, N, K, GRID_DIM);
        cudaDeviceSynchronize();

        CDF<<<griddim, blockdim>>>(d_knnindex, d_intensity, d_intensity_out, K, N);
        cudaDeviceSynchronize();

        cudaMemcpy(h_intensity, d_intensity_out, N * sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(d_cellid);
        cudaFree(d_cell_count);
        cudaFree(d_cell_start);
        cudaFree(d_counter);
        cudaFree(d_sorted);
        cudaFree(d_knnindex);
    }
    
    string out_file = mode+".txt";
    writeoutput(h_px,h_py,h_pz,h_intensity,N,out_file);

}
