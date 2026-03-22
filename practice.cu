#include <cuda_runtime.h>
#include <stdio.h>
#include <bits/stdc++.h>


__global__ void doublefun(int* arr, int N){

    __shared__ int tile[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x ;

    if(i<N){
        tile[tid] = arr[i];
    }else{
        tile[tid] = 0;
    }
    __syncthreads();
  if(i==N-1 || tid == blockDim.x -1 ){
    arr[i] = tile[tid];
  }else{
      arr[i] = tile[tid] + tile[tid+1];
  }
}

#define TILE 16 

__global__ void matrix(int* d_a, int* d_b, int* d_c, int N){
  __shared__ int tileA[TILE][TILE];
  __shared__ int tileB[TILE][TILE];

  int row = blockIdx.y * TILE + threadIdx.y;
  int col = blockIdx.x * TILE + threadIdx.x;

  int sum = 0;
  for(int tile = 0;tile < (N+TILE-1)/TILE;tile++){
    int colA = tile * TILE + threadIdx.x;
    int rowB = tile * TILE + threadIdx.y;

    tileA[threadIdx.y][threadIdx.x] = (colA < N && row < N) ? d_a[row*N + (tile * TILE + threadIdx.x)] : 0;
    tileB[threadIdx.y][threadIdx.x] = (col < N && rowB < N) ? d_b[(tile * TILE + threadIdx.y)*N + col] : 0;
    __syncthreads();

    for(int k = 0;k<TILE;k++){
      sum += tileA[threadIdx.y][k]*tileB[k][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < N && col < N){
    d_c[row*N + col] = sum;
  }
  
}


__global__ void vector(int* d_a, int* d_b, int* d_blocksums, int N){
  __shared__ int tile[256];
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  tile[threadIdx.x] = d_a[index]*d_b[index];
  __syncthreads();
  int stride = 128;
  while(stride){
    if(threadIdx.x<stride){
      tile[threadIdx.x]+=tile[threadIdx.x + stride];
    }
    __syncthreads();
    stride = stride >> 1;
  }
  d_blocksums[blockIdx.x]=tile[0];
}

__const__ float d_query[3];
float INFINITY = 1e20;
__global__ squared_distance(float* d_px, float* d_py, float* d_pz, pair<int,float>*d_block_mins ,int N){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ tileIndex[blockDim.x];
  __shared__ tileMin[blockDim.x];
  float dx = (d_px[i] - d_query[0])**2;
  float dy = (d_py[i] - d_query[1])**2;
  float dz = (d_pz[i] - d_query[2])**2;

  tileIndex[threadIdx.x] = (i<N) ? i : -1;
  tileMin[threadIdx.x] = (i < N) ? (dx+dy+dz) : INFINTY;
  __syncthreads();

  int min_index = -1;
  float min_element = INFINTY;

  for(int stride = blockDim.x / 2 ; stride > 0; stride>>1){
    if(threadIdx.x < stride){
      if(tileMin[threadIdx.x] > tileMin[threadIdx.x + stride]){
        tileMin[threadIdx.x] = tileMin[threadIdx.x + stride];
        tileIndex[threadIdx.x] = tileIndex[threadIdx.x + stride];
      }
    }
  }

  d_block_mins[blockIdx.x] = {tileIndex,tileMin[0]};

}
#define TILE_SIZE 256
#define MAX_K 50
using namespace std;
__global__ void KNN(float* d_px, float* d_py, float* d_pz,
                           float* q_px, float* q_py, float* q_pz,
                           int* knn_index, float* knn_distance,
                           int N, int Q, int K) {

    // Shared memory tile for reference points
    __shared__ float s_px[TILE_SIZE];
    __shared__ float s_py[TILE_SIZE];
    __shared__ float s_pz[TILE_SIZE];

    int q = blockDim.x * blockIdx.x + threadIdx.x;

    // Load query point into registers
    float qx = 0, qy = 0, qz = 0;
    if (q < Q) {
        qx = q_px[q];
        qy = q_py[q];
        qz = q_pz[q];
    }

    // Local top-K
    float localDist[MAX_K];
    int   localIdx[MAX_K];
    for (int j = 0; j < K; j++) {
        localDist[j] = 1e20f;
        localIdx[j]  = -1;
    }

    // Loop over tiles of reference points
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {

        // Cooperatively load tile into shared memory
        int refIdx = t * TILE_SIZE + threadIdx.x;
        if (refIdx < N) {
            s_px[threadIdx.x] = d_px[refIdx];  // coalesced ✅
            s_py[threadIdx.x] = d_py[refIdx];  // coalesced ✅
            s_pz[threadIdx.x] = d_pz[refIdx];  // coalesced ✅
        } else {
            s_px[threadIdx.x] = 1e20f;  // pad out of bounds
            s_py[threadIdx.x] = 1e20f;
            s_pz[threadIdx.x] = 1e20f;
        }
        __syncthreads();  // wait for full tile to load

        // Each thread computes distances to ALL points in tile
        if (q < Q) {
            for (int i = 0; i < TILE_SIZE; i++) {
                int globalIdx = t * TILE_SIZE + i;
                if (globalIdx >= N) break;

                float dx = s_px[i] - qx;  // shared memory ✅ fast
                float dy = s_py[i] - qy;
                float dz = s_pz[i] - qz;
                float dist = dx*dx + dy*dy + dz*dz;

                // Update top-K
                int worstPos = 0;
                for (int j = 1; j < K; j++)
                    if (localDist[j] > localDist[worstPos])
                        worstPos = j;

                if (dist < localDist[worstPos]) {
                    localDist[worstPos] = dist;
                    localIdx[worstPos]  = globalIdx;
                }
            }
        }
        __syncthreads();  // before loading next tile ✅
    }

    // Write results
    if (q < Q) {
        for (int j = 0; j < K; j++) {
            knn_distance[q*K + j] = localDist[j];
            knn_index[q*K + j]    = localIdx[j];
        }
    }
}



int main_for_KNN(){
  int N = 1024;
  float h_px[N], h_py[N], h_pz[N];
  for(int i = 0;i<N;i++){
    h_px[i]=i;
    h_py[i]=i;
    h_pz[i]=i;
  } 
  int Q = 10;
  float h_qx[N],h_qy[N],h_qz[N];
  for(int i = 0;i<Q;i++){
    h_qx[i] = 100*i;
    h_qy[i] = 100*i;
    h_qz[i] = 100*i;
  }
  float *d_px, *d_py, *d_pz;
  float *d_qx, *d_qy, *d_qz;
  cudaMalloc(&d_px, N*sizeof(float));
  cudaMalloc(&d_py, N*sizeof(float));
  cudaMalloc(&d_pz, N*sizeof(float));

  cudaMalloc(&d_qx, Q*sizeof(float));
  cudaMalloc(&d_qy, Q*sizeof(float));
  cudaMalloc(&d_qz, Q*sizeof(float));


  cudaMemcpy(d_px,h_px,N*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_py,h_py,N*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_pz,h_pz,N*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_qx,h_qx,Q*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_qy,h_qy,Q*sizeof(float),cudaMemcpyHostToDevice);
  cudaMemcpy(d_qz,h_qz,Q*sizeof(float),cudaMemcpyHostToDevice);

  int blockDim = 256;
  int gridDim = (Q + blockDim - 1)/ blockDim;
  
  float* d_knnDistance;
  int* d_knnIndex;
  
  int K = 3;
  float h_knnDistance[Q*K];
  int h_knnIndex[Q*K];


  cudaMalloc(&d_knnDistance , Q*K*sizeof(float));
  cudaMalloc(&d_knnIndex, Q*K*sizeof(int));


  KNN<<<gridDim,blockDim>>>(d_px,d_py,d_pz,d_qx,d_qy,d_qz,d_knnIndex,d_knnDistance,N,Q,K);
  cudaDeviceSynchronize();
  
  cudaMemcpy(h_knnDistance,d_knnDistance,K*Q*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(h_knnIndex,d_knnIndex,K*Q*sizeof(int),cudaMemcpyDeviceToHost);
  
  for(int q= 0;q<Q;q++){
    cout<<"The "<<K<<" neares points to the Query "<<q<<": "<<h_qx[q]<<", "<<h_qy[q]<<", "<<h_qz[q]<<"are: \n";
    for(int k = 0;k<K;k++){
      cout<<"Point "<<h_knnIndex[q*K + k]<<" \n";
    }
  }

}

#define BINS 256

__global__ void buildHistogram(int* data, int* bins, int N){
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if(i>=N)return;
  __shared__ int local_bins[256];
  /// assumin i will run code with only 256 threads
  local_bins[threadIdx.x] = 0;
  __syncthreads();

  atomicAdd(&local_bins[data[i]],1);
  __syncthreads();
  
  atomicAdd(&bins[threadIdx.x],local_bins[threadIdx.x]);
}

int main() {
  int N = 1024  
  int *h_data, *h_bins;
  int *d_data, *d_bins;

  h_data = (int*)malloc(N * sizeof(int));
  h_bins = (int*)calloc(BINS, sizeof(int));  // initialized to 0

  // Generate random data 0-255
  for (int i = 0; i < N; i++)
      h_data[i] = i % BINS;

  cudaMalloc(&d_data, N * sizeof(int));
  cudaMalloc(&d_bins, BINS * sizeof(int));

  cudaMemcpy(d_data, h_data, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(d_bins, 0, BINS * sizeof(int));  

  int blockSize = 256;
  int gridSize  = (N + blockSize - 1) / blockSize;

  histShared<<<gridSize, blockSize>>>(d_data, d_bins, N);
  cudaDeviceSynchronize();

  cudaMemcpy(h_bins, d_bins, BINS*sizeof(int), cudaMemcpyDeviceToHost);
  int cdf[256];
  cdf[0]=h_bins[0];
  for(int i = 1;i < 256; i++){
    cdf[i] = cdf[i-1] + h_bins[i];
  }

  
}