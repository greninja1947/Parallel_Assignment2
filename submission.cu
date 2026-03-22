#include <bits/stdc++.h>
#define TILE_SIZE 256
#define MAX_K 128
#define BINS 256
#define ll long long int

using namespace std;

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
                for (int j = 1; j < K; j++)
                    if (localDist[j] > localDist[worstPos])
                        worstPos = j;

                if (dist < localDist[worstPos]) {
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

__global__ void CDF(int* d_knnindex, int* d_intensity, int K, int N){
    int i = blockDim.x * blockIdx.x +  threadIdx.x ;
    if(i>=N)return;
    int localBins[BINS];
    for(int j = 0; j < BINS; j++) localBins[j] = 0;

    for(int k = 0; k<K ;k++){
        int cur_intensity = d_intensity[d_knnindex[i*K + k]];
        localBins[cur_intensity]++;
    }
    int cmin = -1;

    for(int j = 1;j<BINS;j++){
        localBins[j] += localBins[j-1];
    }
    for(int j = 0;j<BINS;j++){
        if(localBins[j]>0){
            cmin = localBins[j];
            break;
        }
    }
    int Ii = d_intensity[i];
    int CiIi = localBins[Ii];

    d_intensity[i] = (cmin == K) ? Ii : ((CiIi - cmin) * 255) / (K - cmin) ; 
    
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

    int *d_px, *d_py, *d_pz, *d_intensity, *d_knnindex;

    cudaMalloc(&d_px, N*sizeof(int));
    cudaMalloc(&d_py, N*sizeof(int));
    cudaMalloc(&d_pz, N*sizeof(int));
    cudaMalloc(&d_intensity, N*sizeof(int));
    cudaMalloc(&d_knnindex, K*N*sizeof(int));


    cudaMemcpy(d_px,h_px,N*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_py,h_py,N*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pz,h_pz,N*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(d_intensity,h_intensity,N*sizeof(int),cudaMemcpyHostToDevice);

    int blockdim = 256;
    int griddim = (N + blockdim - 1) / blockdim;

    if(mode == "knn"){
        KNN<<<griddim,blockdim>>>(d_px,d_py,d_pz,d_knnindex,N,K);
    }
    cudaDeviceSynchronize();
    
    CDF<<<griddim,blockdim>>>(d_knnindex,d_intensity,K,N);
    cudaDeviceSynchronize();


    cudaMemcpy(h_intensity,d_intensity,N*sizeof(int),cudaMemcpyDeviceToHost);

    string out_file = mode+".txt";
    writeoutput(h_px,h_py,h_pz,h_intensity,N,out_file);


}