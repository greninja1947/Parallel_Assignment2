# Assignment 2: Parallel Programming using CUDA and OpenMP

**Deadline:** March 24, 2026

---

## Overview

Implement **local histogram equalization** on a 3D point cloud using CUDA (and optionally OpenMP for host-side optimizations). Three different approaches are required for defining the local neighborhood of each point.

---

## Problem Description

### Input: 3D Point Cloud

A point cloud consists of N points: `P = {p1, p2, ..., pN}`

Each point `pi` has:
- 3D spatial coordinates: `(xi, yi, zi)`
- An intensity value: `Ii ∈ {0, 1, ..., 255}`

Unlike regular images, point clouds are **not arranged on a 2D grid**. Neighborhood relationships are defined by **spatial proximity in 3D space** using Euclidean distance.

---

## Algorithm

For each approach, the same 3-step pipeline is applied:

### Step 1: Define Local Neighborhood

**(a) Exact KNN**

For each point `pi`, find its k-nearest neighbors:

```
Nk(i) = { j | pj is among the k closest points to pi }
```

Compute local histogram:
```
Hi[v] = |{ j ∈ Nk(i) : Ij = v }|,  v ∈ {0, 1, ..., 255}
```

**(b) Approximate KNN**

Find k-nearest neighbors using any approximate KNN algorithm that:
- Can be parallelized for significant GPU speedup
- Gives output close enough to exact KNN (evaluated by MAE threshold)

Report trade-offs between time and accuracy vs exact KNN.

**(c) K-Means Clustering**

- Define k clusters using parallelized K-means
- **Centroid initialization:** use the first k points
- **Stopping condition:** no change in assignments OR after T iterations
- Each point `pi` is assigned to a cluster `Nk(i)`
- Compute histogram over the cluster members

```
Hi[v] = |{ j ∈ Nk(i) : Ij = v }|
```

Note: for K-means, `m = |Nk(i)|` (cluster size), not necessarily k.

---

### Step 2: Local CDF Computation

For each point `pi`, compute its local cumulative distribution function:

```
Ci[v] = sum_{u=0}^{v} Hi[u]
```

Let:
```
Ci,min = min{ Ci[v] | Ci[v] > 0 }
```

---

### Step 3: Local Intensity Remapping

The new intensity of point `pi` is:

```
I'i = floor( (Ci[Ii] - Ci,min) / (m - Ci,min) * (L - 1) )
```

Where:
- `L = 256`
- `m = k` for KNN and Approximate KNN
- `m = |Nk(i)|` (cluster size) for K-Means

**Edge case:** If `m == Ci,min` (all neighbors have the same intensity), set `I'i = Ii`.

---

## Input Format

Single text file with:

```
N
K
T
x1 y1 z1 I1
x2 y2 z2 I2
...
xN yN zN IN
```

- **Line 1:** N — number of points
- **Line 2:** K — number of neighbors (KNN) / clusters (K-Means)
- **Line 3:** T — maximum iterations for K-Means
- **Next N lines:** `x y z I` where `(x,y,z) ∈ R³` and `I ∈ {0,...,255}`

**Constraints:**
- `N <= 1e5`
- `K <= 128`
- `T <= 50`

---

## Output Format

Three output files, each with N lines in the format:

```
x y z I'
```

Points must appear in the **same order as input**.

| File | Approach |
|------|----------|
| `knn.txt` | Exact KNN |
| `approx_knn.txt` | Approximate KNN |
| `kmeans.txt` | K-Means Clustering |

---

## Evaluation / Marks Breakdown

| Component | Marks | Correctness | Performance |
|-----------|-------|-------------|-------------|
| KNN Algorithm | 25 | 25% | 75% |
| Approximate KNN | 30 | 30% | 70% |
| K-Means Clustering | 30 | 25% | 75% |
| Report | 10 | — | — |

- **Performance for KNN and K-Means:** compared against the TA's sequential version
- **Performance for Approximate KNN:** compared against your own exact KNN version
- **Correctness for Approximate KNN:** evaluated using Mean Absolute Error (MAE) vs exact KNN output; threshold determined from all submissions
- **Performance is graded only if the program is correct**

---

## Implementation Requirements

- All computations must use **CUDA** (GPU) and optionally **OpenMP** (host-side)
- **NOT allowed:** Thrust, cuML, FAISS, or other high-level GPU libraries

---

## Files

| File | Description |
|------|-------------|
| `submission.cu` | CUDA implementation (KNN + CDF kernels) |
| `submission.cpp` | Sequential reference implementation |
| `input.txt` | Sample input (N=100, K=16, T=50) |
| `dataset_generator.py` | Script to generate test datasets |
| `config.txt` | Parameter constraints reference |

---

## Build & Run

### Sequential (CPU reference)

```bash
g++ -O2 -o seq submission.cpp
./seq input.txt knn
# Output: knn_seq.txt
```

### CUDA

```bash
nvcc -O2 -o a2 submission.cu
./a2 input.txt knn
# Output: knn.txt
```

---

## KNN Correctness Analysis

### Current `submission.cpp` KNN function

**Status: Logically correct** — the algorithm properly finds K nearest neighbors using a worst-replacement approach:

1. Initialize K slots with `LLONG_MAX` distance and index `-1`
2. For each candidate point (skipping self), compute squared Euclidean distance
3. Find the current worst (maximum distance) slot
4. If candidate is closer than the worst, replace it

This correctly fills all K slots with the nearest neighbors.

### CDF Function — **BUG FOUND**

**Bug:** In-place intensity modification while reading neighbor intensities.

```cpp
// BUG: d_intensity[i] is written in iteration i,
// but d_intensity[neighbor] is read in later iterations
// where neighbor < i has already been overwritten.
for (int i = 0; i < N; i++) {
    for (int k = 0; k < K; k++) {
        int cur_intensity = d_intensity[d_knnindex[i*K + k]]; // reads modified values!
        ...
    }
    d_intensity[i] = new_value; // overwrites original
}
```

**Fix:** Copy original intensities before the loop and read from the copy.

### Main Flow — **BUG FOUND**

```cpp
// BUG: KNN only runs when mode == "knn",
// but CDF always runs with potentially uninitialized knn_index
if(mode == "knn"){
    KNN(...);
}
CDF(knn_index, ...); // knn_index may be garbage!
```

**Fix:** Always run KNN before CDF in the sequential version.

---

## Report Checklist

- [ ] Implementation details and design decisions for each part
- [ ] Runtime analysis comparing all three approaches
- [ ] Trade-off analysis for Approximate KNN (accuracy vs speed)
- [ ] Algorithm choice justification for Approximate KNN
