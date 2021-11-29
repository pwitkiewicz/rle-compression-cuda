#include <iostream>
#include <chrono>
#include <filesystem>
#include <vector>
#include <fstream>
#include <iterator>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

int THREADS_PER_BLOCK = 512;
int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2;

__global__ void generateMask(uint8_t *input, uint32_t *mask, uint64_t blockCount)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (index == 0)
    {
        mask[0] = 1;
        for (int i = index + 1; i < blockCount; i += stride)
        {
            mask[i] = input[i] == input[i - 1] ? 0 : 1;
        }
        return;
    }

    for (int i = index; i < blockCount - 1; i += stride)
    {
        mask[i] = input[i] == input[i - 1] ? 0 : 1;
    }
}

void sequentialScan(uint32_t *output, uint32_t *input, uint64_t blockCount)
{
    output[0] = input[0];
    for (int j = 1; j < blockCount; ++j)
    {
        output[j] = input[j] + output[j - 1];
    }
}

__global__ void scan(uint32_t *output, uint32_t *input, uint32_t *sums, uint64_t n)
{
    int blockID = blockIdx.x;
    int threadID = threadIdx.x;
    int blockOffset = blockID * n;

    extern __shared__ int temp[];
    temp[2 * threadID] = input[blockOffset + (2 * threadID)];
    temp[2 * threadID + 1] = input[blockOffset + (2 * threadID) + 1];

    int offset = 1;
    for (int d = n >> 1; d > 0; d >>= 1)
    {
        __syncthreads();
        if (threadID < d)
        {
            int ai = offset * (2 * threadID + 1) - 1;
            int bi = offset * (2 * threadID + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }
    __syncthreads();


    if (threadID == 0) {
        sums[blockID] = temp[n - 1];
        temp[n - 1] = 0;
    }

    for (int d = 1; d < n; d *= 2)
    {
        offset >>= 1;
        __syncthreads();
        if (threadID < d)
        {
            int ai = offset * (2 * threadID + 1) - 1;
            int bi = offset * (2 * threadID + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    output[blockOffset + (2 * threadID)- 1] = temp[2 * threadID];
    output[blockOffset + (2 * threadID)] = temp[2 * threadID + 1];
} 

__global__ void add(uint32_t* output, uint32_t length, uint32_t* n) {
    int blockID = blockIdx.x;
    int threadID = threadIdx.x;
    int blockOffset = blockID * length - 1;

    output[blockOffset + threadID] += n[blockID - 1];
}

void compress(const string filename)
{
    ifstream inputFile;

    const uint64_t filesize = filesystem::file_size(filesystem::path(filename));
    const uint64_t blockCount = filesize / sizeof(uint8_t);
    const int gridSize = (blockCount + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    const int sharedMemArraySize = ELEMENTS_PER_BLOCK * sizeof(int);
    const int smallGridSize = (gridSize + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    uint32_t *scannedMask;
    //uint32_t *sequentialScannedMask = new uint32_t[blockCount];
    uint32_t *blockSums;
    uint32_t *scannedBlockSums;
    uint32_t* blockSumSums;
    uint32_t *mask;
    uint8_t *memblock;

    cudaMallocManaged(&memblock, blockCount * sizeof(uint8_t));
    cudaMallocManaged(&mask, blockCount * sizeof(uint32_t));
    cudaMallocManaged(&scannedMask, blockCount * sizeof(uint32_t));
    cudaMallocManaged(&blockSums, blockCount * sizeof(uint32_t));
    cudaMallocManaged(&scannedBlockSums, gridSize * sizeof(uint32_t));
    cudaMallocManaged(&blockSumSums, smallGridSize * sizeof(uint32_t));


    inputFile.open(filename, ios::binary);
    inputFile.read((char *)memblock, blockCount * sizeof(uint8_t));

    generateMask<<<gridSize, THREADS_PER_BLOCK>>>(memblock, mask, blockCount);
    cudaDeviceSynchronize();
    cout << +mask[0] << endl;

    //sequentialScan(sequentialScannedMask, mask, blockCount);

    scan<<<gridSize, THREADS_PER_BLOCK, sharedMemArraySize>>>(scannedMask, mask, blockSums, ELEMENTS_PER_BLOCK);
    cudaDeviceSynchronize();

    scan<<<smallGridSize, THREADS_PER_BLOCK, sharedMemArraySize>>>(scannedBlockSums, blockSums, blockSumSums, ELEMENTS_PER_BLOCK);
    cudaDeviceSynchronize();

    for (int i = 1; i < smallGridSize; i++) {
        blockSumSums[i] += blockSumSums[i - 1];
    }

    add<<<smallGridSize, ELEMENTS_PER_BLOCK>>>(scannedBlockSums, ELEMENTS_PER_BLOCK, blockSumSums);
    cudaDeviceSynchronize();

    add<<<gridSize, ELEMENTS_PER_BLOCK>>>(scannedMask, ELEMENTS_PER_BLOCK, scannedBlockSums);
    cudaDeviceSynchronize();

    cudaFree(memblock);
    cudaFree(mask);
    cudaFree(scannedMask);
    cudaFree(blockSums);
    cudaFree(scannedBlockSums);
}

int main(int argc, char const *argv[])
{
    string filename = "image.bmp";

    compress(filename);
}
