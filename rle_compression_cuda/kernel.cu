#include <iostream>
#include <chrono>
#include <filesystem>
#include <vector>
#include <fstream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

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
    for (uint64_t i = 1; i < blockCount; i++)
    {
        output[i] = input[i] + output[i - 1];
    }
}

__global__ void scan(uint32_t *g_odata, uint32_t *g_idata, uint32_t *blockSums, uint64_t n)
{
    extern __shared__ int temp[];
    int thid = threadIdx.x;
    int index = 2 * blockIdx.x * blockDim.x + threadIdx.x;
    int offset = 1;

    temp[2 * thid] = 0;
    temp[2 * thid + 1] = 0;

    if (index < n)
    {
        temp[2 * thid] = g_idata[index];
        temp[2 * thid + 1] = g_idata[index + 1];
    }

    for (int d = 2 * blockDim.x >> 1; d > 0; d >>= 1)
    {
        __syncthreads();

        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (thid == 0)
    {
        blockSums[blockIdx.x] = temp[2 * blockDim.x - 1];
        temp[2 * blockDim.x - 1] = 0;
    }

    for (int d = 1; d < 2 * blockDim.x; d *= 2)
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d)
        {
            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            int t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    temp[2 * thid] = temp[2 * thid + 1];

    if (thid == blockDim.x - 1)
    {
        temp[2 * thid + 1] = blockSums[blockIdx.x];
    }
    else
    {
        temp[2 * thid + 1] = temp[2 * thid + 2];
    }

    g_odata[index] = temp[2 * thid];
    g_odata[index + 1] = temp[2 * thid + 1];
}

__global__ void addOffsets(uint32_t *preScannedMask, uint32_t *blockScan)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (blockIdx.x == 0)
        return;

    preScannedMask[index] += blockScan[blockIdx.x - 1];
}

void compress(const string filename)
{
    ifstream inputFile;

    uint64_t filesize = filesystem::file_size(filesystem::path(filename));
    uint64_t blockCount = filesize / sizeof(uint8_t);
    uint32_t gridSize = (blockCount + 512 - 1) / 512;
    uint32_t *scannedMask;
    uint32_t *sequentialScannedMask = new uint32_t[blockCount];
    uint32_t *block_sums;
    uint32_t *scannedBlockSums;
    uint32_t *bs;
    uint32_t *mask;
    uint8_t *memblock;

    cudaMallocManaged(&memblock, blockCount * sizeof(uint8_t));
    cudaMallocManaged(&mask, blockCount * sizeof(uint32_t));
    cudaMallocManaged(&scannedMask, blockCount * sizeof(uint32_t));
    cudaMallocManaged(&block_sums, gridSize * sizeof(uint32_t));
    cudaMallocManaged(&scannedBlockSums, gridSize * sizeof(uint32_t));
    cudaMallocManaged(&bs, gridSize * sizeof(uint32_t));

    inputFile.open(filename, ios::binary);
    inputFile.read((char *)memblock, blockCount * sizeof(uint8_t));

    generateMask<<<gridSize, 512>>>(memblock, mask, blockCount);
    cudaDeviceSynchronize();

    sequentialScan(sequentialScannedMask, mask, blockCount);

    scan<<<gridSize, 512>>>(scannedMask, mask, block_sums, blockCount);
    cudaDeviceSynchronize();

    scan<<<1, ceil(gridSize)>>>(scannedBlockSums, block_sums, bs, gridSize);
    cudaDeviceSynchronize();

    addOffsets<<<gridSize, 512>>>(scannedMask, scannedBlockSums);

    for (uint64_t i = 0; i < blockCount; i++) {
        cout << "error at i = " << i << " | scannedMask[i] = " << scannedMask[i] << " | sequentialMask[i] = " << sequentialScannedMask[i] << endl;
    }

    cudaFree(memblock);
    cudaFree(mask);
    cudaFree(scannedMask);
    cudaFree(block_sums);
    cudaFree(scannedBlockSums);
    cudaFree(bs);
}

int main(int argc, char const *argv[])
{
    string filename = "simple_image.bmp";

    compress(filename);
}
