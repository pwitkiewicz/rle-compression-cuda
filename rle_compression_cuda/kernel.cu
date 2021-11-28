#include <iostream>
#include <chrono>
#include <filesystem>
#include <vector>
#include <fstream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

__global__ void backwardMask(uint8_t *input, uint8_t *mask, uint64_t maskSize)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (index == 0)
    {
        mask[0] = 1;
        for (int i = index + 1; i < maskSize; i += stride)
        {
            if (input[3 * i] == input[3 * i - 3] &&
                input[3 * i + 1] == input[3 * i - 2] &&
                input[3 * i + 2] == input[3 * i - 1])
            {
                mask[i] = 0;
            }
            else
            {
                mask[i] = 1;
            }
        }
        return;
    }

    for (int i = index; i < maskSize; i += stride)
    {
        if (input[3 * i] == input[3 * i - 3] &&
            input[3 * i + 1] == input[3 * i - 2] &&
            input[3 * i + 2] == input[3 * i - 1])
        {
            mask[i] = 0;
        }
        else
        {
            mask[i] = 1;
        }
    }
}

__global__ void inclusivePrefixSum(uint8_t *scannedMask, uint8_t *mask, uint64_t maskSize)
{
    extern __shared__ float temp[];
    uint32_t threadId = threadIdx.x;
    int offset = 1;

    temp[2 * threadId] = mask[2 * threadId];
    temp[2 * threadId + 1] = mask[2 * threadId + 1];

    for (uint32_t d = n >> 1; d > 0; d >>= 1) 
    {
        __syncthreads();
        if (threadId < d)
        {
            uint32_t ai = offset * (2 * threadId + 1) - 1;
            uint32_t bi = offset * (2 * threadId + 2) - 1;
            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (threadId == 0)
    {
        temp[n - 1] = 0;
    }     

    for (int d = 1; d < n; d *= 2) 
    {
        offset >>= 1;
        __syncthreads();
        if (threadId < d)
        {
            int ai = offset * (2 * threadId + 1) - 1;
            int bi = offset * (2 * threadId + 2) - 1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    scannedMask[2 * threadId] = temp[2 * threadId];
    scannedMask[2 * threadId + 1] = temp[2 * threadId + 1];
}

void compress(const string filename)
{
    ifstream inputFile;

    uint64_t filesize = filesystem::file_size(filesystem::path(filename));
    uint64_t blockCount = filesize / sizeof(uint8_t);
    uint32_t maskSize = blockCount / 3;
    uint32_t gridSize = (blockCount + 512 - 1) / 512;
    uint8_t *memblock;
    uint8_t *mask;
    uint8_t *scannedMask;

    cudaMallocManaged(&memblock, blockCount * sizeof(uint8_t));
    cudaMallocManaged(&mask, maskSize * sizeof(uint8_t));
    cudaMallocManaged(&scannedMask, maskSize * sizeof(uint8_t));

    inputFile.open(filename, ios::binary);
    inputFile.read((char *)memblock, blockCount * sizeof(uint8_t));

    backwardMask<<<gridSize, 512>>>(memblock, mask, maskSize);
    cudaDeviceSynchronize();

    inclusivePrefixSum<<<gridSize, 512>>>(scannedMask, mask, maskSize);
    cudaDeviceSynchronize();

    cudaFree(memblock);
    cudaFree(mask);
    //runLengthEncode(memblock, outputData, counter, blockCount);
    //writeCompressedFile(filename, outputData, counter);
}

int main(int argc, char const *argv[])
{
    string filename = "simple_image.bmp";

    auto t1 = chrono::high_resolution_clock::now();
    compress(filename);
    auto t2 = chrono::high_resolution_clock::now();
    auto ms_int = chrono::duration_cast<chrono::milliseconds>(t2 - t1);
    cout << filename << " GPU compression time: " << ms_int.count() << "ms\n";
}
