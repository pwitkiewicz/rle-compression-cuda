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

void sequentialScan(uint32_t *output, uint32_t *input, uint64_t blockCount, bool inclusive)
{
    if (inclusive) {
        output[0] = input[0];
        for (int j = 1; j < blockCount; ++j)
        {
            output[j] = input[j] + output[j - 1];
        }
    }
    else {
        output[0] = 0;
        for (int j = 1; j < blockCount; ++j)
        {
            output[j] = input[j - 1] + output[j - 1];
        }
    }
}

__global__ void scan(uint32_t *output, uint32_t *input, uint32_t *sums, uint64_t n, bool inclusive)
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

    if (inclusive)
    {
        output[blockOffset + (2 * threadID) - 1] = temp[2 * threadID];
        output[blockOffset + (2 * threadID)] = temp[2 * threadID + 1];
    }
    else
    {
        output[blockOffset + (2 * threadID)] = temp[2 * threadID];
        output[blockOffset + (2 * threadID) + 1] = temp[2 * threadID + 1];
    }
} 

__global__ void inclusiveAdd(uint32_t* output, uint32_t length, uint32_t* n) {
    int blockID = blockIdx.x;
    int threadID = threadIdx.x;
    int blockOffset = blockID * length - 1;

    output[blockOffset + threadID] += n[blockID - 1];
}

__global__ void add(uint32_t* output, uint32_t length, uint32_t* n) {
    int blockID = blockIdx.x;
    int threadID = threadIdx.x;
    int blockOffset = blockID * length;

    output[blockOffset + threadID] += n[blockID];
}

__global__ void compact(uint32_t* scannedMask, uint32_t* compactedMask, uint32_t* totalSize, uint64_t n) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    if (index == 0) {
        compactedMask[0] = 0;
    }

    for (int i = index; i < n; i += stride) {

        if (i == (n - 1)) {
            compactedMask[scannedMask[i]] = i + 1;
            *totalSize = scannedMask[i];
        }

        if (scannedMask[i] != scannedMask[i - 1]) {
            compactedMask[scannedMask[i] - 1] = i;
        }
    }
}

__global__ void scatter(uint32_t* compactedMask, uint32_t* totalSize, uint8_t* input, uint8_t* outputData, uint32_t* occurences) {

    int n = *totalSize;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        int a = compactedMask[i];
        int b = compactedMask[i + 1];

        outputData[i] = input[a];
        occurences[i] = b - a;
    }
}

void prefixSum(const int gridSize, uint64_t blockCount, uint32_t* mask, uint32_t* scannedMask, bool inclusive)
{
    const int sharedMemArraySize = ELEMENTS_PER_BLOCK * sizeof(int);

    uint32_t* blockSums;
    uint32_t* scannedBlockSums;

    cudaMallocManaged(&blockSums, blockCount * sizeof(uint32_t));
    cudaMallocManaged(&scannedBlockSums, gridSize * sizeof(uint32_t));

    scan<<<gridSize, THREADS_PER_BLOCK, sharedMemArraySize >> > (scannedMask, mask, blockSums, ELEMENTS_PER_BLOCK, inclusive);
    cudaDeviceSynchronize();

    const int smallGridSize = (gridSize + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;

    // if scan size is smaller than viable for parallel scan do it sequentialy
    if (smallGridSize < 2) {
        sequentialScan(scannedBlockSums, blockSums, gridSize, inclusive);
    }
    else
    {
        prefixSum(smallGridSize, gridSize, blockSums, scannedBlockSums, inclusive);
    }

    if (inclusive)
    {
        inclusiveAdd<<<gridSize, ELEMENTS_PER_BLOCK>>>(scannedMask, ELEMENTS_PER_BLOCK, scannedBlockSums);
    }
    else
    {
        add<<<gridSize, ELEMENTS_PER_BLOCK>>>(scannedMask, ELEMENTS_PER_BLOCK, scannedBlockSums);
    }
    cudaDeviceSynchronize();

    cudaFree(blockSums);
    cudaFree(scannedBlockSums);
}

void compress(const string filename)
{
    ifstream inputFile;

    const uint64_t filesize = filesystem::file_size(filesystem::path(filename));
    const uint64_t blockCount = filesize / sizeof(uint8_t);
    const int gridSize = (blockCount + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    uint32_t *scannedMask;
    uint32_t *compactedMask;
    uint32_t *sequentialScannedMask = new uint32_t[blockCount];
    uint32_t *mask;
    uint32_t *totalSize;
    uint32_t *occurences;
    uint8_t *outputData;
    uint8_t *memblock;

    cudaMallocManaged(&memblock, blockCount * sizeof(uint8_t));
    cudaMallocManaged(&mask, blockCount * sizeof(uint32_t));
    cudaMallocManaged(&scannedMask, blockCount * sizeof(uint32_t));
    cudaMallocManaged(&totalSize, blockCount * sizeof(uint32_t));
    cudaMallocManaged(&compactedMask, blockCount * sizeof(uint32_t));

    inputFile.open(filename, ios::binary);
    inputFile.read((char *)memblock, blockCount * sizeof(uint8_t));

    auto t1 = chrono::high_resolution_clock::now();

    generateMask<<<gridSize, THREADS_PER_BLOCK>>>(memblock, mask, blockCount);
    cudaDeviceSynchronize();

    prefixSum(gridSize, blockCount, mask, scannedMask, true);

    compact<<<gridSize, THREADS_PER_BLOCK>>>(scannedMask, compactedMask, totalSize, blockCount);
    cudaDeviceSynchronize();

    cudaMallocManaged(&outputData, *totalSize * sizeof(uint8_t));
    cudaMallocManaged(&occurences, *totalSize * sizeof(uint32_t));

    scatter<<<gridSize, THREADS_PER_BLOCK>>>(compactedMask, totalSize, memblock, outputData, occurences);
    cudaDeviceSynchronize();

    auto t2 = chrono::high_resolution_clock::now();
    auto ms_int = chrono::duration_cast<chrono::milliseconds>(t2 - t1);
    cout << filename << " GPU compression time: " << ms_int.count() << "ms\n";
    
    ofstream outputFile;
    outputFile.open(filename + ".rlz", ios::binary);
    outputFile.write((char*) totalSize, sizeof(uint32_t));
    outputFile.write((char*) outputData, *totalSize * sizeof(uint8_t));
    outputFile.write((char*) occurences, *totalSize * sizeof(uint32_t));

    cudaFree(memblock);
    cudaFree(mask);
    cudaFree(scannedMask);
    cudaFree(compactedMask);
    cudaFree(totalSize);
    cudaFree(outputData);
    cudaFree(occurences);
}

__global__ void generateDecompressedData(uint8_t* input, uint8_t* output, uint32_t* occurences, uint32_t* positions, uint64_t blockCount)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < blockCount; i += stride)
    {
        int position = positions[i];
        int occurence = occurences[i];

        for (int j = position; j < position + occurence; j++)
        {
            output[j] = input[i];
        }
    }
}

void decompress(const string filename)
{
    ifstream inputFile;

    inputFile.open(filename, ios::binary);

    uint32_t totalSize;
    uint32_t* occurences;
    uint32_t* scannedOccurences;
    uint8_t* compressedData;
    uint8_t* decompressedData;
    uint64_t decompressedSize;


    inputFile.read((char*)&totalSize, sizeof(uint32_t));

    cudaMallocManaged(&occurences, totalSize * sizeof(uint32_t));
    cudaMallocManaged(&compressedData, totalSize * sizeof(uint8_t));
    cudaMallocManaged(&scannedOccurences, totalSize * sizeof(uint32_t));

    inputFile.read((char*)compressedData, totalSize * sizeof(uint8_t));
    inputFile.read((char*)occurences, totalSize * sizeof(uint32_t));

    const int gridSize = (totalSize + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;

    auto t1 = chrono::high_resolution_clock::now();

    prefixSum(gridSize, totalSize, occurences, scannedOccurences, false);
    cudaDeviceSynchronize();

    decompressedSize = scannedOccurences[totalSize - 1] + occurences[totalSize - 1];
    cudaMallocManaged(&decompressedData, decompressedSize * sizeof(uint8_t));

    const int bigGridSize = (totalSize + THREADS_PER_BLOCK + 1) / THREADS_PER_BLOCK;
    generateDecompressedData<<<bigGridSize, THREADS_PER_BLOCK>>>(compressedData, decompressedData, occurences, scannedOccurences, totalSize);
    cudaDeviceSynchronize();

    auto t2 = chrono::high_resolution_clock::now();
    auto ms_int = chrono::duration_cast<chrono::milliseconds>(t2 - t1);
    cout << filename << " GPU decompression time: " << ms_int.count() << "ms\n" << endl;

    ofstream outputFile;
    outputFile.open(filename + "_decompressed.bmp", ios::binary);
    outputFile.write((char*)decompressedData, decompressedSize * sizeof(uint8_t));
}


int main(int argc, char const *argv[])
{
    string filename = "image.bmp";
    compress(filename);
    decompress(filename + ".rlz");

    filename = "simple_image.bmp";
    compress(filename);
    decompress(filename + ".rlz");
}
