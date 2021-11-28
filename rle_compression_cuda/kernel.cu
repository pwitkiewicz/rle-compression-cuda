#include <iostream>
#include <chrono>
#include <filesystem>
#include <vector>
#include <fstream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

__global__ void backwardMask(uint8_t *input, uint8_t *mask, uint64_t blockCount)
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

    for (int i = index; i < blockCount; i += stride)
    {
        mask[i] = input[i] == input[i - 1] ? 0 : 1;
    }
}

void compress(const string filename)
{
    ifstream inputFile;

    uint64_t filesize = filesystem::file_size(filesystem::path(filename));
    uint64_t blockCount = filesize / sizeof(uint8_t);
    uint32_t gridSize = (blockCount + 512 - 1) / 512;
    uint8_t *memblock;
    uint8_t* mask;
    

    cudaMallocManaged(&memblock, blockCount * sizeof(uint8_t));
    cudaMallocManaged(&mask, blockCount * sizeof(uint8_t));

    inputFile.open(filename, ios::binary);
    inputFile.read((char *)memblock, blockCount * sizeof(uint8_t));

    backwardMask<<<gridSize, 512>>>(memblock, mask, blockCount);
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
