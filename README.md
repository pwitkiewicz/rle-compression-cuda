# RLE compression

This is mine version of compression algorithm based on run length encoding implemented for nVidia cards having CUDA compute capability.

I based my work on:
- [rle encoding in cuda](https://erkaman.github.io/posts/cuda_rle.html)
- [parallel prefix sum](https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/scan/doc/scan.pdf)

While compression algorithm might be farily complex it goes as follows (as a __run__ I define a sequence of repeated character):
1. Compute mask out of input data - put __1__ where new run starts, __0__ otherwise
2. Compute __inclusive prefix sum__ (scan) for the mask, which encodes the output location of all the compressed pairs.
3. Compute __compacted mask__ which encodes starting index for all the runs.
4. Finally compute __occurences__ (which are difference between elements in compacted mask) and save
   character to __output matrix__ of each run (which can be found by using compacted mask value as index).
   
Decompression algorithm is a more straightforward:
1. Compute __exclusive prefix sum__ out of occurences matrix, it's elemnts will point to the place of start of each run.
2. Compute orginal data, in my approach each thread is responsible for filling out output matrix for one run. 
   (It might be inefficient approach for a highly RLE compressible data)
