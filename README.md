GPUOpticalFlow
===========

A CUDA-based .dll for calculating optical flow on video frames. 

Exported functions:

    void GPUOpticalFlow(*int[][] frame1, *int[][] frame2, *int[][][] opticalFlowOut, int width, int height, int aperture, int maxFlow)

* **frame1**, **frame2**: sequential greyscale video image frames. Dimensions: int[WIDTH][HEIGHT]
* **opticalFlowOut**: return value. Dimensions: int[WIDTH][HEIGHT][2]
* **aperture**: Size of window used to compute match
* **maxFlow**: Maximum tested flow value