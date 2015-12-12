GPUOpticalFlow
===========

A CUDA-based .dll for calculating optical flow on video frames. 

Exported functions:

    void GPUOpticalFlow(*int[][] frame1, *int[][] frame2, *int[][][] opticalFlowOut, int WIDTH, int HEIGHT, int APERTURE, int MAXFLOW)

* **frame1**, **frame2**: sequential greyscale video image frames. Dimensions: int[WIDTH][HEIGHT]
* **opticalFlowOut**: return value. Dimensions: int[WIDTH][HEIGHT][2]
* **APERTURE**: Size of window used to compute match
* **MAXFLOW**: Maximum tested flow value