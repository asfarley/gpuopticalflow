GPUOpticalFlow
===========

A CUDA-based .exe for calculating optical flow on video frames. 

    >> GPUOpticalFlow.exe MAXFLOW APERTURE Frame1Path Frame2Path

* ``MAXFLOW``: Maximum tested flow value
* ``APERTURE``: Size of window used to compute match
* ``Frame1Path``, ``Frame2Path``: sequential 640 x 480 .png images

Output: opticalflow.png
