GPUOpticalFlow
===========

A CUDA-based .exe for calculating optical flow on video frames. 

```
>> GPUOpticalFlow.exe MAXFLOW APERTURE Frame1Path Frame2Path
```

* ``MAXFLOW``: Maximum tested flow value
* ``APERTURE``: Size of window used to compute match
* ``Frame1Path``, ``Frame2Path``: sequential 640 x 480 .png images

Usage example:

* Copy "frame1.png" and "frame2.png" to \Debug folder
* Build solution in Visual Studio
* Run the following command:


```
>> GPUOpticalFlow.exe 19 6 "frame1.png" "frame2.png"
```
	
Output: ``opticalflow.png``
