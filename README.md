# m3DFaceTrack
The MATLAB implementation of our GoMBF-Cascade algorithm for 3D facial tracking.

- The algorithm is based on this paper: http://arxiv.org/abs/2009.00935.
- Tracking results: https://drive.google.com/file/d/1__M7KEN0jsjbgZMyTdn49MpQR95jk7ZJ/view?usp=sharing. Please note that these results were produced by a C++ implementation of the same algorithm. 

# Setup
- Create a 'model' folder under the main directory and put the models mentioned below inside. 
- Put the following OpenCV DLLs and files into the main directory: opencv_core2413.dll, opencv_gpu2413.dll, opencv_highgui2413.dll, opencv_imgproc2413.dll, pencv_objdetect2413.dll, haarcascade_frontalface_alt2.xml.

# Data and Models
- Please send emails to jianwen.lou.ly@gmail.com for the training data, the parametric face model, a pre-trained 2D landmark detection model and a 3D facial tracking model. 

# Dependency
- L-BFGS-B: https://uk.mathworks.com/matlabcentral/fileexchange/35104-lbfgsb-l-bfgs-b-mex-wrapper?s_tid=prof_contriblnk
- OpenCV 2.4.13
