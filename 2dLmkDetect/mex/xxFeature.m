% Extract HOG features around the current landmarks
% X = xxFeature(img, xScale, yScale, lms)
% img: input image
% lms(2 x n): input landmark locations
% xScale, yScale: the scale of width and height used to warp the input image 
% X: computed descriptors in single, default size: 128 x n
%
% Dependence: OpenCV 2.4.13
% Authors: Jianwen Lou
% Creation Date: 22/11/2018
