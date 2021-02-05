function main

impath = 'D:\data\'; % replace with your own path
param_fm = init_pfm;

%% generate training data
load('.\data\imgset-R.mat');
[trainset, wexpset] = gen_train_data(imgset, param_fm, impath);
save('.\data\traindata.mat', 'trainset', 'wexpset', '-v7.3');

%% train
load('.\data\traindata.mat');
ndata = length(trainset);
idxtrain = randperm(ndata, ndata);
trainset = trainset(idxtrain);
reg3d_train(trainset, wexpset, impath, param_fm);

%% test
vidpath = '.\data\vid.mp4'; % test with your own video
load('.\model\m3dreg.mat');
mdetect2d = load('.\model\mdetect2d.mat');
reg3d_test(mreg, wexpset, mdetect2d, param_fm, vidpath);

end
