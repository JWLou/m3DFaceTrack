function reg3d_train(trainset, wexpset, impath, param_fm)
%REG3D_TRAIN trains a 3D shape parameter regressor
tic;

%% Pre-processing
% training parameters
param_train = get_param_reg;
param_train.nlmk = size(trainset{1}.lmk_im, 1);
param_train.ndim_wexp = param_fm.nexp;
param_train.indset = {1:param_train.ndim_wexp; param_train.ndim_wexp+1:param_train.ndim_wexp+3; ...
    param_train.ndim_wexp+4:param_train.ndim_wexp+6; param_train.ndim_wexp+7: ...
    param_train.ndim_wexp+6+param_train.nlmk*2};
param_train.ndim_P = param_train.ndim_wexp + 6 + param_train.nlmk*2;
param_train.nimg = length(trainset);
param_train.lmkT = param_fm.faceimT.lmkT;
param_train.DT = param_fm.DT;
param_train.sDT = param_fm.sDT;
param_train.DTc = param_fm.DTc;
% get the pupil distance of the mean shape
eyedist = norm((mean(param_fm.faceimT.lmkT(37:42, :)) - mean(param_fm.faceimT.lmkT(43:48, :))));
param_train.k = param_train.k * eyedist;

% sort out the training data  
Pmat = zeros(param_train.ndim_P, param_train.nimg);
Pmat_gt = zeros(param_train.ndim_P, param_train.nimg);
widmat = zeros(length(trainset{1}.wid)+1, param_train.nimg);
% create sparse matrcies to store all R and Q
I = repmat(1:param_train.nimg*3, 3, 1); I = I(:);
J1 = repmat(1:3:param_train.nimg*3-2, 3, 1); J1 = J1(:);
J2 = repmat(2:3:param_train.nimg*3-1, 3, 1); J2 = J2(:);
J3 = repmat(3:3:param_train.nimg*3, 3, 1); J3 = J3(:);
J = [J1';J2';J3']; J = J(:);
VR = ones(param_train.nimg*9, 1);
VQ = ones(param_train.nimg*9, 1);
for i = 1:param_train.nimg
    if i == 1
        nimid = 1;
        trainset{i}.imid = nimid;
        imnameset{nimid} = trainset{i}.imname;
        imgset{nimid} = rgb2gray(imread([impath, trainset{i}.imname]));
        lmkset{nimid} = trainset{i}.lmk_im;
    else
        tf = strcmp(trainset{i}.imname, imnameset);
        loc = find(tf);
        if isempty(loc)
            nimid = nimid + 1;
            trainset{i}.imid = nimid;
            imnameset{nimid} = trainset{i}.imname;
            img = imread([impath, trainset{i}.imname]);
            if size(img,3) == 3
                imgset{nimid} = rgb2gray(img);
            else
                imgset{nimid} = img;
            end
            lmkset{nimid} = trainset{i}.lmk_im;
        else
            trainset{i}.imid = loc;
        end
    end   
    Pmat(:,i) = trainset{i}.Pin;     
    Pmat_gt(:,i) = trainset{i}.Pout;
    widmat(:,i) = [1;trainset{i}.wid.*param_fm.ev_id];
    eulerR = Pmat(param_train.indset{2},i);
    R = angle2R(eulerR(1), eulerR(2), eulerR(3)); R = R';
    Q = trainset{i}.Q; Q = Q';
    irng = ((i-1)*9+1):(i*9);
    VR(irng) = R(:);
    VQ(irng) = Q(:);   
    trainset{i} = rmfield(trainset{i}, {'Pin','Pout','wid','Q','lmk_im'}); 
end
lmkmat_camq = obj2im_batch(trainset, widmat, Pmat, I, J, VR, VQ, param_train, param_fm);
for i = 1:param_train.nimg
    lmk_camq = lmkmat_camq((i-1)*3+1:i*3,:);
    lmk_proj = [lmk_camq(1,:)./lmk_camq(3,:); lmk_camq(2,:)./lmk_camq(3,:)];
    D = Pmat(param_train.indset{4},i);
    D = reshape(D, 2, []);
    lmk_proj = lmk_proj + D;
    lmk_proj = lmk_proj';
    trainset{i}.lmk_proj = lmk_proj;
end
toc;

% save
% mpath = [impath, 'model\'];
% save([mpath, 'param_train.mat'], 'param_train');
% save([mpath, 'imgset.mat'], 'imgset', '-v7.3');
% save([mpath, 'lmkset.mat'], 'lmkset');
% save([mpath, 'Pmat_gt.mat'], 'Pmat_gt');
% save([mpath, 'widmat.mat'], 'widmat');
% save([mpath, 'VQ.mat'], 'VQ');

%% 3D Shape Regression
err = zeros(param_train.nimg, 1);
mreg = cell(param_train.niter, 1);
for t = 1:param_train.niter    
    % learn the stage regressor
    fprintf('Start the %dth stage training...\n', t);
    param_train.t = t;
    deltaPmat = (Pmat_gt - Pmat)';
    [deltaP_pred, regP] = stage_reg_train(deltaPmat, trainset, imgset, param_train);

    % update 3d shape parameters and the 2d projected shape
    Pmat = Pmat + deltaP_pred';
    for i = 1:param_train.nimg
        eulerR = Pmat(param_train.indset{2},i);
        R = angle2R(eulerR(1), eulerR(2), eulerR(3)); R = R';
        irng = ((i-1)*9+1):(i*9);
        VR(irng) = R(:);
    end
    lmkmat_camq = obj2im_batch(trainset, widmat, Pmat, I, J, VR, VQ, param_train, param_fm);
    for i = 1:param_train.nimg
        lmk_camq = lmkmat_camq((i-1)*3+1:i*3,:);
        lmk_proj = [lmk_camq(1,:)./lmk_camq(3,:); lmk_camq(2,:)./lmk_camq(3,:)];
        D = Pmat(param_train.indset{4},i);
        D = reshape(D, 2, []);
        lmk_proj = lmk_proj + D;
        lmk_proj = lmk_proj';
        trainset{i}.lmk_proj = lmk_proj;
 
        devlmk = lmkset{trainset{i}.imid} - lmk_proj;
        err(i) = mean(sqrt(sum(abs(devlmk).^2, 2))); 
    end
    mreg{t} = regP;
    % save
%     save([mpath, 'trainset.mat'], 'trainset');
%     save([mpath, 'Pmat.mat'], 'Pmat');
%     save([mpath, 'mreg.mat'], 'mreg');

    % error
    fprintf('Stage-%d, error: %.2f\n', t, mean(err));
    toc;
end
save('.\model\m3dreg.mat', 'mreg', 'wexpset', '-v7.3');

end

%%
function [deltaP_pred, regP] = stage_reg_train(deltaPmat, trainset, imgset, param_train)
%STAGE_REG_TRAIN trains a stage regressor of the cascaded regressors 

%% Pre-processing
npt = param_train.npx;
pts = zeros(npt, 2);
for i = 1:npt
    indlmk = randi(param_train.nlmk, 1, 1);
    devlmk = (2*rand(1,2)-1)*param_train.k(param_train.t); % sample in the 2d mean shape coordinate [-k, k]
    pts(i,:) = param_train.lmkT(indlmk,:) + devlmk;
end
indDT = zeros(param_train.npx, 1);
bcDT = zeros(param_train.npx, 3);
nDT = length(param_train.DT);
for i = 1:param_train.npx
    if i <= npt
        pt = pts(i,:); 
        dist = sqrt(sum((param_train.DTc-repmat(pt, nDT, 1)).^2, 2));
        [~, iDT] = min(dist);
        [~, bc] = ptintri(pt, param_train.lmkT(param_train.DT(iDT,:),:));
        bc = bc';
        indDT(i) = iDT;
        bcDT(i,:) = bc;
    else
        indDT(i) = randi(nDT);
        r1 = rand;
        r2 = sqrt(rand);
        bcDT(i,:) = [r2*r1, r2*(1-r1), 1-r2];
    end
end

I = repmat(1:param_train.npx, 3, 1);
I = I(:);
J = 1:param_train.npx*3;
J = J';
V = bcDT';
V = V(:);
sparseBC = sparse(I, J, V, param_train.npx, param_train.npx*3);

% regression target
regtar = cell(1, 4);
regtar{1} = deltaPmat(:,param_train.indset{1});
regtar{2} = deltaPmat(:,param_train.indset{2});
regtar{3} = deltaPmat(:,param_train.indset{3});
regtar{4} = deltaPmat(:,param_train.indset{4});
px = zeros(param_train.nimg, param_train.npx);
for i = 1:param_train.nimg    
    % pixel values
    im = imgset{trainset{i}.imid};
    imh = size(im, 1);
    imw = size(im, 2);
    
    indlmk = param_train.DT(indDT,:);
    indlmk = indlmk';
    indlmk = indlmk(:);
    xy = sparseBC * trainset{i}.lmk_proj(indlmk,:);
    x = round(xy(:,1));
    y = round(xy(:,2));
    x(x>(imw-1)) = imw-1; x(x<1) = 1;
    y(y>(imh-1)) = imh-1; y(y<1) = 1;
    
    for j = 1:param_train.npx        
        px(i,j)= double(im(y(j),x(j)));
    end
end
covpx = cov(px); % compute pixel-pixel covariance

%% Regression    
nimg = param_train.nimg;
F = param_train.F;
beta = param_train.beta;
nfern_int = param_train.nfern_int;
mLocal = cell(4, 1);
feafern = cell(1, 4);
parfor j = 1:4
    % local regression
    [mLocal{j}, feafern{j}] = mexfern(px, regtar{j}, covpx, int32(nfern_int{j}), int32(F), beta);
end
feafern = cell2mat(feafern);
A = parsefea(feafern, nimg, nfern_int, F);
% global refinement/fusion 
mGlobal = (A'*A + 100*eye(size(A, 2))) \ (A'*[regtar{1},regtar{2},regtar{3},regtar{4}]);
deltaP_pred = A*mGlobal;

regP.indDT = indDT;
regP.bcDT = bcDT; 
regP.mLocal = mLocal;
regP.mGlobal = mGlobal;

end

%%
function feabin = parsefea(feafern, nsample, nfernset, F)

nbin = 2^F;
nfernset = cell2mat(nfernset);
nfern_all = sum(nfernset);
I = repmat(1:nsample, nfern_all, 1); 
J = repmat(0:nbin:(nfern_all-1)*nbin, nsample, 1) + double(feafern);

I = [I; 1:nsample];
I = I(:);
J = J + 1;
J = [ones(nsample,1), J];
J = J'; 
J = J(:);
V = ones(nsample*nfern_all+nsample, 1);
feabin = sparse(I, J, V, nsample, nbin*nfern_all+1);

end

