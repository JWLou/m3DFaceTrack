function reg3d_test(mreg3d, wexpset, mdetect2d, param_fm, vidpath)
%REG3D_TEST tests the trained regressor in video facial tracking. 

%% Pre-processing
param_test = get_param_reg;
param_test.ninit = 30;
param_test.nlmk = length(param_fm.indlmk_mesh);
param_test.ndim_wexp = param_fm.nexp;
param_test.indset = {1:param_test.ndim_wexp; param_test.ndim_wexp+1:param_test.ndim_wexp+3; ...
          param_test.ndim_wexp+4:param_test.ndim_wexp+6; param_test.ndim_wexp+7: ...
          param_test.ndim_wexp+6+param_test.nlmk*2};
param_test.ndim_P = param_test.ndim_wexp + 6 + param_test.nlmk*2;
param_test.DT = param_fm.DT;

param_test.nexp = size(wexpset,2);
vert_objset = [];
for i = 1:param_test.nexp
    wexp = wexpset(:,i);
    vert_obj = param_fm.mu_id + param_fm.delta_bldshp*wexp;    
    vert_objset = [vert_objset, vert_obj];
end
param_test.vert_objset = vert_objset;

% write video
vid = VideoWriter('test.avi'); 
vid.FrameRate = 25;
open(vid);

%% Track Face in the Video 
dotrack3d = false;
facedetect = true;
vidobj = VideoReader(vidpath); 
while hasFrame(vidobj)
    img = readFrame(vidobj);
    img = rgb2gray(img);
    if facedetect
        faces = double(faceDetection(img));
        if isempty(faces)
            continue;
        else
            facebbox = largestFace(faces);
        end
        facedetect = false;
    end

    if ~dotrack3d
        % delect 2d landmarks
        lmk_im = lmkDetect(img, facebbox, mdetect2d.faceT, mdetect2d.dmDetect);
        
        param_im = init_pim(param_fm, img, lmk_im, 1000);
%         param_im = solvef_im(param_fm, param_im);
        param_im = mm_fit(param_im, param_fm);
        [~, frame] = show_res(param_fm, param_im);

        dotrack3d = true;
    else
        param_im.img = img; 
        param_im.wexp(param_im.wexp<0) = 0;
        param_im.wexp(param_im.wexp>1) = 1;
        param_im.D = 0*param_im.D;

        % 3d shape parameter regression
        P_pred = reg3d_search(param_im, param_test, param_fm, mreg3d, wexpset);
        [wexp, R, T, D] = decomp_param(P_pred, param_test.indset);
        param_im.wexp = wexp';
        param_im.R = R;
        param_im.T = T';
        param_im.D = D;
        vert = param_fm.mu_id + param_fm.pc_id*(param_im.wid.*param_fm.ev_id) + param_fm.delta_bldshp*param_im.wexp;
        vert = reshape(vert, 3, []);
        param_im.indlmk_mesh = update_lmkind(param_fm.indlmk_mesh, param_im.R, vert, param_fm.lmk_horzlines);

        [lmk_im, frame] = show_res(param_fm, param_im);
        param_im.lmk_im = lmk_im';
    end 
    writeVideo(vid, frame);
end  
close(vid);

end

%%
function P_pred = reg3d_search(param_im, param_test, param_fm, mreg3d, wexpset)
 
indlmk = [3*param_im.indlmk_mesh'-2; 3*param_im.indlmk_mesh'-1; 3*param_im.indlmk_mesh']; indlmk = indlmk(:);  
vert_obj = param_fm.mu_id + param_fm.delta_bldshp*param_im.wexp;
devlmk = repmat(vert_obj(indlmk), 1, param_test.nexp) - param_test.vert_objset(indlmk,:);
dist = sum(abs(devlmk).^2); 
[~, indset_wexp] = sort(dist);

P_pred = zeros(param_test.ninit, param_test.ndim_P);
dataset = cell(param_test.ninit, 1);
for n = 1:param_test.ninit
    P = zeros(param_test.ndim_P, 1);

    wexp = wexpset(:, indset_wexp(n));
    R = param_im.R;
    T = param_im.T;
    D = param_im.D;
    P(param_test.indset{1}) = wexp;
    [eulerRx, eulerRy, eulerRz] = R2angle(R);
    P(param_test.indset{2}) = [eulerRx; eulerRy; eulerRz]; 
    P(param_test.indset{3}) = T;
    P(param_test.indset{4}) = reshape(D', param_test.nlmk*2, 1);

    vert_proj = obj2im(param_im.wid, wexp, param_im.Q, R, T, param_fm);
    lmk_proj = vert_proj(param_im.indlmk_mesh, :);
    lmk_proj = lmk_proj + D;
    
    dataset{n}.lmk_proj = lmk_proj;
    dataset{n}.P = P;
    for t = 1:param_test.niter
        deltaP_pred = stage_reg_test(param_im.img, dataset{n}, mreg3d{t}, param_test);
        P = dataset{n}.P + deltaP_pred';        
        
        [wexp, R, T, D] = decomp_param(P, param_test.indset);
        vert_proj = obj2im(param_im.wid, wexp, param_im.Q, R, T, param_fm);
        lmk_proj = vert_proj(param_im.indlmk_mesh, :);
        lmk_proj = lmk_proj + D;

        dataset{n}.lmk_proj = lmk_proj;
        dataset{n}.P = P;
    end
    P_pred(n, :) = dataset{n}.P;
end
if param_test.ninit > 1
    P_pred = mean(P_pred);
end

end

function deltaP_pred = stage_reg_test(img, data, regP, param_test)
%STAGE_REG_TEST

%% Feature extraction
npx = param_test.npx;
px = zeros(1, npx);
indDT = regP.indDT;
bcDT = regP.bcDT;
imh = size(img, 1);
imw = size(img, 2);
for i = 1:npx
    xy = bcDT(i,:) * data.lmk_proj(param_test.DT(indDT(i),:),:);
    x = round(xy(1));
    y = round(xy(2));
    x(x>(imw-1)) = imw-1; x(x<1) = 1;
    y(y>(imh-1)) = imh-1; y(y<1) = 1;

    px(i)= double(img(y,x));
end

%% Prediction    
feabin = cell(1, 4);
for i = 1:4
    ferns = regP.mLocal{i};
    for k = 1:param_test.nfern_int{i}
        fern = ferns(k);
        [predBin, ~] = fern_reg(px, fern, param_test);
        feabin{i} = [feabin{i}, predBin]; 
    end
end
% global refinement/fusion
deltaP_pred = [1, cell2mat(feabin)] * regP.mGlobal;    

end

%%
function [predBin, predFern] = fern_reg(pxVal, fern, params_reg)
    predBin = zeros(1, 2^params_reg.F);
    idx = 0;
    for i = 1:params_reg.F
        m_f = fern.ind_px(i, 1);
        n_f = fern.ind_px(i, 2);
        
        pxVal_1 = pxVal(m_f);
        pxVal_2 = pxVal(n_f);
        
        if (pxVal_1 - pxVal_2) >= fern.thd_fea(i)
            idx = idx + 2^(i-1);
        end
    end
    idx = idx + 1;
    predFern = fern.pred_bin(idx, :);
    predBin(idx) = 1;
end


