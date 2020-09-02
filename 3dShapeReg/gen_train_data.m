function [trainset, wexpset] = gen_train_data(imgset, param_fm, impath)
%GEN_TRAIN_DATA generates training data for 3D shape regression

tic;

npairset = [8, 8, 30, 0, 0];
npair = sum(npairset);
nimg = length(imgset);
trainset = cell(nimg*npair, 1);
nlmk = length(param_fm.indlmk_mesh);

% idsubj_im and wexpset
nsubj = 1;
idsubj_im = zeros(nimg, 1);
widset = zeros(1, param_fm.nid);

nexp = 1;
idexp_im = zeros(nimg, 1);
wexpset = zeros(1, param_fm.nexp);
for i = 1:nimg
    wid = imgset(i).param_im.wid';
    [Lia, Locb] = ismember(wid, widset, 'rows');
    if ~Lia
        widset = [widset; wid];
        nsubj = nsubj + 1;
        idsubj_im(i) = nsubj;
    else
        idsubj_im(i) = Locb;
    end
    
    wexp = imgset(i).param_im.wexp';
    [Lia, Locb] = ismember(wexp, wexpset, 'rows');
    if ~Lia
        wexpset = [wexpset; wexp];
        nexp = nexp + 1;
        idexp_im(i) = nexp;
    else
        idexp_im(i) = Locb;
    end
end
widset = widset';
wexpset = wexpset';

n = 1;
for i = 1:nimg
    fprintf('Img-%d\n', i);
    
    % gt shape parameters
    u0 = imgset(i).param_im.u0;
    v0 = imgset(i).param_im.v0;
    fg = imgset(i).param_im.f;
    Qg = [fg,0,u0; 0,fg,v0; 0,0,1];
    widg = imgset(i).param_im.wid;
    wexpg = imgset(i).param_im.wexp;
    eulerRg = imgset(i).param_im.eulerR';
    Rg = imgset(i).param_im.R;
    Tg = imgset(i).param_im.T;
    Dg = imgset(i).param_im.D;
    lmk_im = imgset(i).lmk;
    indlmk_mesh = imgset(i).param_im.indlmk_mesh;
    
    % indexp and indid
    indexp = setdiff(1:nexp, idexp_im(i));
    indexp = randperm(length(indexp));kexp = 1;
    indid = setdiff(1:nsubj, idsubj_im(i));
    indid = randperm(length(indid)); kid = 1;

    % training pairs
    for k = 1:npair
        trainset{n}.imname = imgset(i).name;
        trainset{n}.lmk_im = lmk_im; 
        trainset{n}.indlmk_mesh = indlmk_mesh;
        trainset{n}.Q = Qg; 
        trainset{n}.wid = widg;
       
        Dr = zeros(nlmk, 2);    
        if k <= sum(npairset(1))
            % random translation
            %fprintf('%d-random translation\n', k);
            
            devT1 = 5e4; 
            devT2 = 2e5;
            Tr(1,1) = Tg(1) + devT1*normrnd(0,0.25); 
            Tr(2,1) = Tg(2) + devT1*normrnd(0,0.25); 
            Tr(3,1) = Tg(3) + devT2*normrnd(0,0.25); 
            
            Pin = [wexpg; eulerRg; Tr; reshape(Dr', nlmk*2, 1)];
            Pout = [wexpg; eulerRg; Tg; reshape(Dg', nlmk*2, 1)];
        elseif k <= sum(npairset(1:2))
            % random rotation
            %fprintf('%d-random  rotation\n', k);
            
            devr = pi/5;
            eulerRr(1,1) = eulerRg(1) + devr*normrnd(0,0.25); 
            eulerRr(2,1) = eulerRg(2) + devr*normrnd(0,0.25); 
            eulerRr(3,1) = eulerRg(3) + devr*normrnd(0,0.25); 
            
            Pin = [wexpg; eulerRr; Tg; reshape(Dr', nlmk*2, 1)];
            Pout = [wexpg; eulerRg; Tg; reshape(Dg', nlmk*2, 1)];
        elseif k <= sum(npairset(1:3))
            % random expression
            %fprintf('%d-random expression\n', k);
            
            ind = indexp(kexp);
            wexpr = wexpset(:,ind);
            kexp = kexp + 1;
            
            Pin = [wexpr; eulerRg; Tg; reshape(Dr', nlmk*2, 1)];
            Pout = [wexpg; eulerRg; Tg; reshape(Dg', nlmk*2, 1)];
        elseif k <= sum(npairset(1:4))       
            % random camera
            %fprintf('%d-random camera\n', k);
            
            devf = fg/15 * normrnd(0,1,1);
            fr = fg + devf;
            Qr = [fr,0,u0; 0,fr,v0; 0,0,1];
            vert_proj = obj2im(widg, wexpg, Qr, Rg, Tg, param_fm);
            lmk_proj = vert_proj(indlmk_mesh, :);
            Dg_cam = lmk_im - lmk_proj;
            
            trainset{n}.Q = Qr; 
            Pin = [wexpg; eulerRg; Tg; reshape(Dr', nlmk*2, 1)];
            Pout = [wexpg; eulerRg; Tg; reshape(Dg_cam', nlmk*2, 1)];
        else
            % random identity 
            %fprintf('%d-random identity\n', k);
            
            ind = indid(kid);
            widr = widset(:,ind);
            kid = kid + 1;
            
            vert_proj = obj2im(widr, wexpg, Qg, Rg, Tg, param_fm);
            lmk_proj = vert_proj(indlmk_mesh, :);
            Dg_id = lmk_im - lmk_proj;
            
            trainset{n}.wid = widr; 
            Pin = [wexpg; eulerRg; Tg; reshape(Dr', nlmk*2, 1)];
            Pout = [wexpg; eulerRg; Tg; reshape(Dg_id', nlmk*2, 1)];
        end
       
        trainset{n}.Pin = Pin; 
        trainset{n}.Pout = Pout;
%         show(trainset{n}, impath, param_fm); pause;
        n = n+1;
    end
end
toc;

end

function show(data, impath, param_fm)
%SHOW shows the result of the current shape parameters

ndim_wexp = 46;
nlmk = 66;
indset = {1:ndim_wexp; ndim_wexp+1:ndim_wexp+3; ndim_wexp+4:ndim_wexp+6; ndim_wexp+7:ndim_wexp+6+nlmk*2};

Q = data.Q;
wid = data.wid;

% out
wexpg = data.Pout(indset{1});
eulerRg = data.Pout(indset{2});
Rg = angle2R(eulerRg(1), eulerRg(2), eulerRg(3));
Tg = data.Pout(indset{3});
Dg = data.Pout(indset{4});
Dg = reshape(Dg, 2, []); Dg = Dg';
vert_proj = obj2im(wid, wexpg, Q, Rg, Tg, param_fm);
lmk_proj_g = vert_proj(data.indlmk_mesh, :);
lmk_proj_g = lmk_proj_g + Dg;

% in
wexpr = data.Pin(indset{1});
eulerRr = data.Pin(indset{2});
Rr = angle2R(eulerRr(1), eulerRr(2), eulerRr(3));
Tr = data.Pin(indset{3});
Dr = data.Pin(indset{4});
Dr = reshape(Dr, 2, []); Dr = Dr';
vert_proj = obj2im(wid, wexpr, Q, Rr, Tr, param_fm);
lmk_proj_r = vert_proj(data.indlmk_mesh, :);
lmk_proj_r = lmk_proj_r + Dr;

img = imread([impath, data.imname]); 
subplot(1,3,1);
imshow(img);
title('Input face image');
hold on;
plot(lmk_proj_g(:, 1), lmk_proj_g(:, 2), 'r.', 'markersize', 10);
plot(lmk_proj_r(:, 1), lmk_proj_r(:, 2), 'b.', 'markersize', 10);
hold off;

subplot(1,3,2);
vert_cam_g = obj2cam(wid, wexpg, Rg, Tg, param_fm);
vert_cam_g = vert_cam_g';
display_face(vert_cam_g, param_fm.tri)
title('Target 3D mesh');

subplot(1,3,3);
vert_cam_r = obj2cam(wid, wexpr, Rr, Tr, param_fm);
vert_cam_r = vert_cam_r';
display_face(vert_cam_r, param_fm.tri)
title('Initial 3D mesh');

end
