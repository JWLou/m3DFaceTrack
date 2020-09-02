function param_im = mm_fit(param_im, param_fm)
%MM_FIT fits the parametric face model to image based on landmark
%   constraints.

%% Initialization
wid = param_im.wid; 
wexp = param_im.wexp;
lmk_im = param_im.lmk_im;
Q = param_im.Q;
indlmk_mesh = param_im.indlmk_mesh;
nlmk = length(indlmk_mesh);

% landmarks for fitting
idlmkfit_id = 1:nlmk; 
idlmkfit_exp = 1:nlmk; 
idlmkfit_pose = 1:nlmk; 

% landmark weights
wlmk_exp = ones(1, nlmk); 
wlmk_id = ones(1, nlmk); 

% coarse pose estimation
vert = param_fm.mu_id + param_fm.pc_id*(wid.*param_fm.ev_id) + param_fm.delta_bldshp*wexp;
nvert = length(vert)/3;
vert = reshape(vert, 3, nvert);
lmk_obj = vert(:,indlmk_mesh); 
[R, T] = solve_pose(lmk_im(idlmkfit_pose,:), (lmk_obj(:,idlmkfit_pose))', Q);

% solve the parameters using the coordinate-descent method
niter = 1;
err = zeros(niter, 1);
for i = 1:niter
    % update contour landmarks on the mesh
    indlmk_mesh = update_lmkind(indlmk_mesh, R, vert, param_fm.lmk_horzlines);
    indlmkfit_tmp = indlmk_mesh(idlmkfit_exp); % exp
    indlmkfit_exp = [3*indlmkfit_tmp'-2; 3*indlmkfit_tmp'-1; 3*indlmkfit_tmp'];
    indlmkfit_exp = indlmkfit_exp(:); 
    indlmkfit_tmp = indlmk_mesh(idlmkfit_id); % id
    indlmkfit_id = [3*indlmkfit_tmp'-2; 3*indlmkfit_tmp'-1; 3*indlmkfit_tmp'];
    indlmkfit_id = indlmkfit_id(:); 
    
    % solve for pose parameters 
    lmk_obj = vert(:,indlmk_mesh);
    [R, T] = solve_pose(lmk_im(idlmkfit_pose,:), (lmk_obj(:,idlmkfit_pose))', Q);
    
    % solve for expression weights    
    vert_id = param_fm.mu_id + param_fm.pc_id*(wid.*param_fm.ev_id);
    wexp = solve_wexp(wexp, param_fm.delta_bldshp(indlmkfit_exp,:), vert_id(indlmkfit_exp,:), lmk_im(idlmkfit_exp,:), Q, R, T, wlmk_exp(idlmkfit_exp));
    
    % solve for identity weights
    vert_exp = param_fm.mu_id + param_fm.delta_bldshp*wexp;
    wid = solve_wid(wid, param_fm.ev_id, param_fm.pc_id(indlmkfit_id,:), vert_exp(indlmkfit_id,:), lmk_im(idlmkfit_id,:), Q, R, T, wlmk_id(idlmkfit_id));
    
    % 2d landmark displacements
    vert = param_fm.mu_id + param_fm.pc_id*(wid.*param_fm.ev_id) + param_fm.delta_bldshp*wexp;
    vert = reshape(vert, 3, nvert);
    lmk_obj = vert(:, indlmk_mesh);
    lmk_camq = Q*(R*lmk_obj + repmat(T, 1, nlmk));
    lmk_proj = [lmk_camq(1,:)./lmk_camq(3,:); lmk_camq(2,:)./lmk_camq(3,:)];   
    dev = lmk_im' - lmk_proj;
    
    % error
    K = size(dev, 2);
    for k = 1:K
        err(i) = err(i) + norm(dev(:, k));
    end
    err(i) = err(i)/K;
end

% update parameters
param_im.wid = wid;
param_im.wexp = wexp;
param_im.R = R;  
param_im.T = T;
param_im.D = dev'; 
param_im.indlmk_mesh = indlmk_mesh;
param_im.err = err;

end


