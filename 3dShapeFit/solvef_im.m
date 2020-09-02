function param_im = solvef_im(param_fm, param_im)
%SOLVEF_IM finds the appropriate focal length 
% plot the convex curve between f and the fitting error
% frng = 400:50:2000;
% N = length(frng);
% err = zeros(N, 1);
% for n = 1:N
%     err(n) = ferr_im(param_fm, param_im, frng(n));
% end
% plot(frng, err, '-o', 'MarkerFaceColor', [0 0 1]);
% xlabel('Focal Length - f'); ylabel('Fitting Error (px)'); grid on;
% pause;
    
% search for the optimal f
frng = [400, 1200]; % focal length range
fl = frng(1); fr = frng(2); fm = (fl+fr)/2;
derr_df_tar = 0;
df = 50;
while (fr-fl)>30     
    err_fm = ferr_im(param_fm, param_im, fm);
    fmdf = fm + df;
    err_fmdf = ferr_im(param_fm, param_im, fmdf);
    derr_df = (err_fmdf-err_fm)/(fmdf-fm);
    if derr_df > derr_df_tar
        fr = fm;
    elseif derr_df < derr_df_tar
        fl = fm;
    end
    fm = (fl+fr)/2;
end
f = round(fm); 
param_im.Q(1,1) = f;
param_im.Q(2,2) = f;

end

%%
function err = ferr_im(param_fm, param_im, f)
% Calculate the fitting error with respect to the given f (note: here we only consider
% R and t while keeping wid(mean) and wexp(neutral) as fixed. In this case, the fitting 
% error is a convex function of f. 

% coarse pose estimation
vert = coef2vert(param_im.wid, param_im.wexp, param_fm);
nvert = length(vert)/3;
vert = reshape(vert, 3, nvert);
lmk_obj = vert(:,param_im.indlmk_mesh); 
Q = param_im.Q;
Q(1,1) = f; Q(2,2) = f;
indlmk_mesh = param_im.indlmk_mesh;
idlmkfit_pose = 1:length(indlmk_mesh);

% update contour landmarks on the mesh
% [R, ~] = solve_pose(param_im.lmk_im(idlmkfit_pose,:), lmk_obj(:,idlmkfit_pose)', Q);
% indlmk_mesh = update_lmkind(param_im.indlmk_mesh, R, vert, param_fm.lmk_horzlines);

% solve for pose parameters 
lmk_obj = vert(:, indlmk_mesh);
[R, T] = solve_pose(param_im.lmk_im(idlmkfit_pose,:), (lmk_obj(:,idlmkfit_pose))', Q);

% landmark displacements
lmk_obj = vert(:, indlmk_mesh);
lmk_camq = Q*(R*lmk_obj + repmat(T, 1, size(lmk_obj, 2)));
lmk_proj = [lmk_camq(1, :)./lmk_camq(3, :); lmk_camq(2, :)./lmk_camq(3, :)]; 
dev = param_im.lmk_im' - lmk_proj;  

% error
err = 0; 
K = size(dev, 2);
for k = 1:K
    err = err + norm(dev(:, k));
end
err = err / K;

% fprintf('f = %d, error: %.4f\n', f, err);

end