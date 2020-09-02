function vert = coef2vert(wid, wexp, param_fm)
%COEF2VERT produces a 3D face point cloud from fm coefficients.

ndim = size(wid,1);
vert_id = param_fm.mu_id + param_fm.pc_id(:,1:ndim) * (wid .* param_fm.ev_id(1:ndim));
vert_exp = param_fm.delta_bldshp * wexp;
vert = vert_id + vert_exp;

end