function widk = solve_wid(wid0, ev_id, Bid, vert_exp, lmk_im, Q, R, T, wlmk)
%SOLVE_WID solves for identity coefficients 

% l-bfgs-b
lmk_im = lmk_im';
wreg = 10; % L2-norm regularization term weight
F = @(wid) fmin(wid, ev_id, Bid, vert_exp, lmk_im, Q, R, T, wlmk, wreg);
G = @(wid) gfmin(wid, ev_id, Bid, vert_exp, lmk_im, Q, R, T, wlmk, wreg);

nid = length(wid0);
lb = inf(nid, 1);  % there is no lower bound
ub = inf(nid, 1);  % there is no upper bound
fcn = @(wid) fminunc_wrapper(wid, F, G); 

opts = struct( 'x0', wid0, 'm', 5, 'printEvery', 0, 'factr', 1e7, 'pgtol', 1e-5, 'maxIts', 1e2, 'maxTotalIts', 5e3);
[widk, ~, ~] = lbfgsb(fcn, lb, ub, opts);

end

function E = fmin(wid, ev_id, Bid, vert_exp, lmk_im, Q, R, T, wlmk, wreg)
%FMIN defines the objective function 

lmk_obj = reshape(Bid*(wid.*ev_id) + vert_exp, 3, []);
lmk_cam = R*lmk_obj + repmat(T, 1, size(lmk_obj, 2));
lmk_camq = Q*lmk_cam;
lmk_proj = [lmk_camq(1,:)./lmk_camq(3,:); lmk_camq(2,:)./lmk_camq(3,:)]; 
dev = lmk_im - lmk_proj;      

dev = dev .* repmat(wlmk, 2, 1);
E = sum(sum(dev.^2)) + wreg*sum(wid.^2); 

end

function G = gfmin(wid, ev_id, Bid, vert_exp, lmk_im, Q, R, T, wlmk, wreg)
%GFMIN calculates the gradient of fmin with respect to wid

lmk_obj = reshape(Bid*(wid.*ev_id) + vert_exp, 3, []);
lmk_cam = R*lmk_obj + repmat(T, 1, size(lmk_obj, 2));
lmk_camq = Q*lmk_cam;
lmk_proj = [lmk_camq(1,:)./lmk_camq(3,:); lmk_camq(2,:)./lmk_camq(3,:)]; 
dev = lmk_im - lmk_proj;     
f = Q(1,1);

nid = size(Bid, 2);
G = zeros(nid, 1);
K = size(lmk_cam, 2);
for k = 1:K
    Pc = lmk_cam(:,k); 
    Xc = Pc(1); Yc = Pc(2); Zc = Pc(3); Zc2 = Zc^2;
    d = dev(:, k);

    iz = k*3; ix = iz-2; 
    dPw_dwid = Bid(ix:iz,:).*repmat(ev_id', 3, 1);
    dPc_dwid = R*dPw_dwid;
    dd_dwid = -f/Zc2*[dPc_dwid(1,:)*Zc - Xc*dPc_dwid(3,:); dPc_dwid(2,:)*Zc - Yc*dPc_dwid(3,:)];
    dE_dwid_1 = wlmk(k)*2*d'*dd_dwid;
    G = G + dE_dwid_1';
end
dE_dwid_2 = 2*wreg*wid;
G = G + dE_dwid_2;

end









