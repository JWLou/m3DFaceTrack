function wexpk = solve_wexp(wexp0, Bexp, vert_id, lmk_im, Q, R, T, wlmk)
%SOLVE_WEXP solves for expression coefficients 

% l-bfgs-b
lmk_im = lmk_im';
wreg = 1; % L1-norm regularization term weight, sparsity constraint
F = @(wexp) fmin(wexp, Bexp, vert_id, lmk_im, Q, R, T, wlmk, wreg);
G = @(wexp) gfmin(wexp, Bexp, vert_id, lmk_im, Q, R, T, wlmk, wreg);

nexp = length(wexp0);
lb = zeros(nexp, 1); % lower bound - 0
ub = ones(nexp, 1);  % upper bound - 1
fcn = @(wexp) fminunc_wrapper(wexp, F, G); 

opts = struct( 'x0', wexp0, 'm', 5, 'printEvery', 0, 'factr', 1e7, 'pgtol', 1e-5, 'maxIts', 1e2, 'maxTotalIts', 5e3);
[wexpk, ~, ~] = lbfgsb(fcn, lb, ub, opts);

end

function E = fmin(wexp, Bexp, vert_id, lmk_im, Q, R, T, wlmk, wreg)
%FMIN defines the objective function 

lmk_obj = reshape(vert_id + Bexp*wexp, 3, []);
lmk_cam = R*lmk_obj + repmat(T, 1, size(lmk_obj, 2));
lmk_camq = Q*lmk_cam;
lmk_proj = [lmk_camq(1,:)./lmk_camq(3,:); lmk_camq(2,:)./lmk_camq(3,:)]; 
dev = lmk_im - lmk_proj;      

dev = dev .* repmat(wlmk, 2, 1);
E = sum(sum(dev.^2)) + wreg*sum(wexp); 

end

function G = gfmin(wexp, Bexp, vert_id, lmk_im, Q, R, T, wlmk, wreg)
%GFMIN calculates the gradient of fmin with respect to wexp

lmk_obj = reshape(vert_id + Bexp*wexp, 3, []);
lmk_cam = R*lmk_obj + repmat(T, 1, size(lmk_obj, 2));
lmk_camq = Q*lmk_cam;
lmk_proj = [lmk_camq(1,:)./lmk_camq(3,:); lmk_camq(2,:)./lmk_camq(3,:)]; 
dev = lmk_im - lmk_proj;     
f = Q(1,1);

nexp = size(Bexp, 2);
G = zeros(nexp, 1);
K = size(lmk_cam, 2);
for k = 1:K
    Pc = lmk_cam(:,k); 
    Xc = Pc(1); Yc = Pc(2); Zc = Pc(3); Zc2 = Zc^2;
    d = dev(:, k);

    iz = k*3; ix = iz-2; 
    dPw_dwexp = Bexp(ix:iz,:);
    dPc_dwexp = R*dPw_dwexp;
    dd_dwexp = -f/Zc2*[dPc_dwexp(1,:)*Zc - Xc*dPc_dwexp(3,:); dPc_dwexp(2,:)*Zc - Yc*dPc_dwexp(3,:)];
    dE_dwexp_1 = wlmk(k)*2*d'*dd_dwexp;
    G = G + dE_dwexp_1';
end
dE_dwexp_2 = wreg*ones(nexp, 1);
G = G + dE_dwexp_2;

end









