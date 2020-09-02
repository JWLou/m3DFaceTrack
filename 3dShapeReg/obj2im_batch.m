function lmkmat_camq = obj2im_batch(trainset, widmat, Pmat, I, J, VR, VQ, param_train, param_fm)
%OBJ2IM_BATCH projects the 3D vertex onto the 2D image plane for a batch of
% samples at a time.

nsample = length(trainset);
bfm = [param_fm.mu_id, param_fm.pc_id, param_fm.delta_bldshp];
lmkmat_obj = zeros(nsample*3, param_train.nlmk);
len = 1000;
istart = 1;
iend = 0;
while istart <= nsample
    iend = iend + len;
    if iend > nsample
        iend = nsample;
    end
    vertmat_obj = bfm * [widmat(:,istart:iend); Pmat(param_train.indset{1},istart:iend)];
    nobj = size(vertmat_obj,2);
    for i = 1:nobj
        isample = istart+i-1;
        indlmk = [3*trainset{isample}.indlmk_mesh'-2; 3*trainset{isample}.indlmk_mesh'-1; 3*trainset{isample}.indlmk_mesh'];
        indlmk = indlmk(:); 
        lmkmat_obj((isample-1)*3+1:isample*3,:) = reshape(vertmat_obj(indlmk,i), 3, []);
    end
    istart = iend + 1;
end

Qs = sparse(I, J, VQ);
Rs = sparse(I, J, VR);
Tmat = Pmat(param_train.indset{3},:);
Tmat = repmat(Tmat(:), 1, param_train.nlmk);
lmkmat_camq = Qs*(Rs*lmkmat_obj + Tmat);

end
