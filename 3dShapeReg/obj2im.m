function vert_im = obj2im(wid, wexp, Q, R, T, param_fm)
%OBJ2IM projects the 3D vertex onto the 2D image plane.

nvert = length(param_fm.mu_id)/3;
vert_obj = param_fm.mu_id + param_fm.pc_id*(wid.*param_fm.ev_id) + param_fm.delta_bldshp*wexp;
vert_obj = reshape(vert_obj, 3, nvert);
vert_camq = Q*(R*vert_obj + repmat(T, 1, nvert));
vert_im = [vert_camq(1,:)./vert_camq(3,:); vert_camq(2,:)./vert_camq(3,:)];   
vert_im = vert_im';

end