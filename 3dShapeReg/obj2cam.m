function vert_cam = obj2cam(wid, wexp, R, T, param_fm)
%OBJ2CAM converts object coordinates to image coordinates

nvert = length(param_fm.mu_id)/3;
vert_obj = param_fm.mu_id + param_fm.pc_id*(wid.*param_fm.ev_id) + param_fm.delta_bldshp*wexp;
vert_obj = reshape(vert_obj, 3, nvert);
vert_cam = R*vert_obj + repmat(T, 1, nvert);
vert_cam = vert_cam';

end