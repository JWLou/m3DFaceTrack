function [lmk_proj, frame] = show_res(param_fm, param_im)
%SHOW_RES shows the result of the current shape parameters

nlmk = length(param_im.indlmk_mesh);
vert = param_fm.mu_id + param_fm.pc_id*(param_im.wid.*param_fm.ev_id) + param_fm.delta_bldshp*param_im.wexp;
nvert = length(vert)/3;
vert = reshape(vert, 3, nvert);
lmk_obj = vert(:, param_im.indlmk_mesh); 

lmk_camq = param_im.Q*(param_im.R*lmk_obj + repmat(param_im.T, 1, nlmk));
lmk_proj = [lmk_camq(1, :)./lmk_camq(3, :); lmk_camq(2, :)./lmk_camq(3, :)]; 
lmk_proj = lmk_proj + param_im.D';

% clf;
subplot(1,2,1);
imshow(param_im.img);
title('Input face image');
hold on;
plot(lmk_proj(1, :), lmk_proj(2, :), 'r.', 'markersize', 10);
% plot(param_im.lmk_im(:,1), param_im.lmk_im(:,2), 'g.', 'markersize', 10);
hold off;

subplot(1,2,2);
vert_cam = param_im.R*vert + repmat(param_im.T, 1, nvert);
display_face(vert_cam, param_fm.tri);
title('Recovered 3D mesh');

frame = getframe(gcf);

end
