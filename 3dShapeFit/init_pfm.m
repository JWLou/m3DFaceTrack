function param_fm = init_pfm
%INIT_PFM initializes the parameteric face model

fprintf('Initializing the parametric face model...\n');

% identity
param_fm = load('.\model\BFM-id.mat');
param_fm.mu_id = double(param_fm.mu_id);
param_fm.pc_id = double(param_fm.pc_id(:,1:80));
param_fm.ev_id = double(param_fm.ev_id(1:80));
param_fm.nid = size(param_fm.pc_id, 2);

% expression
load('.\model\BFM-bldshp.mat'); 
param_fm.delta_bldshp = double(delta_bldshape);
param_fm.nexp = size(param_fm.delta_bldshp, 2);

% other 
load('.\model\BFM-info.mat'); 
param_fm.tri = tri;
param_fm.indtri_face = ind_tri_face_1;
param_fm.indvert_face = unique(param_fm.tri(param_fm.indtri_face, :));
param_fm.indlmk_mesh = indlmk_mesh_66;
param_fm.lmk_horzlines = indlmk_horzlines_66;
param_fm.faceimT = faceimT_66;

% Delaunay triangles formed by 2D landmarks
DT = DT_66;
nDT = length(DT);
sDT = zeros(nDT, 1);
DTc = zeros(nDT, 2);
for i = 1:nDT
    pt1 = param_fm.faceimT.lmkT(DT(i,1),:);
    pt2 = param_fm.faceimT.lmkT(DT(i,2),:);
    pt3 = param_fm.faceimT.lmkT(DT(i,3),:);
    e1 = norm(pt1-pt2);
    e2 = norm(pt3-pt2);
    e3 = norm(pt1-pt3);
    p = (e1+e2+e3) / 2;
    sDT(i) = sqrt(p * (p-e1) * (p-e2) * (p-e3));
    
    DTc(i,:) = mean([pt1;pt2;pt3]);
end
sDT = sDT/sum(sDT);
param_fm.DT = DT;
param_fm.sDT = sDT;
param_fm.DTc = DTc;

fprintf('Initialization done.\n');

end