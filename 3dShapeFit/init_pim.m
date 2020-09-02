function param_im = init_pim(param_fm, img, pts, f, wid, wexp)
%INIT_PIM initializes the shape parameters of the image

imh = size(img, 1);
imw = size(img, 2);
if ~isempty(pts)
    param_im.lmk_im = pts;
    param_im.img = img;
    param_im.Q = [f,0,imw/2; 0,f,imh/2; 0,0,1];
    if nargin < 5
        param_im.wid = zeros(param_fm.nid, 1);
        param_im.wexp = zeros(param_fm.nexp, 1);
    elseif nargin >= 5
        param_im.wid = wid; 
    elseif nargin >= 6
        param_im.wexp = wexp;
    end 
    param_im.indlmk_mesh = param_fm.indlmk_mesh; 
else
    param_im = [];     
end

end