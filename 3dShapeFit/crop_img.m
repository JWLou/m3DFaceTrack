function [img_crop, lmk_crop, tlx, tly, s] = crop_img(img, lmk, lmkT, rectT)
%CROP_IMG crops the face image and scales the cropped image to a unified
%   template.

imh = size(img, 1);
imw = size(img, 2);
nlmk = size(lmk, 1);
clmk = mean(lmk);
clmkT = mean(lmkT);

[~, ~, tform] = procrustes(lmkT, lmk, 'reflection', false);
s = tform.b;
tlx = floor(clmk(1,1) - clmkT(1,1)/s);
tly = floor(clmk(1,2) - clmkT(1,2)/s);
img_pad = img;
if tlx < 0
    wpadl = abs(tlx);
    img_pad = padarray(img_pad, [0, wpadl], 0, 'pre');
end
if tly < 0
    hpadt = abs(tly);
    img_pad = padarray(img_pad, [hpadt, 0], 0, 'pre');
end
brx = ceil(tlx + rectT.w/s);
bry = ceil(tly + rectT.h/s);
if brx > imw
    wpadr = brx - imw;
    img_pad = padarray(img_pad, [0, wpadr], 0, 'post');
end
if bry > imh
    hpadb = bry - imh;
    img_pad = padarray(img_pad, [hpadb, 0], 0, 'post');
end

rect = [max(tlx, 0), max(tly, 0), brx-tlx, bry-tly];
img_crop = imcrop(img_pad, rect);
img_crop = imresize(img_crop, s, 'nearest'); % scale
lmk_crop(:,1) = (lmk(:,1) - tlx*ones(nlmk, 1)) * s;
lmk_crop(:,2) = (lmk(:,2) - tly*ones(nlmk, 1)) * s;

end

%%
% indlmk = [36:46,65]; indlmk = indlmk';
% nlmk = size(lmk, 1);
% clmk = mean(lmk(indlmk,:));
% clmkT = mean(lmkT(indlmk,:));
% 
% [~, ~, tform] = procrustes(lmkT, lmk, 'reflection', false);
% s = tform.b;
% 
% lmks = lmk*s;
% clmks = clmk*s;
% imgs = imresize(img, s, 'nearest'); % scale
% 
% tlx = clmks(1)-clmkT(1);
% tly = clmks(2)-clmkT(2);
% rect = [tlx, tly, rectT.w, rectT.h];
% img_crop = imcrop(imgs, rect);
% lmk_crop(:,1) = (lmks(:,1) - tlx*ones(nlmk, 1));
% lmk_crop(:,2) = (lmks(:,2) - tly*ones(nlmk, 1));
