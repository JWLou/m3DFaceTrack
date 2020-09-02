function lmk = faceAlign2d(img, dmDetect, faceT)
% Detect facial landmarks on a given image

faces = double(faceDetection(img));
if ~isempty(faces)
    if size(faces, 1) > 1
        bbox = largestFace(faces);    
    else
        bbox = faces;
    end
    lmk = lmkDetect(img, bbox, faceT, dmDetect);
end

end