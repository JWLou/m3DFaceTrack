function lmksDetected = lmkDetect(faceImg, faceRegion, faceT, dmDetect)
% Detect facial landmarks on a single image 

faceSquare = faceT.bbox;
meanface = faceT.pts;                     
nLmk = length(faceT.pts);   
xScale = faceSquare(3)/faceRegion(3);
yScale = faceSquare(4)/faceRegion(4);
faceRegion_origin_warped = [faceRegion(1)*xScale, faceRegion(2)*yScale];
meanfaceWarped = meanface  + repmat(faceRegion_origin_warped, nLmk, 1);
xkPrev = meanfaceWarped;

nIter = length(dmDetect);
for t = 1:nIter     
    phiTmp = double(xxFeature(faceImg, xScale, yScale, xkPrev'));  
    phikPrev = phiTmp;
    deltax = reshape(dmDetect{t} * [1; phikPrev], [2 nLmk]);
    xk = xkPrev + deltax';  
    xkPrev = xk;
    lmksDetected = xk * [1/xScale, 0; 0, 1/yScale];      
end

end
