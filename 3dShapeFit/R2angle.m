function [angleX, angleY, angleZ] = R2angle(R)
%R2ANGLE decomposes the rotation matrix into Euler angles in accordance
%   with the composition - ZY'X"(equivalent to the extrinsic rotation 
%   composition - XYZ with the same rotation angles) under the right-handed
%   rule. 

% get rotation angles from the rotation matrix, XYZ-extrinsic, left-handed
[angleX, angleY, angleZ] = dcm2angle(R, 'XYZ'); 

% convert to right-handed 
angleX = -angleX;
angleY = -angleY;
angleZ = -angleZ;

end

