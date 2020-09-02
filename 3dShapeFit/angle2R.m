function R = angle2R(angleX, angleY, angleZ)
%ANGLE2R calculates the rotation matrix with the intrinsic rotation 
%   composition - ZY'X"(equivalent to the extrinsic rotation composition - 
%   XYZ with the same rotation angles) obeying the right-handed rule. 

phi = angleX;
theta = angleY;
psi = angleZ;

Rx = [1 0 0 ; 0 cos(phi) -sin(phi); 0 sin(phi) cos(phi)];
Ry = [cos(theta) 0 sin(theta); 0 1 0; -sin(theta) 0 cos(theta)];
Rz = [cos(psi) -sin(psi) 0; sin(psi) cos(psi) 0; 0 0 1];

R = Rz * Ry * Rx;

end

