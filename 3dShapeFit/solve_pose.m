function [R, T] = solve_pose(pt2d, pt3d, Q)
%SOLVE_POSE estimates the head pose using POSIT under landmark constraints

%% POSIT
[R, T] = modernPosit(pt2d, pt3d, Q(1,1), [Q(1, 3),Q(2, 3)]);

end