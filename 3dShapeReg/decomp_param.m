function [wexp, R, T, D] = decomp_param(P, indset)
%DECOMP_PARAM decomposes the shape parameter vector.

wexp = P(indset{1});
eulerR = P(indset{2});
R = angle2R(eulerR(1), eulerR(2), eulerR(3));
T = P(indset{3});
D = P(indset{4});
D = reshape(D, 2, []);
D = D';

end