function [isin, bc] = ptintri(ptq, pttri)
% Test if a point locate inside a triangle based on barycentric coodinate
% system

x = ptq(1); y = ptq(2);
x1 = pttri(1, 1); y1 = pttri(1, 2);
x2 = pttri(2, 1); y2 = pttri(2, 2);
x3 = pttri(3, 1); y3 = pttri(3, 2);
denominator = (y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3);
bc1 = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / denominator;
bc2 = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / denominator;
bc3 = 1 - bc1 - bc2;

if 0 <= bc1 && bc1 <= 1 && 0 <= bc2 && bc2 <= 1 && 0 <= bc3 && bc3<= 1
    isin = true;
else
    isin = false;
end
bc = [bc1; bc2; bc3];

end

