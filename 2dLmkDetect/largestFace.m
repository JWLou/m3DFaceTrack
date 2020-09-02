function maxface = largestFace(faces)
% Return the largest face bounding box

nface = size(faces, 1);
if nface <= 1
    maxface = faces;
else
    area = 0;
    for i = 1:nface
        tmp = faces(i,3)*faces(i,4);
        if  tmp > area
            area = tmp;
            k = i;
        end
    end
    maxface  = faces(k,:);
end   
    
end


 
