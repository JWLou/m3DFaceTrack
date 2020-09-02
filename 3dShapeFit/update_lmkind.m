function indlmk_pose = update_lmkind(indlmk_pose, R, vert, lmk_horzlines)
%UPDATE_LMKIND updates contour landmarks on the mesh across poses.

% [~, angleY, ~] = R2angle(R);
nlmk = length(lmk_horzlines);
midnum = (nlmk+1)/2;
vertR = R*vert;
for i = 1:nlmk
    if i <= midnum
        [~, ind] = min(vertR(1,lmk_horzlines{i}));
    else
        [~, ind] = max(vertR(1,lmk_horzlines{i}));
    end
    indlmk_pose(i) = lmk_horzlines{i}(ind);           
end

end