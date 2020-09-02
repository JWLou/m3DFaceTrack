function display_face(vert, tri)
%DISPLAY_FACE plots face in 3D

set(gcf, 'Renderer', 'opengl');
trisurf(tri, vert(1, :), vert(2, :), vert(3, :), 0, 'edgecolor', 'none');
set(gca,'cameraviewanglemode','manual');
set(gca, 'Projection','perspective');
axis vis3d;
axis equal;
axis off;
rotate3d on;
grid off;
xlabel('x');
ylabel('y');
zlabel('z');

view(0, -90);
material([.5 .5 .1 1])
camlight('headlight');
% light('Position', [0 0 0], 'Visible', 'on', 'Style', 'infinite');
% lighting gouraud;

end
	
