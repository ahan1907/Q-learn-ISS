clear;
clc;

optPol_25 = table2array(readtable('optPolicy_25.xlsx'));
optPol_50 = table2array(readtable('optPolicy_50.xlsx'));

%%
grid1d = linspace(-10,40,25);
[X25, Y25, Z25] = ndgrid(grid1d, grid1d, grid1d);  % grid of indices
% opt_25vals = mean(optPol_25, 2); 
opt_25vals = optPol_25(:,3); 

% Flatten grid for scatter3
x25 = X25(:);
y25 = Y25(:);
z25 = Z25(:);

% Plot
figure;
scatter3(x25, y25, z25, 15, opt_25vals, 'filled');
xlabel('Room 1');
ylabel('Room 2');
zlabel('Room 3');
title('3D Visualization of Q-values');
colorbar;
view(135, 30);  % Adjust viewing angle

%%
grid1d = linspace(-10,40,50);
[X50, Y50, Z50] = ndgrid(grid1d, grid1d, grid1d);  % grid of indices
% opt_50vals = mean(optPol_50, 2); 
opt_50vals = optPol_50(:,3); 

% Flatten grid for scatter3
x50 = X50(:);
y50 = Y50(:);
z50 = Z50(:);

% Plot
figure;
scatter3(x50, y50, z50, 15, opt_50vals, 'filled');
xlabel('Room 1');
ylabel('Room 2');
zlabel('Room 3');
title('3D Visualization of Q-values');
colorbar;
view(135, 30);  % Adjust viewing angle