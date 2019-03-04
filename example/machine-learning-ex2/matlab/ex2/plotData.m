function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
m = length(y);
sz = 40;
k = find(y==0);
scatter(X(k,1),X(k,2),sz,'ro','r');
k2 = find(y==1);
scatter(X(k2,1),X(k2,2),sz,'b+','r');

% ===========.==============================================================



hold off;

end
