clear; close all; clc

input_layer_size  = 400;  % 20x20 Input Images of Digits
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)
                          
%% loading data
fprintf('loading data and visuallize:')
load('ex3data1.mat')
m = size(X,1);
rand_indicate = randperm(m);
cell = X(rand_indicate(1:100),:);
% displayData(cell);

%% one vs all
fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda = 0.1;
[all_theta] = oneVsAll(X, y, num_labels, lambda);
%% ================ Part 3: Predict for One-Vs-All ================
%  After ...
pred = predictOneVsAll(all_theta, X)
% 
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

