function p = predictOneVsAll(all_theta, X)
    
X = [ones(size(X,1),1) X];

[val, index] = max(X * all_theta', [], 2);

p = index;
    
end   