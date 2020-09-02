function param_reg = get_param_reg
%GET_PARAM_REG gets regression parameters

param_reg.niter = 10; % the iteration amount
param_reg.npx = 600; % the number of pixels
param_reg.k = 0.3*ones(param_reg.niter, 1);  
param_reg.F = 5; % the number of features in fern, default: 5
param_reg.beta = 1000; % shrinkage parameter, default: 1000
param_reg.nfern_int = {80;80;80;80};

end