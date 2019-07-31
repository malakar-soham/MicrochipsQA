function g = sigmoid(z)
%SIGMOID Computes sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

g = zeros(size(z));

g=1./(1+exp(-z));

end
