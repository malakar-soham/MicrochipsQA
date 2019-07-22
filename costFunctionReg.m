function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

n = size(theta);

h = sigmoid(X*theta);
y1 = -y.*log(h);
y0 = -(1-y).*log(1-h);

sqrTheta = theta(2:n,1)'*theta(2:n,1);
req = lambda*sqrTheta/(2*m);

J = sum(y1+y0)/m + req;

grad(1) = (X(:,1)'*(h-y))/m;
grad(2:n) = (X(:,2:n)'*(h-y))/m + lambda/m*theta(2:n);



% =============================================================

end
