function result = sigmoid(x)
% Sigmoid 激活函数
    result = 1 ./ (1 + exp(-x));
end