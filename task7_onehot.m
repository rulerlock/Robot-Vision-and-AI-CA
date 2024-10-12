% MLP Picture Classification based on task7_2.m and task8.m

% Clear environment
clc; clear;

% Check if GPU is available
if gpuDeviceCount > 0
    useGPU = true;
    disp('GPU detected, using GPU for acceleration...');
    mini_batch_size = 128; % Increase mini batch size for better GPU utilization
else
    useGPU = false;
    disp('No GPU detected, using CPU...');
end

% Load dataset
[trainFeatures, trainLabels] = load_data('train');
[testFeatures, testLabels] = load_data('test');
categories = eye(7);

% Network parameters
layers = [128 * 128, 64 * 64, 32 * 32, 16 * 16, 7]; % Deeper architecture
initial_lr = 0.001;
epochs = 100;
varFrec = 10;
mini_batch_size = 64;

% Prepare dataset
trainFeatures = double(trainFeatures)' / 255.0; % Normalize features
testFeatures = double(testFeatures)' / 255.0;
trainLabels = double(trainLabels)'; % Labels are now one-hot encoded
testLabels = double(testLabels)'; % Labels are now one-hot encoded

if useGPU
    trainFeatures = gpuArray(trainFeatures);
    trainLabels = gpuArray(trainLabels);
    testFeatures = gpuArray(testFeatures);
    testLabels = gpuArray(testLabels);
end

numImages = size(trainFeatures, 2);
testImages = size(testLabels, 2);
trainImages = size(trainLabels, 2);

% Initialize network
params = initialize_network(layers);
moments = initialize_moments(layers); % Initialize moments for Adam optimizer

if useGPU
    for k = 1:length(params)
        params{k}{1} = gpuArray(params{k}{1});
        params{k}{2} = gpuArray(params{k}{2});
        moments{k}{1} = gpuArray(moments{k}{1});
        moments{k}{2} = gpuArray(moments{k}{2});
        moments{k}{3} = gpuArray(moments{k}{3});
        moments{k}{4} = gpuArray(moments{k}{4});
    end
end

% Training loop
trainLoss = [];
testLoss = [];
trainAcc = [];
testAcc = [];
lr = initial_lr;

for i = 1:epochs
    % Shuffle training data
    idx = randperm(numImages);
    trainFeatures = trainFeatures(:, idx);
    trainLabels = trainLabels(:, idx);
    
    for j = 1:mini_batch_size:numImages
        % Mini-batch gradient descent
        end_idx = min(j + mini_batch_size - 1, numImages);
        X_batch = trainFeatures(:, j:end_idx);
        Y_batch = trainLabels(:, j:end_idx);
        
        % Forward propagation
        [AL, caches] = model_forward(X_batch, params);
        
        % Backward propagation to calculate gradients
        grads = model_backward(AL, Y_batch, caches);
        
        % Clip the gradients to avoid exploding gradients
        max_gradient_value = 2; % Reduce the clipping value for better stability
        grads = clip_gradients(grads, max_gradient_value);
        
        % Update parameters using Adam optimizer
        [params, moments] = update_parameters_adam(params, grads, moments, lr, i);
    end
    
    % Learning rate scheduling: Reduce learning rate every 20 epochs
    if mod(i, 20) == 0
        lr = lr * 0.5;
    end
    
    % Calculate and display metrics every 'varFrec' epochs
    if mod(i, varFrec) == 0
        % Training metrics
        [AL, ~] = model_forward(trainFeatures, params);
        loss = compute_loss(AL, trainLabels);
        disp(['epoch: ', num2str(i)])
        disp(['train loss: ', num2str(loss)])
        trainLoss(end + 1) = loss;
        acc = compute_accuracy(AL, trainLabels);
        trainAcc(end + 1) = acc;
        disp(['train acc: ', num2str(acc)])
        
        % Test metrics
        [AL, ~] = model_forward(testFeatures, params);
        loss = compute_loss(AL, testLabels);
        disp(['test loss: ', num2str(loss)])
        testLoss(end + 1) = loss;
        acc = compute_accuracy(AL, testLabels);
        testAcc(end + 1) = acc;
        disp(['test acc: ', num2str(acc)])
    end
end

% Plot results
fig = figure;
x = varFrec:varFrec:epochs;
plot(x, trainLoss, '-*b', x, trainAcc, '-ob', x, testLoss, '-*r', x, testAcc, '-or');
axis([0, epochs, 0, 1])
set(gca, 'XTick', [0:varFrec:epochs])
set(gca, 'YTick', [0:0.1:1])
legend('trainLoss', 'trainAcc', 'testLoss', 'testAcc');
xlabel('epoch')
imwrite(frame2im(getframe(fig)), 'results.png');

% Function Definitions

function [A, cache] = relu(Z)
    A = max(0, Z);
    cache = Z;
end

function dZ = relu_backward(dA, cache)
    Z = cache;
    dZ = dA;
    dZ(Z <= 0) = 0;
end

function [W, b] = initialize_weights(n_x, n_y)
    W = randn(n_y, n_x) * sqrt(2 / n_x); % He initialization
    b = zeros(n_y, 1);
end

function params = initialize_network(layers)
    params = {};
    for i = 1:length(layers) - 1
        [W, b] = initialize_weights(layers(i), layers(i + 1));
        params{end + 1} = {W, b};
    end
end

function moments = initialize_moments(layers)
    moments = {};
    for i = 1:length(layers) - 1
        [W, b] = initialize_weights(layers(i), layers(i + 1));
        moments{end + 1} = {zeros(size(W)), zeros(size(b)), zeros(size(W)), zeros(size(b))}; % m and v for Adam
    end
end

function [AL, caches] = model_forward(X, params)
    caches = {};
    A = X;
    L = length(params);
    for i = 1:L - 1
        A_prev = A;
        [Z, linear_cache] = linear_forward(A_prev, params{i}{1}, params{i}{2});
        [A, activation_cache] = relu(Z);
        caches{end + 1} = {linear_cache, activation_cache};
    end
    A_prev = A;
    [Z, linear_cache] = linear_forward(A_prev, params{L}{1}, params{L}{2});
    A = softmax(Z);
    caches{end + 1} = {linear_cache, Z};
    AL = A;
end

function [Z, cache] = linear_forward(A, W, b)
    % Ensure the dimensions match for matrix multiplication
    assert(size(W, 2) == size(A, 1), "Dimensions of W and A do not match for multiplication");
    Z = W * A + b;
    cache = {A, W, b};
end

function loss = compute_loss(AL, Y)
    % Compute categorical cross-entropy loss
    loss = -sum(sum(Y .* log(AL + 1e-8))) / size(AL, 2);
end

function [dA_prev, dW, db] = linear_backward(dZ, cache)
    [A_prev, W, b] = cache{:};
    m = size(A_prev, 2);
    dW = 1 / m * dZ * A_prev' + 1e-4 / m * W;
    db = 1 / m * sum(dZ, 2);
    dA_prev = W' * dZ;
end

function [dA_prev, dW, db] = linear_activation_backward(dA, cache, activation)
    [linear_cache, activation_cache] = cache{:};
    if activation == 1
        dZ = relu_backward(dA, activation_cache);
    else
        dZ = softmax_backward(dA, activation_cache);
    end
    [dA_prev, dW, db] = linear_backward(dZ, linear_cache);
end

function grads = model_backward(AL, Y, caches)
    grads = {};
    L = length(caches);
    dAL = AL - Y; % Derivative of softmax with cross-entropy
    [dA, dW, db] = linear_activation_backward(dAL, caches{L}, 2);
    grads{end + 1} = {dA, dW, db};

    for l = L - 1:-1:1
        [dA, dW, db] = linear_activation_backward(dA, caches{l}, 1);
        grads{end + 1} = {dA, dW, db};
    end
end

function [params, moments] = update_parameters_adam(params, grads, moments, lr, t)
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1e-8;
    alpha = lr;
    
    for i = 1:length(params)
        [W, b] = params{i}{:};
        [dW, db] = grads{end + 1 - i}{2:3};
        [mW, mb, vW, vb] = moments{i}{:};
        
        % Update biased first moment estimate
        mW = beta1 * mW + (1 - beta1) * dW;
        mb = beta1 * mb + (1 - beta1) * db;
        % Update biased second raw moment estimate
        vW = beta2 * vW + (1 - beta2) * (dW .^ 2);
        vb = beta2 * vb + (1 - beta2) * (db .^ 2);
        
        % Compute bias-corrected first moment estimate
        mW_hat = mW / (1 - beta1 ^ t);
        mb_hat = mb / (1 - beta1 ^ t);
        % Compute bias-corrected second raw moment estimate
        vW_hat = vW / (1 - beta2 ^ t);
        vb_hat = vb / (1 - beta2 ^ t);
        
        % Update parameters
        W = W - alpha * mW_hat ./ (sqrt(vW_hat) + epsilon);
        b = b - alpha * mb_hat ./ (sqrt(vb_hat) + epsilon);
        
        % Save updated parameters and moments
        params{i} = {W, b};
        moments{i} = {mW, mb, vW, vb};
    end
end

function acc = compute_accuracy(AL, Y)
    [~, pred] = max(AL, [], 1);
    [~, true_labels] = max(Y, [], 1);
    acc = sum(pred == true_labels) / size(Y, 2);
end

function grads = clip_gradients(grads, max_value)
    for i = 1:length(grads)
        grads{i}{2} = min(max(grads{i}{2}, -max_value), max_value);
        grads{i}{3} = min(max(grads{i}{3}, -max_value), max_value);
    end
end

function A = softmax(Z)
    expZ = exp(Z - max(Z, [], 1)); % Subtract max for numerical stability
    A = expZ ./ sum(expZ, 1);
end

function dZ = softmax_backward(dA, cache)
    Z = cache;
    s = softmax(Z);
    dZ = dA .* s .* (1 - s);
end
