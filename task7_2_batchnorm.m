% MLP Picture Classification based on task7_2.m and task8.m

% Clear environment
clc; clear;

mini_batch_size = 64;

% Initialize timer
lastVerifyTime = tic;

% Check if GPU is available
if gpuDeviceCount > 0
    useGPU = true;
    disp('GPU detected, using GPU for acceleration...');
    mini_batch_size = 512; % Increase mini batch size for better GPU utilization
else
    useGPU = false;
    disp('No GPU detected, using CPU...');
end

% Load dataset
[trainFeatures, trainLabels] = load_data('train');
[testFeatures, testLabels] = load_data('test');


% Network parameters
layers = [128 * 128, 32 * 32, 8 * 8, 7]; % Deeper architecture
initial_lr = 1e-3;
epochs = 1000;
varFrec = 50;


% Prepare dataset
trainFeatures = double(trainFeatures)' / 255.0; % Normalize features
testFeatures = double(testFeatures)' / 255.0;
trainLabels = double(trainLabels)'; % Labels are now one-hot encoded
testLabels = double(testLabels)'; % Labels are now one-hot encoded


% Apply label smoothing
smoothing_factor = 0.1;
trainLabels = trainLabels * (1 - smoothing_factor) + smoothing_factor / size(trainLabels, 1);
testLabels = testLabels * (1 - smoothing_factor) + smoothing_factor / size(testLabels, 1);

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
gradMaxValues = [];
gradMinValues = [];


% Display elapsed time since last verification
elapsedTime = toc(lastVerifyTime);
disp(['Time for initialising: ', num2str(elapsedTime), ' seconds'])
% Reset timer
lastVerifyTime = tic;

for i = 1:epochs
    % Shuffle training data
    idx = randperm(trainImages);
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
        max_gradient_value = 1; % Reduce the clipping value for better stability
        grads = clip_gradients(grads, max_gradient_value);
        
        % Update parameters using Adam optimizer
        [params, moments] = update_parameters_adam(params, grads, moments, lr, i);
    end
    
    %Learning rate scheduling: Reduce learning rate every 20 epochs
    if mod(i, 100) == 0
        lr = lr * 0.99;
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


% 
%         if i >= 250 && mod(i, 200) == 0
%             if acc < 0.2
%                 lr = 0.8;
%             end
%         end
        testAcc(end + 1) = acc;
        disp(['test acc: ', num2str(acc)])
        
        % Display gradient information for debugging
        disp('Gradient statistics for each layer:');
        grad_max_vals = [];
        grad_min_vals = [];
        for l = 1:length(grads)
            dW = grads{l}{2};
            db = grads{l}{3};
            grad_max_vals(end + 1) = max(dW(:));
            grad_min_vals(end + 1) = min(dW(:));
            disp(['Layer ', num2str(l), ' - dW max: ', num2str(max(dW(:))), ', dW min: ', num2str(min(dW(:))), ', db max: ', num2str(max(db(:))), ', db min: ', num2str(min(db(:)))]);
        end
        gradMaxValues = [gradMaxValues; grad_max_vals];
        gradMinValues = [gradMinValues; grad_min_vals];



        
        % Display elapsed time since last verification
        elapsedTime = toc(lastVerifyTime);
        disp(['Time since last verification: ', num2str(elapsedTime), ' seconds'])
        % Reset timer
        lastVerifyTime = tic;

        % Plot gradient max and min values dynamically
        figure(2);
        clf;
        hold on;
        plot(1:size(gradMaxValues, 1), gradMaxValues, '-r', 'LineWidth', 1.5);
        plot(1:size(gradMinValues, 1), gradMinValues, '-b', 'LineWidth', 1.5);
        title('Gradient Max and Min Values Over Epochs');
        xlabel('Verification Step');
        ylabel('Gradient Value');
        legend('Max Gradient', 'Min Gradient');
        grid on;
        drawnow;
    end
end

% Plot results
fig = figure;
x = varFrec:varFrec:epochs;
plot(x, trainLoss, '-*b', x, trainAcc * 100, '-ob', x, testLoss, '-*r', x, testAcc * 100, '-or');
axis([0, epochs, 0, 100])
set(gca, 'XTick', [0:varFrec:epochs])
set(gca, 'YTick', [0:5:100])
legend('trainLoss', 'trainAcc (%)', 'testLoss', 'testAcc (%)');
xlabel('epoch')
imwrite(frame2im(getframe(fig)), 'results.png');

% Function Definitions

function [A, cache] = activation_function(Z, activation_type)
    % Activation function switch to handle different activation types
    switch activation_type
        case 'swish'
            A = Z ./ (1 + exp(-Z));
        case 'leaky_relu'
            A = max(0.2 * Z, Z);
        case 'relu'
            A = max(0, Z);
        case 'sigmoid'
            A = 1 ./ (1 + exp(-Z));
        case 'softmax'
            A = softmax(Z);
        otherwise
            error('Unsupported activation function');
    end
    cache = Z;
end

function dZ = activation_function_backward(dA, cache, activation_type)
    % Backward propagation for different activation functions
    Z = cache;
    switch activation_type
        case 'swish'
            sigmoid_Z = 1 ./ (1 + exp(-Z));
            dZ = dA .* (sigmoid_Z + Z .* sigmoid_Z .* (1 - sigmoid_Z));
        case 'leaky_relu'
            dZ = dA;
            dZ(Z <= 0) = dZ(Z <= 0) * 0.2;
        case 'relu'
            dZ = dA;
            dZ(Z <= 0) = 0;
        case 'sigmoid'
            s = 1 ./ (1 + exp(-Z));
            dZ = dA .* s .* (1 - s);
        case 'softmax'
            Z = cache;
            s = softmax(Z);
            dZ = dA .* s .* (1 - s);
        otherwise
            error('Unsupported activation function');
    end
end

function [W, b] = initialize_weights(n_x, n_y)
    W = randn(n_y, n_x) * 0.01;
%     sqrt(2 / n_x); 
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
        Z = batchnorm(Z); % Batch Normalization         
        [A, activation_cache] = activation_function(Z, 'swish'); 
        caches{end + 1} = {linear_cache, activation_cache};
    end
    A_prev = A;
    [Z, linear_cache] = linear_forward(A_prev, params{L}{1}, params{L}{2});
    A = softmax(Z); % Apply softmax to get probabilities for each label
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
%     loss = sum(sum(- ((Y .* log(AL + 1e-8)) + ((1 - Y) .* log(1 - AL + 1e-8))))) / size(AL, 2);
end


function [dA_prev, dW, db] = linear_backward(dZ, cache)
    [A_prev, W, b] = cache{:};
    m = size(A_prev, 2);
    dW = 1 / m * dZ * A_prev' + 1e-4 / m * W;
    db = 1 / m * sum(dZ, 2);
    dA_prev = W' * dZ;
end

function [dA_prev, dW, db] = linear_activation_backward(dA, cache, activation_type)
    [linear_cache, activation_cache] = cache{:};
%     dZ = activation_function_backward(dA, activation_cache, activation_type);

    if activation_cache == 1
        dZ = activation_function_backward(dA, activation_cache, 'sigmoid');
    else
        dZ = activation_function_backward(dA, activation_cache, 'softmax');
    end
    [dA_prev, dW, db] = linear_backward(dZ, linear_cache);
end

function grads = model_backward(AL, Y, caches)
    grads = {};
    L = length(caches);
    dAL = AL - Y; % Derivative of softmax with cross-entropy
    [dA, dW, db] = linear_activation_backward(dAL, caches{L}, 'swish');
    grads{end + 1} = {dA, dW, db};

    for l = L - 1:-1:1
        [dA, dW, db] = linear_activation_backward(dA, caches{l}, 'swish');
        grads{end + 1} = {dA, dW, db};
    end
end

function [params, moments] = update_parameters_adam(params, grads, moments, lr, t)
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1e-8;
    
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
        mW_hat = mW / (1 - beta1^t);
        mb_hat = mb / (1 - beta1^t);
        
        % Compute bias-corrected second raw moment estimate
        vW_hat = vW / (1 - beta2^t);
        vb_hat = vb / (1 - beta2^t);
        
        % Update parameters
        W = W - lr * mW_hat ./ (sqrt(vW_hat) + epsilon);
        b = b - lr * mb_hat ./ (sqrt(vb_hat) + epsilon);
        
        % Store updated parameters and moments
        params{i} = {W, b};
        moments{i} = {mW, mb, vW, vb};
    end
end

function A = softmax(Z)
    % Subtract the maximum value from each column for numerical stability
    Z = Z - max(Z, [], 1);
    expZ = exp(Z);
    A = expZ ./ sum(expZ, 1);
end

function grads = clip_gradients(grads, max_value)
    for i = 1:length(grads)
        grads{i}{2} = min(max(grads{i}{2}, -max_value), max_value);
        grads{i}{3} = min(max(grads{i}{3}, -max_value), max_value);
    end
end

function acc = compute_accuracy(AL, Y)
    [~, pred] = max(AL, [], 1);
    [~, true_labels] = max(Y, [], 1);
    acc = sum(pred == true_labels) / size(Y, 2);
end

function Z_norm = batchnorm(Z)
    % Batch Normalization 操作
    epsilon = 1e-8; % 防止除零
    mu = mean(Z, 2); % 计算均值
    sigma2 = var(Z, 0, 2); % 计算方差

    % 标准化
    Z_norm = (Z - mu) ./ sqrt(sigma2 + epsilon);

    % 可选的 scale 和 shift 参数（gamma 和 beta），可以用来提高灵活性
    % 这里假设 gamma = 1, beta = 0，等价于只进行标准化
end
