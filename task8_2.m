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
layers = [128 * 128, 16 * 16, 8 * 8, 7]; % Deeper architecture
initial_lr = 1e-3;
epochs = 3000;
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
        lr = lr * 0.8;
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
        
%        % Save confusion matrix for the last verification step
        if mod(i, 500) == 0
            % Compute confusion matrix
            [~, predictedLabels] = max(AL, [], 1); % Predicted labels (index of max probability)
            [~, trueLabels] = max(testLabels, [], 1); % True labels (index of max one-hot encoded value)
            confusionMat = confusionmat(trueLabels, predictedLabels); % Calculate confusion matrix
            display_color_matrix(confusionMat);
        end
        
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

        % Plot loss and accuracy
        figure(1);
        clf;
        hold on;
        plot(1:size(trainLoss,2), trainLoss, '-*m', 'LineWidth', 1.5);
        plot(1:size(trainAcc,2), trainAcc, '-ob', 'LineWidth', 1.5);
        plot(1:size(testLoss,2), testLoss, '-*r', 'LineWidth', 1.5);
        plot(1:size(testAcc,2), testAcc, '-og', 'LineWidth', 1.5);
        title('Accuracy and Lost Values Over Epochs');
        xlabel('Verification Step');
        ylabel('Loss or Accuracy Value');
        legend('trainLoss', 'trainAcc','testLoss', 'testAcc');
        grid on;
        drawnow;
    end
end

% Plot results
fig = figure;
x = varFrec:varFrec:epochs;
plot(x, trainLoss, '-*m', x, trainAcc, '-ob', x, testLoss, '-*r', x, testAcc, '-og');
axis([0, epochs, 0.5, 1.5])
set(gca, 'XTick', [0:varFrec:epochs])
set(gca, 'YTick', [0.5:0.1:1.5])
legend('trainLoss', 'trainAcc', 'testLoss', 'testAcc');
xlabel('epoch')
imwrite(frame2im(getframe(fig)), 'results.png');

% Function Definitions
function display_color_matrix(matrix, filename)
    % Display a color-coded matrix with custom colormap and value annotations
    % Args:
    %   matrix: The input matrix to visualize (e.g., confusion matrix)
    %   filename (optional): Name of the file to save the visualization as an image

    % Convert GPU array to regular array if needed
    if isa(matrix, 'gpuArray')
        matrix = gather(matrix); % Transfer matrix to CPU memory
    end

    % Create a mask for diagonal and non-diagonal elements
    diagonal_mask = eye(size(matrix));
    non_diagonal_mask = 1 - diagonal_mask;

    % Separate diagonal and non-diagonal values
    diagonal_values = matrix .* diagonal_mask;
    non_diagonal_values = matrix .* non_diagonal_mask;

    % Normalize each set of values independently for proper colormap scaling
    max_diag = max(diagonal_values(:));
    max_non_diag = max(non_diagonal_values(:));
    if max_diag == 0, max_diag = 1; end % Avoid division by zero
    if max_non_diag == 0, max_non_diag = 1; end

    % Generate the final matrix for color mapping
    color_matrix = matrix;
    for i = 1:size(matrix, 1)
        for j = 1:size(matrix, 2)
            if i == j % Diagonal elements
                color_matrix(i, j) = matrix(i, j) / max_diag; % Normalize diagonal
            else % Non-diagonal elements
                color_matrix(i, j) = matrix(i, j) / max_non_diag; % Normalize non-diagonal
            end
        end
    end

    % Create a new figure for the matrix visualization
    figure;
    imagesc(color_matrix); % Display the matrix as a color-coded image
    hold on; % Enable overlay for annotations

    % Define the colormap for diagonal and non-diagonal elements
    colormap(create_colormap()); % Apply custom colormap
    colorbar; % Add color bar to indicate intensity scale
    title('Confusion Matrix');
    xlabel('Predicted Labels');
    ylabel('True Labels');
    axis equal;
    xticks(1:size(matrix, 2)); % Set x-tick positions
    yticks(1:size(matrix, 1)); % Set y-tick positions
%     set(gca, 'XTickLabel', 1:size(matrix, 2)); % Set x-tick labels as class numbers
%     set(gca, 'YTickLabel', 1:size(matrix, 1)); % Set y-tick labels as class numbers
    set(gca, 'XTickLabel', {'0', '4', '7', '8', 'A', 'D', 'H'}); % Set x-tick labels as class numbers
    set(gca, 'YTickLabel', {'0', '4', '7', '8', 'A', 'D', 'H'}); % Set y-tick labels as class numbers

    
    set(gca, 'TickLength', [0 0]); % Remove tick marks
    grid on;

    % Annotate matrix with values
    for i = 1:size(matrix, 1)
        for j = 1:size(matrix, 2)
            value = matrix(i, j);
            if value ~= 0 % Skip zero values
                text(j, i, num2str(value, '%.0f'), 'HorizontalAlignment', 'center', ...
                    'VerticalAlignment', 'middle', 'Color', 'black', 'FontWeight', 'bold');
            end
        end
    end

    % Save the visualization as an image file, if filename is provided
    if nargin > 1 && ~isempty(filename)
        saveas(gcf, filename); % Save the figure to the specified filename
    end
end

function cmap = create_colormap()
    % Create a custom colormap for matrix visualization
    % Returns:
    %   cmap: Custom colormap for visualization

    n = 256; % Number of colors

    % Diagonal color gradient (white to blue)
    white_to_blue = [linspace(1, 0, n)', linspace(1, 1, n)', linspace(1, 1, n)'];

    % Non-diagonal color gradient (white to brown)
    white_to_brown = [linspace(1, 0.6, n)', linspace(1, 0.3, n)', linspace(1, 0, n)'];

    % Combine both gradients into one colormap
    cmap = [white_to_blue];
end




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
