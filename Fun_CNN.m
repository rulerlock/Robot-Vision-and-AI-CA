% Function: Create a CNN Convolutional Neural Network
function [lgraph] = Fun_CNN(layers_num, filters_num, imgsize, numClasses)

% Input parameters:
% layers_num: Number of convolutional layers
% filters_num: Number of filters in each convolutional layer
% imgsize: Input image size for the network [height, width, color channels]
% numClasses: Number of categories for the classification task

% Define input layer
% Create an input layer to accept images of the specified size
inputLayer = imageInputLayer(imgsize(1:3), 'Name', 'input');

% Define output layer
% Create a fully connected layer, followed by a softmax layer and a classification layer
outputLayer = [
    fullyConnectedLayer(filters_num * 2^layers_num * 2, 'Name', 'FCoutput')
    fullyConnectedLayer(numClasses, 'Name', 'FCoutput1')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'classLayer')];

% Initialize the network
lgraph = layerGraph;

% Add the input layer to the network
lgraph = addLayers(lgraph, inputLayer);
inputString = 'input'; % Set the input name for the convolutional layers

% Define the convolutional neural network
for i = 1:layers_num
    % Create multiple layers: two convolutional layers, one batch normalization layer, one ReLU activation layer, and one max-pooling layer

    convLayer = [
        convolution2dLayer(3, filters_num * 2^i, 'Padding', 'same', 'Name', ['conv_' num2str(i) num2str(1)]) % Convolution layer parameters
        convolution2dLayer(3, filters_num * 2^i, 'Padding', 'same', 'Name', ['conv_' num2str(i) num2str(2)]) % Convolution layer parameters
        batchNormalizationLayer('Name', ['BN_' num2str(i)]) % Regularization layer
        reluLayer('Name', ['relu_' num2str(i)]) % Activation function layer
        maxPooling2dLayer(2, 'Stride', 2, 'Name', ['maxpool_' num2str(i)]) % Max pooling layer
        ];

    % Add these created layers to the network
    lgraph = addLayers(lgraph, convLayer);

    % Connect the newly added convolutional layers to the network
    outputString = ['conv_' num2str(i) num2str(1)];

    % Connect the output of each layer to the input of the next layer
    lgraph = connectLayers(lgraph, inputString, outputString);

    % Update the input name for the next convolutional layer
    inputString = ['maxpool_' num2str(i)];
end

% Add the output layer
lgraph = addLayers(lgraph, outputLayer);

% Connect the output of the last convolutional layer to the first fully connected layer
lgraph = connectLayers(lgraph, inputString, 'FCoutput');
end