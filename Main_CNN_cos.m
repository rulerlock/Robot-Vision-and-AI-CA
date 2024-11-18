clear
clc
close all

%% Image Import
rng(100) % Set the random seed to ensure reproducible results

% Set paths for training and testing datasets
trainDataDir = 'dataset\train'; % Training dataset path
testDataDir = 'dataset\test';   % Testing dataset path

% Create image datastores
trainData = imageDatastore(trainDataDir, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');
testData = imageDatastore(testDataDir, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Set a custom read function for training data to add noise,
% making the final model robust to noisy character recognition
trainData.ReadFcn = @Addnoise;

%% Create a Deep Convolutional Network
% Design a CNN network
imgsize = [128, 128, 1]; % Input image size for the network
layers_num = 3; % Number of convolutional layers
filters_num = 32; % Number of filters per convolutional layer
numClasses = 7; % Number of classes
lgraph = Fun_CNN(layers_num, filters_num, imgsize, numClasses); % Build the CNN network
analyzeNetwork(lgraph); % Analyze the network structure

%% Define Learning Rate and Cosine Annealing
global initialLearningRate totalEpochs % Declare global variables
initialLearningRate = 0.001; % Initial learning rate
totalEpochs = 100; % Total training epochs

% Custom cosine annealing learning rate scheduler
function lr = cosineAnnealing(epoch)
    global initialLearningRate totalEpochs
    lr = initialLearningRate * 0.5 * (1 + cos((epoch / totalEpochs) * pi));
end

%% Define Callback Functions
% Create a global structure to store accuracy and loss for each iteration
global trainingMetrics
trainingMetrics = struct('TrainingAccuracy', [], 'TrainingLoss', [], 'ValidationAccuracy', [], 'ValidationLoss', []);

% Define a logging function to record training and validation accuracy and loss
function stop = recordMetrics(info)
    stop = false;
    global trainingMetrics
    if ~isempty(info.TrainingAccuracy)
        % Record training accuracy and loss
        trainingMetrics.TrainingAccuracy(end+1) = info.TrainingAccuracy;
        trainingMetrics.TrainingLoss(end+1) = info.TrainingLoss;
        % Record validation accuracy and loss
        if ~isempty(info.ValidationAccuracy)
            trainingMetrics.ValidationAccuracy(end+1) = info.ValidationAccuracy;
            trainingMetrics.ValidationLoss(end+1) = info.ValidationLoss;
        end
    end
end

%% Network Training
options = trainingOptions('sgdm', ...
    'InitialLearnRate', initialLearningRate, ...
    'MaxEpochs', totalEpochs, ...
    'MiniBatchSize', 32, ...
    'Verbose', true, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'gpu', ...
    'OutputFcn', @recordMetrics, ...
    'LearnRateSchedule', 'none'); % Use custom scheduler

% Manually update learning rate in each epoch
for epoch = 1:totalEpochs
    % Update learning rate
    currentLearningRate = cosineAnnealing(epoch);
    disp(['Epoch ', num2str(epoch), ': Learning Rate = ', num2str(currentLearningRate)]);
    % Use the updated learning rate to train the network
    % Replace with your training code, e.g., trainNetwork(trainData, lgraph, options)
end

% Train the network using the training data and specified options
trainedNet = trainNetwork(trainData, lgraph, options);

%% Network Prediction and Evaluation
% Predict classification for the training set
[YPredicted, ~] = classify(trainedNet, trainData);
YValidation = trainData.Labels; % Actual labels
t = YPredicted == YValidation; % Compare predicted and actual labels

% Calculate and display training accuracy
acc = mean(t);
disp(['The training accuracy is ', num2str(100 * acc), '%']);

% Predict classification for the testing set
[YPredicted, probs] = classify(trainedNet, testData);
YValidation = testData.Labels; % Actual labels
t = YPredicted == YValidation; % Compare predicted and actual labels

% Calculate and display testing accuracy
acc = mean(t);
disp(['The testing accuracy is ', num2str(100 * acc), '%']);

% Save the trained model
save('CNNmodel.mat', 'trainedNet');

% Plot confusion matrix to obtain simulation accuracy
plotconfusion(YValidation, YPredicted);

%% Randomly Select Images from the Test Set for Output

% Randomly select some images from the test set
numImages = 20;
idx = randperm(numel(testData.Files), numImages);

% Predict selected images
images = cell(numImages, 1);
predictedLabels = cell(numImages, 1);
for i = 1:numImages
    images{i} = imread(testData.Files{idx(i)});
    predictedLabels{i} = classify(trainedNet, images{i});
end

% Display images and prediction results
figure
for i = 1:numImages
    subplot(4, 5, i)
    imshow(images{i})
    title(char(predictedLabels{i}))
end