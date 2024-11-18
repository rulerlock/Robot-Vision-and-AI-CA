%% Use trained CNN network to classify each character in Image1

%Load the trained model
load('trained_network\CNN1113.mat', 'trainedNet'); % Ensure the trained network file is in the working directory

% Load image data
[~, seg_img] = Pic_Preprocess_CNN();

% Create a temporary folder to save the images
preproPicDir = 'NewImage';
if ~exist(preproPicDir, 'dir')
    mkdir(preproPicDir);
end

% Save images to the folder
for i = 1:length(seg_img)
    imwrite(seg_img{i}, fullfile(preproPicDir, sprintf('img1_%d.png', i-1)));
end

% Use imageDatastore to load image files
testData = imageDatastore(preproPicDir, 'FileExtensions', '.png', 'LabelSource', 'none');

% Create a label array and add it to testData
labels = categorical(["H", "D", "4", "4", "7", "8", "0", "A", "0", "0"]);
testData.Labels = labels';

%% Predict and display images in the test set
numImages = numel(testData.Files);

% Predict each image in the test set
images = cell(numImages, 1);
predictedLabels = cell(numImages, 1);
for i = 1:numImages
    images{i} = imread(testData.Files{i});
    predictedLabels{i} = classify(trainedNet, images{i});
end

% Display all test images and their predictions
figure
for i = 1:numImages
    subplot(ceil(numImages / 5), 5, i) % Divide images into rows of 5
    imshow(images{i})
    title(['Predicted: ' char(predictedLabels{i}) ', Actual: ' char(labels(i))])
end