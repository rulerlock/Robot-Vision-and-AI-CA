% Original dataset path
originalDataSetDir = 'dataset';
% New dataset path
rotatedDataSetDir = 'dataset_Rotation';

% Copy the entire folder to the new path
if ~exist(rotatedDataSetDir, 'dir')
    copyfile(originalDataSetDir, rotatedDataSetDir);
end

% Set the train folder path
trainDataDir = fullfile(rotatedDataSetDir, 'train');

% Create an image datastore object to access all image files
trainData = imageDatastore(trainDataDir, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Calculate 30% of the total number of images and randomly select images
numTrainImages = numel(trainData.Files);
numRotatedImages = round(0.3 * numTrainImages);
randomIndices = randperm(numTrainImages, numRotatedImages);

% Randomly rotate the selected images
for i = 1:numRotatedImages
    % Read the image file
    imgPath = trainData.Files{randomIndices(i)};
    img = imread(imgPath);
    
    % Randomly generate a rotation angle (clockwise, 0° to 30°)
    rotationAngle = -30 * rand;
    
    % Rotate the image without filling blank areas
    rotatedImg = imrotate(img, rotationAngle, 'bilinear', 'crop');
    
    % Save the rotated image back to its original path, replacing the original image
    imwrite(rotatedImg, imgPath);
end

disp('Image rotation and replacement completed!');