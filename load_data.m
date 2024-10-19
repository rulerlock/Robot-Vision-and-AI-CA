function [image_data, image_label] = load_data(which_set)
    % Get the current path of the script
    baseFolder = fileparts(mfilename('fullpath')); 
    labelFolders = {'0', '4', '7', '8', 'A', 'D', 'H'}; 

    % Data is stored in the \data folder
    dataFolder = fullfile(baseFolder, 'dataset', char(which_set));
    
    % Initialize data storage
    X_train = {};  % Used to store images
    Y_train = [];  % Used to store labels
    
    % Iterate over each label folder
    for labelIdx = 1:length(labelFolders)
        labelFolder = labelFolders{labelIdx};
        currentFolder = fullfile(dataFolder, labelFolder); % Construct the current label path
        
        % Get all image files in the current label folder
        imageFiles = dir(fullfile(currentFolder, '*.png')); % Assuming images are in png format
        
        % Loop to read each image in the current label folder
        for i = 1:length(imageFiles)
            % Construct the full file path of the image
            filePath = fullfile(currentFolder, imageFiles(i).name);
            % Read the image
            img = imread(filePath);    
            % Flatten the image into a one-dimensional vector and store in X_train
            imgFlattened = img(:)';
            X_train{end+1} = imgFlattened;  % Add the flattened image to X_train            
            
            % Store the corresponding one-hot label in Y_train
            one_hot_label = zeros(1, length(labelFolders));
            one_hot_label(labelIdx) = 1;
            Y_train = [Y_train; one_hot_label]; % Append the one-hot encoded label
        end
    end
    
    % Convert cell array to matrix form (each row of X_train is a flattened image)
    X_train = cell2mat(X_train'); % Ensure conversion to matrix and arranged as rows
    Y_train = Y_train;  % Labels are already in the desired format
    
    disp(string(which_set) + string(size(Y_train,1)) + ' images and labels successfully loaded');

    

    % Output the image data and labels
    image_data = double(X_train);
    image_label = double(Y_train);
end
