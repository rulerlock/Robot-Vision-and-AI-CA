%% Function: Preprocess Training Set Images
function I=Addnoise(str)

    A = imread(str);

    % Blurring
    % Apply Gaussian blur to the image using the imgaussfilt function
    % Multiply by a random number for randomized blurring
    A = imgaussfilt(A, 5*rand);

    % Add noise
    % Ensure the final model can recognize images with significant noise
    A = imnoise(A, "salt & pepper");

    % Morphological operations
    se = strel('disk', 1);

    % Opening operation: erosion followed by dilation,
    % used to remove small objects or details
    A = imopen(A, se);

    % Closing operation: dilation followed by erosion,
    % used to fill small holes in the image
    A = imclose(A, se);

    I = A; % Output the processed image
end