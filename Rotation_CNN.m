% 原数据集路径
originalDataSetDir = 'dataset';
% 新的数据集路径
rotatedDataSetDir = 'dataset_Rotation';

% 复制整个文件夹到新路径
if ~exist(rotatedDataSetDir, 'dir')
    copyfile(originalDataSetDir, rotatedDataSetDir);
end

% 设置 train 文件夹路径
trainDataDir = fullfile(rotatedDataSetDir, 'train');

% 创建图像数据存储对象以获取所有图像文件
trainData = imageDatastore(trainDataDir, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% 计算 30% 图像数量并随机选择图像
numTrainImages = numel(trainData.Files);
numRotatedImages = round(0.3 * numTrainImages);
randomIndices = randperm(numTrainImages, numRotatedImages);

% 随机旋转选定的图像
for i = 1:numRotatedImages
    % 读取图像文件
    imgPath = trainData.Files{randomIndices(i)};
    img = imread(imgPath);
    
    % 随机生成旋转角度（顺时针 0° 到 30°）
    rotationAngle = -30 * rand;
    
    % 旋转图像，不填充空白区域
    rotatedImg = imrotate(img, rotationAngle, 'bilinear', 'crop');
    
    % 将旋转后的图像保存回原路径，替换原始图像
    imwrite(rotatedImg, imgPath);
end

disp('图像旋转并替换完成！');
