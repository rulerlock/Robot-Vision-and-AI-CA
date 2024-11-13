%% 加载已训练的模型
load('CNNmodel.mat', 'trainedNet'); % 确保 CNNmodel.mat 文件在工作目录中

% 加载图像数据
[~, seg_img] = Pic_Preprocess_CNN();

% 创建临时文件夹来保存图像
preproPicDir = 'NewImage';
if ~exist(preproPicDir, 'dir')
    mkdir(preproPicDir);
end

% 保存图像到文件夹
for i = 1:length(seg_img)
    imwrite(seg_img{i}, fullfile(preproPicDir, sprintf('img1_%d.png', i-1)));
end

% 使用 imageDatastore 加载图像文件
testData = imageDatastore(preproPicDir, 'FileExtensions', '.png', 'LabelSource', 'none');

% 创建标签数组并将其添加到 testData
labels = categorical(["H", "D", "4", "4", "7", "8", "0", "A", "0", "0"]);
testData.Labels = labels';

%% 对测试集中的图像进行预测和显示
numImages = numel(testData.Files);

% 对测试集中每张图像进行预测
images = cell(numImages, 1);
predictedLabels = cell(numImages, 1);
for i = 1:numImages
    images{i} = imread(testData.Files{i});
    predictedLabels{i} = classify(trainedNet, images{i});
end

% 显示所有测试集图像及其预测结果
figure
for i = 1:numImages
    subplot(ceil(numImages / 5), 5, i) % 将图像分为每行5个
    imshow(images{i})
    title(['Predicted: ' char(predictedLabels{i}) ', Actual: ' char(labels(i))])
end