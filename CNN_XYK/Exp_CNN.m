% clear
% clc
% close all % do not clear the work space, need to read the trained network

%% 图片导入
rng(100) % 设置随机数生成器的种子，确保结果可重复

% 设置训练集和测试集的路径
trainDataDir = 'dataset\train'; % 训练集路径
testDataDir = 'dataset\test';   % 测试集路径

% 创建图像数据存储
trainData = imageDatastore(trainDataDir, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');
% testData = imageDatastore(testDataDir, ...
%     'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% 读取task1-7预处理图像的函数
[~, seg_img] = pic_preprocess_CNN();

% 创建临时文件夹来保存图像
preproPicDir = 'CNN_XYK\dataset\image1';
if ~exist(preproPicDir, 'dir')
    mkdir(preproPicDir);
end

% 保存图像到dataset\image1文件夹
for i = 1:length(seg_img)
    imwrite(seg_img{i}, fullfile(preproPicDir, sprintf('img1_%d.png', i-1)));
end

% 使用 imageDatastore 加载图像文件
testData = imageDatastore(preproPicDir, 'FileExtensions', '.png', 'LabelSource', 'none');

% 创建标签数组
labels = ["H", "D", "4", "4", "7", "8", "0", "A", "0", "0"];

% 确保标签数组的长度与图像数量一致
if numel(labels) ~= numel(testData.Files)
    error('Number of labels not match with images, clear dataset\image1 folder');
end

% 将标签添加到 imageDatastore
testData.Labels = labels';


% 设置训练数据的自定义读取函数，用于添加噪声以使得最终模型适应于噪声下的字符识别
trainData.ReadFcn = @Addnoise;

%% 创建深度卷积网络 设计一个CNN网络
imgsize = [128, 128, 1]; % 网络输入图像的大小
layers_num = 3; % 卷积层个数
filters_num = 32; % 卷积核个数
numClasses = 7; % 分类数
lgraph = Fun_CNN(layers_num, filters_num, imgsize, numClasses); % 构建CNN网络
analyzeNetwork(lgraph); % 分析网络结构    

%% 网络训练
% 网络参数选择
options = trainingOptions('adam', ...  % 使用Adam优化器
    'Plots', 'training-progress', ...  % 在训练时显示进度图
    'InitialLearnRate', 1e-4, ...      % 初始学习率
    'MaxEpochs', 20, ...               % 最大迭代次数
    'VerboseFrequency', 1, ...         % 显示训练进度的频率
    'MiniBatchSize', 64, ...           % 批量大小
    'ExecutionEnvironment', 'gpu', ... % 使用GPU进行训练（如果可用）
    'ValidationData', testData, ...    % 设置验证数据
    'ValidationFrequency', 10);        % 验证频率

% 使用训练数据和指定的选项训练网络
trainedNet = trainNetwork(trainData, lgraph, options);

%% 网络预测与评估
% 对训练集进行分类预测
[YPredicted, ~] = classify(trainedNet, trainData);
YValidation = trainData.Labels; % 实际标签
t = YPredicted == YValidation; % 比较预测和实际标签

% 计算并显示训练精度
acc = mean(t);
disp(['The training accuracy is ', num2str(100 * acc), '%']);

% 对测试集进行分类预测
[YPredicted, probs] = classify(trainedNet, testData);
YValidation = testData.Labels; % 实际标签
t = YPredicted == YValidation; % 比较预测和实际标签

% 计算并显示测试精度
acc = mean(t);
disp(['The testing accuracy is ', num2str(100 * acc), '%']);

% 保存训练好的模型
save('CNNmodel.mat', 'trainedNet');

% 绘制混淆矩阵，得到仿真精度
plotconfusion(YValidation, YPredicted);

%% 随机选择测试集中的图像输出测试结果

 % 随机选择测试集中的一些图像
numImages = 20;
idx = randperm(numel(testData.Files), numImages);

% 对选定的图像进行预测
images = cell(numImages, 1);
predictedLabels = cell(numImages, 1);
for i = 1:numImages
    images{i} = imread(testData.Files{idx(i)});
    predictedLabels{i} = classify(trainedNet, images{i});
end

% 显示图像和预测结果
figure
for i = 1:numImages
    subplot(4, 5, i)
    imshow(images{i})
    title(char(predictedLabels{i}))
end