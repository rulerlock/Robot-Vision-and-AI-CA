clear
clc
close all

%% 图片导入
rng(100) % 设置随机数生成器的种子，确保结果可重复

% 设置训练集和测试集的路径
trainDataDir = 'dataset\train'; % 训练集路径
testDataDir = 'dataset\test';   % 测试集路径

% 创建图像数据存储
trainData = imageDatastore(trainDataDir, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');
testData = imageDatastore(testDataDir, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% 设置训练数据的自定义读取函数，用于添加噪声以使得最终模型适应于噪声下的字符识别
trainData.ReadFcn = @Addnoise;

%% 创建深度卷积网络 设计一个CNN网络
imgsize = [128, 128, 1]; % 网络输入图像的大小
layers_num = 3; % 卷积层个数
filters_num = 32; % 卷积核个数
numClasses = 7; % 分类数
lgraph = Fun_CNN(layers_num, filters_num, imgsize, numClasses); % 构建CNN网络
analyzeNetwork(lgraph); % 分析网络结构

%% 定义学习率和余弦退火
global initialLearningRate totalEpochs % 声明全局变量
initialLearningRate = 0.001; % 初始学习率
totalEpochs = 100; % 总训练周期数

% 自定义余弦退火学习率调度器
function lr = cosineAnnealing(epoch)
    global initialLearningRate totalEpochs
    lr = initialLearningRate * 0.5 * (1 + cos((epoch / totalEpochs) * pi));
end

%% 定义回调函数
% 创建一个全局结构体来存储每次迭代的准确率和损失
global trainingMetrics
trainingMetrics = struct('TrainingAccuracy', [], 'TrainingLoss', [], 'ValidationAccuracy', [], 'ValidationLoss', []);

% 定义记录函数，获取训练和验证准确率、损失
function stop = recordMetrics(info)
    stop = false;
    global trainingMetrics
    if ~isempty(info.TrainingAccuracy)
        % 记录训练集准确率和损失
        trainingMetrics.TrainingAccuracy(end+1) = info.TrainingAccuracy;
        trainingMetrics.TrainingLoss(end+1) = info.TrainingLoss;
        % 记录验证集准确率和损失
        if ~isempty(info.ValidationAccuracy)
            trainingMetrics.ValidationAccuracy(end+1) = info.ValidationAccuracy;
            trainingMetrics.ValidationLoss(end+1) = info.ValidationLoss;
        end
    end
end

%% 网络训练
options = trainingOptions('sgdm', ...
    'InitialLearnRate', initialLearningRate, ...
    'MaxEpochs', totalEpochs, ...
    'MiniBatchSize', 32, ...
    'Verbose', true, ...
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'gpu', ...
    'OutputFcn', @recordMetrics, ...
    'LearnRateSchedule', 'none'); % 使用自定义调度器

% 手动在每个 epoch 更新学习率
for epoch = 1:totalEpochs
    % 更新学习率
    currentLearningRate = cosineAnnealing(epoch);
    disp(['Epoch ', num2str(epoch), ': Learning Rate = ', num2str(currentLearningRate)]);
    % 使用更新后的学习率训练网络
    % 这里替换为你的训练代码，例如：trainNetwork(trainData, lgraph, options)
end

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