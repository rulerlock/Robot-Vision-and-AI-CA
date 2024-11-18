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

%% 定义学习率和固定阶梯衰减
initialLearningRate = 0.001; % 初始学习率
decayRate = 0.1; % 每次衰减的比例
decayEpochs = 1; % 每隔多少个 epoch 衰减一次

%% 自定义学习率调度器
function lr = stepDecay(epoch)
    lr = initialLearningRate * decayRate ^ floor(epoch / decayEpochs);
end

%% 网络训练
% 网络参数选择
options = trainingOptions('adam', ...  % 使用Adam优化器
    'Plots', 'training-progress', ...  % 在训练时显示进度图
    'InitialLearnRate', 1e-3, ...      % 初始学习率
    'MaxEpochs', 10, ...               % 最大迭代次数
    'VerboseFrequency', 1, ...         % 显示训练进度的频率
    'MiniBatchSize', 64, ...           % 批量大小
    'ExecutionEnvironment', 'gpu', ... % 使用GPU进行训练（如果可用）
    'ValidationData', testData, ...    % 设置验证数据
    'ValidationFrequency', 10, ...     % 验证频率
    'OutputFcn', @recordMetrics);      % 每次迭代后记录指标

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