% 函数：创建CNN卷积神经网络
function [lgraph] = Fun_CNN(layers_num,filters_num,imgsize,numClasses)

% 输入参数：
% layers_num：卷积层的数量
% filters_num：每个卷积层的卷积核数量
% imgsize：网络输入图像的大小[高度, 宽度, 颜色通道]
% numClasses：分类任务的类别数

% 定义输入层
% 创建一个接受指定大小图像的输入层
inputLayer = imageInputLayer(imgsize(1:3),'Name','input');

% 定义输出层
% 创建一个完全连接的层，然后是一个softmax层和一个分类层
outputLayer = [ fullyConnectedLayer(filters_num*2^layers_num*2,'Name','FCoutput')
fullyConnectedLayer(numClasses,'Name','FCoutput1')
softmaxLayer('Name','softmax')
classificationLayer('Name','classLayer')];

%初始化网络
lgraph = layerGraph;

% 将输入层添加到网络
lgraph = addLayers(lgraph,inputLayer);
inputString = 'input';% 设置卷积层输入的名称

%定义卷积神经网络
for i=1:layers_num
    % 创建多个层：两个卷积层、一个批量归一化层、一个ReLU激活层和一个最大池化层

    convLayer = [
        convolution2dLayer(3,filters_num*2^i,'Padding','same','Name',['conv_' num2str(i) num2str(1)])%卷积层参数
        convolution2dLayer(3,filters_num*2^i,'Padding','same','Name',['conv_' num2str(i) num2str(2)])%卷积层参数
        batchNormalizationLayer('Name',['BN_' num2str(i)])%正则化层
        reluLayer('Name',['relu_' num2str(i)])%激活函数层
        maxPooling2dLayer(2,'Stride',2,'Name',['maxpool_' num2str(i)])
        ];%最大池化层

        % 将这些创建的层添加到网络
        
        lgraph = addLayers(lgraph,convLayer);

        % 连接新添加的卷积层到网络
        outputString = ['conv_' num2str(i) num2str(1)];
        
        % 每一层的输出连接到下一层的输入
        lgraph = connectLayers(lgraph,inputString,outputString);

        % 更新下一层的卷积层输入名称
        inputString = ['maxpool_' num2str(i)];
end

%添加输出层
lgraph = addLayers(lgraph,outputLayer);

% 连接最后一个卷积层的输出到第一个全连接层
lgraph = connectLayers(lgraph,inputString,'FCoutput');
end