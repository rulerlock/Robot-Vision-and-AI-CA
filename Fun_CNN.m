% ����������CNN���������
function [lgraph] = Fun_CNN(layers_num,filters_num,imgsize,numClasses)

% ���������
% layers_num������������
% filters_num��ÿ�������ľ��������
% imgsize����������ͼ��Ĵ�С[�߶�, ���, ��ɫͨ��]
% numClasses����������������

% ���������
% ����һ������ָ����Сͼ��������
inputLayer = imageInputLayer(imgsize(1:3),'Name','input');

% ���������
% ����һ����ȫ���ӵĲ㣬Ȼ����һ��softmax���һ�������
outputLayer = [ fullyConnectedLayer(filters_num*2^layers_num*2,'Name','FCoutput')
fullyConnectedLayer(numClasses,'Name','FCoutput1')
softmaxLayer('Name','softmax')
classificationLayer('Name','classLayer')];

%��ʼ������
lgraph = layerGraph;

% ���������ӵ�����
lgraph = addLayers(lgraph,inputLayer);
inputString = 'input';% ���þ�������������

%������������
for i=1:layers_num
    % ��������㣺��������㡢һ��������һ���㡢һ��ReLU������һ�����ػ���

    convLayer = [
        convolution2dLayer(3,filters_num*2^i,'Padding','same','Name',['conv_' num2str(i) num2str(1)])%��������
        convolution2dLayer(3,filters_num*2^i,'Padding','same','Name',['conv_' num2str(i) num2str(2)])%��������
        batchNormalizationLayer('Name',['BN_' num2str(i)])%���򻯲�
        reluLayer('Name',['relu_' num2str(i)])%�������
        maxPooling2dLayer(2,'Stride',2,'Name',['maxpool_' num2str(i)])
        ];%���ػ���

        % ����Щ�����Ĳ���ӵ�����
        
        lgraph = addLayers(lgraph,convLayer);

        % ��������ӵľ���㵽����
        outputString = ['conv_' num2str(i) num2str(1)];
        
        % ÿһ���������ӵ���һ�������
        lgraph = connectLayers(lgraph,inputString,outputString);

        % ������һ��ľ������������
        inputString = ['maxpool_' num2str(i)];
end

%��������
lgraph = addLayers(lgraph,outputLayer);

% �������һ���������������һ��ȫ���Ӳ�
lgraph = connectLayers(lgraph,inputString,'FCoutput');
end