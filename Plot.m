clear
clc

% 假设文件名为 'data.mat'
data = load('Data_1e-2_64_10.mat');

% 查看文件中包含的变量
disp('Loaded variables:');
disp(fieldnames(data));

% 假设文件中有变量名为 'inputData'
inputData = data.inputData;

% 使用变量
disp('Data from inputData:');
disp(inputData);
