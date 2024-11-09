clear all
close all
[filename,filepath]=uigetfile('*.png','输入一个图像');%选择一张图片
str=strcat(filepath,filename);
%待识别图像预处理
a=imread(str);% 读取图片
a=rgb2gray(a);%图像灰度化


figure
imshow(a)%显示图片
a=im2bw(a);%图像二值化
a=padarray(a,[20 20]);

se = strel('disk',2);
a = imopen(a,se);
a = imclose(a,se);

a=im2uint8(~a);
load CNNmodel%导入训练好的CNN模型
a=imresize(a,[128,128]);%模型输入尺寸标准化；
[YPredicted,probs] = classify(trainedNet,a);
title(['The recognition result is ',char(YPredicted)])


