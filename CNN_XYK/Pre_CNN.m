clear all
close all
[filename,filepath]=uigetfile('*.png','����һ��ͼ��');%ѡ��һ��ͼƬ
str=strcat(filepath,filename);
%��ʶ��ͼ��Ԥ����
a=imread(str);% ��ȡͼƬ
a=rgb2gray(a);%ͼ��ҶȻ�


figure
imshow(a)%��ʾͼƬ
a=im2bw(a);%ͼ���ֵ��
a=padarray(a,[20 20]);

se = strel('disk',2);
a = imopen(a,se);
a = imclose(a,se);

a=im2uint8(~a);
load CNNmodel%����ѵ���õ�CNNģ��
a=imresize(a,[128,128]);%ģ������ߴ��׼����
[YPredicted,probs] = classify(trainedNet,a);
title(['The recognition result is ',char(YPredicted)])


