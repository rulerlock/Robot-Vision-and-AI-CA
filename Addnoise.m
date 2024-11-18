%% 函数：预处理训练集图像
   function I=addnoise(str)

    A= imread(str);

    % 模糊化处理
    % 使用 imgaussfilt 函数对图像应用高斯模糊
    % 乘以随机数以进行随机模糊
    A= imgaussfilt(A,5*rand);

    % 添加 noise
    % 使得最终模型能够识别具有较大噪声的图像
    A = imnoise(A,"salt & pepper");

     % 形态学处理
     se = strel('disk',1);

    % 开运算，先侵蚀后膨胀，用于去除小的物体或细节
     A = imopen(A,se);

    % 闭运算，先膨胀后侵蚀，用于填补图像中的小洞
    A = imclose(A,se);

    I=A; % 输出处理后的图像

   end