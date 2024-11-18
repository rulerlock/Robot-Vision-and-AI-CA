function [flatten_image, seg_image] = Pic_Preprocess_CNN()
    img=imread("dataset\charact2.jpg");
    img_gray=rgb2gray(img);
    sub_img=imcrop(img_gray,[38.51	196.51	922.98	144.98]);
    
    flatten_image = {};
    seg_image = {};
    img_bin=create(sub_img,10,2,110);
    buffer=img_bin;
    [h,k]=size(img_bin);
    for i=2:h-1
        for j=2:k-1
            if check_bound(img_bin,i,j)
                buffer(i,j)=256;
            end
        end
    end
    width=88.5;
    height=127;
    start_x=23;
    start_y=12;
    for i=1:10
        
        % Get the size of segmented images
        temp = imcrop(img_bin,[start_x+(i-1)*width,start_y,width,height]);
        scaledImage = imresize(temp, 0.75);
        [rows, cols, ~] = size(scaledImage); 

        % Add padding
        pad_top = floor((128 - rows) / 2);
        pad_bottom = 128 - rows - pad_top;
        pad_left = floor((128 - cols) / 2);
        pad_right = 128 - cols - pad_left;
        padded_img = padarray(scaledImage, [pad_top, pad_left], 0, 'pre');
        padded_img = padarray(padded_img, [pad_bottom, pad_right], 0, 'post');
        
        % 读取灰度图像
        % I = imread('text_image.png'); % 请替换为你的图像文件路径
        % I_gray = rgb2gray(I); % 若图像已是灰度图像可忽略此步
        
        % 二值化图像（假设文字为白色，背景为黑色）
        BW = imbinarize(padded_img);
        
        % 检测文字的边缘
        edges = edge(BW, 'Canny');
        
        % 设置填充的像素距离（例如5像素）
        se = strel('disk', 2); % 创建5像素半径的圆形结构元素
        dilated_edges = imdilate(edges, se);
        
        % 将膨胀后的边缘与原始图像叠加
        filled_image = BW | dilated_edges;
        filled_image = mat2gray(filled_image);
        % 显示结果
        % imshow(filled_image);
        % title('文字边缘填充后的图像');
        
        inversed_img = imcomplement(padded_img);
        seg_image{end + 1} = inversed_img;
        flatten_image{end + 1} = inversed_img(:)';
        
        % subplot(2,5,i),imshow(padded_img)
    end

    flatten_image = cell2mat(flatten_image');
    
    function img_avg_fil=avg_filter(img,k_size)
        kernel = ones(k_size, k_size) / k_size ^ 2;
        img_avg_fil = imfilter(img, kernel, 'symmetric');
    end
    
    function h_p_img=high_pass_filter(img,d)
        s=fftshift(fft2(im2double(img)));
        [m,n]=size(img);
        for i= 1:m
            for j=1:n
                distance=sqrt((i-round(m/2))^2+(j-round(n/2))^2);
                if distance<=d
                    h=0;
                else
                    h=1;
                end
                s(i,j)=h*s(i,j);
            end
        end
        h_p_img=real(ifft2(ifftshift(s)));
    end
    
    function img_bin=make_bin(img,threshold_value)
        buffer=img;
        [m,n]=size(img);
        for i=1:m
            for j=1:n
                if img(i,j)<=threshold_value
                    buffer(i,j)=0;
                else 
                    buffer(i,j)=256;
                end
            end
        end
        img_bin=buffer;
    end
      
    function cr=create(sub_img,avg,high,bin)
        sub_img_1=avg_filter(sub_img,avg);
        sub_img_2=mat2gray(high_pass_filter(sub_img_1,high));
        sub_img_3=rescale(sub_img_2,1,256);
        new_img_bin=make_bin(sub_img_3,bin);
        cr=new_img_bin;
        end
    function is_bound=check_bound(img,i,j)
        top=img(i,j-1);
        bottom=img(i,j+1);
        left=img(i-1,j);
        right=img(i+1,j);
        is_bound=((top+bottom+left+right)==0);
    end
end