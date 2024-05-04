clc
close all;
clear;

[file,path]=uigetfile({'*.jpg;*.bmp;*.png;*.tif'},'Choose an image');
s=[path,file];
picture=imread(s);
[~,cc]=size(picture);
picture=imresize(picture,[300 500]);

if size(picture,3)==3
  picture=rgb2gray(picture);
end
% se=strel('rectangle',[5,5]);
% a=imerode(picture,se);
% figure,imshow(a);
% b=imdilate(a,se);
threshold = graythresh(picture);
picture =~im2bw(picture,threshold);
picture = bwareaopen(picture,30);
imshow(picture)
if cc>2000
    picture1=bwareaopen(picture,3500);
else
    picture1=bwareaopen(picture,3000);
end
figure,imshow(picture1)
picture2=picture-picture1;
figure,imshow(picture2)
picture2=bwareaopen(picture2,200);
figure,imshow(picture2)

[L,Ne]=bwlabel(picture2);
propied=regionprops(L,'BoundingBox');
hold on
pause(1)
for n=1:size(propied,1)
  rectangle('Position',propied(n).BoundingBox,'EdgeColor','g','LineWidth',2)
end
hold off

figure
final_output=[];
t=[];
for n=1:Ne
  [r,c] = find(L==n);
  n1=picture(min(r):max(r),min(c):max(c));
  n1=imresize(n1,[42,24]);
  imshow(n1)
  pause(0.2)
  filename = sprintf('cropped_image/cropped%d.jpg', n); % Create a filename for each image
  imwrite(n1, filename); % Save the image
end

if ~exist('cropped_image', 'dir')
    mkdir('cropped_image')
end
