I1= imread('D:\footlongmodel2\DSC_0536.jpg');
J1= imread('D:\footlongmodel2\DSC_0570.jpg');
I1 = double(I1);
J1 = double(J1);

I2= rgb2gray(I1); %convert colour images to grayscale
J2= rgb2gray(J1);
% BWI = im2bw(I2);% convert grayscale images to binary
% BWJ = im2bw(J2);
% IMI = imcomplement(BWI); % invert colours, black to white and white to black
% IMJ = imcomplement(BWJ);
% %imshow(IMI);
% %imshow(IMJ);
% BWIremove = bwareaopen(IMI, 5000); % remove objects less that 4500 pixels
% BWJremove = bwareaopen(IMJ, 5000);
% %imshow(BWIremove);
% %imshow(BWJremove);
% [LI, numI] = bwlabel(BWIremove);
% [LJ, numJ] = bwlabel(BWJremove);
% centroidsI = regionprops(LI,'Centroid');
% centroidsJ = regionprops(LJ,'Centroid');
% Create a transformation structure for an affine % transformation.
t_concord = fitgeotrans(I2(600:1800,3400:4000),J2(600:1800,3400:4000),'affine');
% Get the width and height of the orthophoto and perform % the transformation.
info = imfinfo(I1);
registered = imwarp(J2,t_concord);
figure, imshow(registered)