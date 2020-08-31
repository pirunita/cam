
train_list = './voc12/train_aug.txt';
voc12_root = 'VOCdevkit/VOC2012/JPEGImages/';
result_root = 'VOCdevkit/VOC2012/EdgesJPG/';

[image1] = textread(train_list, '%s\n');

for i = 1:length(image1)
    i
    image_name = image1{i};
    filename = [voc12_root, image_name, '.jpg'];
    targetFilename = [result_root, image_name, '.jpg'];
    
    imwrite(edge(imgaussfilt(rgb2gray(imread(filename)), 2), 'Canny'), targetFilename);
    
end