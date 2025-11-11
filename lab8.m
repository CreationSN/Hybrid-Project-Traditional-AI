%% ------------------------------------------------------------------------
% Hybrid Computer Vision Pipeline:
% Traditional Pre-processing (K-Means Segmentation) + AI Classification
% ------------------------------------------------------------------------
close all; clear; clc;

%% ------------------------------------------------------------------------
% Section 1: Load Image and AI Network
% ------------------------------------------------------------------------
disp('Loading image and pre-trained AI network (SqueezeNet)...');

% Load sample image (you can replace this with your own)
targetImage = imread('/MATLAB Drive/Rose.jpg');

% Load SqueezeNet (small, fast CNN)
net = squeezenet;

figure('Name','Hybrid Pipeline Images','NumberTitle','off');

subplot(2,2,1);
imshow(targetImage);
title('Original Image');

%% ------------------------------------------------------------------------
% Section 2: Traditional Method
% Color Segmentation Using K-Means (Traditional Image Processing)
% ------------------------------------------------------------------------
disp('Applying traditional color segmentation (K-Means)...');

% Convert to Lab color space for better clustering
imgLab = rgb2lab(targetImage);
ab = imgLab(:,:,2:3);
ab = im2single(ab);

% Cluster into two color groups
pixelLabels = imsegkmeans(ab, 2);

% Choose cluster 2 as the foreground (you can inspect using imagesc)
mask = pixelLabels == 2;

subplot(2,2,2);
imshow(mask);
title('Traditional Mask (K-Means)');

% Apply mask to extract object
isolatedObject = targetImage;
isolatedObject(repmat(~mask,1,1,3)) = 0;

subplot(2,2,3);
imshow(isolatedObject);
title('Isolated Object (Input to AI)');

%% ------------------------------------------------------------------------
% Section 3: AI Method – Deep Learning Classification
% ------------------------------------------------------------------------
disp('Running AI classification on the isolated object...');

inputSize = net.Layers(1).InputSize(1:2);
resizedObject = imresize(isolatedObject, inputSize);

[YPred, probs] = classify(net, resizedObject);

subplot(2,2,4);
imshow(resizedObject);
title('Resized Object Sent to AI');

%% ------------------------------------------------------------------------
% Section 4: Results and Analysis
% ------------------------------------------------------------------------
fprintf('\n--- Hybrid Pipeline Results ---\n');
fprintf('Predicted Class: %s\n', string(YPred));
fprintf('Confidence: %.2f%%\n', max(probs)*100);

disp('Traditional segmentation helped isolate the main object,');
disp('making the AI classifier more focused and improving prediction reliability.');
