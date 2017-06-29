%Read Images
standard = imread('stan.jpg');
standard2=rgb2gray(standard);
%figure;
%imshow(stopImage2);
%title('Image of a Stop Sign');

sceneImage = imread('StopSign4.jpg');
sceneImage2=rgb2gray(sceneImage);
%figure;
%imshow(sceneImage2);
%title('Image of a Cluttered Scene');

%Detect Feature Points
signPoints = detectSURFFeatures(standard2);
scenePoints = detectSURFFeatures(sceneImage2);

% figure;
% imshow(standard2);
% title('100 Strongest Feature Points from Stop Sign Image');
% hold on;
% plot(selectStrongest(signPoints, 100));

% figure;
% imshow(sceneImage2);
% title('300 Strongest Feature Points from Scene Image');
% hold on;
% plot(selectStrongest(scenePoints, 1000));

%Extract Feature Descriptors
[signFeatures, signPoints] = extractFeatures(standard2, signPoints);
[sceneFeatures, scenePoints] = extractFeatures(sceneImage2, scenePoints);

%Find Putative Point Matches
signPairs = matchFeatures(signFeatures, sceneFeatures);

matchedSignPoints = signPoints(signPairs(:, 1), :);
matchedScenePoints = scenePoints(signPairs(:, 2), :);
% figure;
% showMatchedFeatures(standard2, sceneImage2, matchedSignPoints, ...
%     matchedScenePoints, 'montage');
% title('Putatively Matched Points (Including Outliers)');

%Locate the Object in the Scene Using Putative Matches
[tform, inlierSignPoints, inlierScenePoints] = ...
    estimateGeometricTransform(matchedSignPoints, matchedScenePoints, 'affine');


% figure;
% showMatchedFeatures(standard2, sceneImage2, inlierSignPoints, ...
%     inlierScenePoints, 'montage');
% title('Matched Points (Inliers Only)');


%Get the bounding polygon of the reference image
signPolygon = [1, 1;...                           % top-left
        size(standard2, 2), 1;...                 % top-right
        size(standard2, 2), size(standard2, 1);... % bottom-right
        1, size(standard2, 1);...                 % bottom-left
        1, 1];   

    
%Transform the polygon into the coordinate system of the target image
newSignPolygon = transformPointsForward(tform, signPolygon);    


%Display the detected object
figure;
imshow(sceneImage2);
hold on;
line(newSignPolygon(:, 1), newSignPolygon(:, 2), 'Color', 'y');
title('Detected Box');








