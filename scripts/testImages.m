
clc; close all;

%% Loading the Previously trained SVM classifier & Feature Extraction network.
load('featureExtractionNet.mat');
load('classificationSVMs1.mat');

%% Reading the Image from the File Explorer
[fname, path] = uigetfile('*.*', "Select a Image File");
img = imread(strcat(path,fname));
img = imresize(img, [227,227]);

%% Apply Feature Extraction network
featureImg = activations(featureExtractionNet, img, 'fc8', 'OutputAs', 'rows');


%% Predict the category of the Image
prediction = classificationSVM.predict(featureImg);
disp(prediction(1));