clc ; clear ; close all ;

%% Loading the Image Data from the folders
imds = imageDatastore("D:\Workspace\MATLAB\Image Classification using SVM\Temporary Datasets\Facemask Resized",...
    'IncludeSubfolders',true,'LabelSource','foldernames');


%% Spliting the dataset into test & validation
[imdsTrain, imdsValidation] = splitEachLabel(imds, 0.8, 'randomized');

%% 
lgraph = layerGraph();

layers = [
    imageInputLayer([227 227 3],"Name","imageinput")
    convolution2dLayer([11 11],96,"Name","conv1","BiasLearnRateFactor",2,"Stride",[4 4])
    reluLayer("Name","relu1")
    crossChannelNormalizationLayer(5,"Name","normalization1","K",1)
    maxPooling2dLayer([3 3],"Name","maxpool1","Stride",[2 2])
    convolution2dLayer([5 5],32,"Name","conv2","BiasLearnRateFactor",2,"Padding","same")
    reluLayer("Name","relu2")
    crossChannelNormalizationLayer(5,"Name","normalization2","K",1)
    maxPooling2dLayer([5 5],"Name","maxpool2","Padding","same")
    convolution2dLayer([3 3],32,"Name","conv3","BiasLearnRateFactor",2,"Padding","same")
    reluLayer("Name","relu3")
    convolution2dLayer([3 3],32,"Name","conv4","BiasLearnRateFactor",2,"Padding","same")
    reluLayer("Name","relu4")
    convolution2dLayer([3 3],32,"Name","conv5","BiasLearnRateFactor",2,"Padding","same")
    reluLayer("Name","relu5")
    maxPooling2dLayer([5 5],"Name","maxpool5","Stride",1,"Padding",[1 1])
    fullyConnectedLayer(4096,"Name","fc6","BiasLearnRateFactor",2)
    reluLayer("Name","relu6")
    dropoutLayer(0.5,"Name","dropout6")
    fullyConnectedLayer(4096,"Name","fc7")
    reluLayer("Name","relu7")
    dropoutLayer(0.5,"Name","dropout7")
    fullyConnectedLayer(1000,"Name","fc8","BiasLearnRateFactor",2)
    softmaxLayer("Name","softmax")
    classificationLayer("Name", "output", "Classes", {'Mask'; 'NoMask'})];
% lgraph = addLayers(lgraph,layers);

% clean up helper variable
% clear tempLayers;

% lgraph = connectLayers(lgraph,"normalization2","maxunpool2/indices");
% lgraph = connectLayers(lgraph,layers);

% FeatureExtractionNet = assembleNetwork(lgraph);

options = trainingOptions('sgdm', ...
    'InitialLearnRate', 2, ...
    'MaxEpochs', 30, ...
    'Shuffle', 'once', ...
    'ValidationData', imdsValidation, ...
    'ValidationFrequency', 10, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

net = trainNetwork(imdsTrain, layers, options);



