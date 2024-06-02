clc ; clearvars;

% Veri seti yolu
dataPath = 'C:\Users\onugu\OneDrive\Masaüstü\AlzheimerSýnýflandýrma\archive';

% Veri setini oku
imageDatastore = imageDatastore(dataPath, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

inputSize = [227 227];
imageDatastore.ReadFcn = @(loc) repmat(imresize(imread(loc), inputSize), [1, 1, 3]);
% Veri setini eðitim v  e test setlerine ayýr
[trainingSet,validationSet, testSet] = splitEachLabel(imageDatastore, 0.7,0.15, 'randomize');


% CNN modeli için uygun hale getir

%targetSize = [224, 224, 1];  % Gri tonlamalý görüntüler için

net = alexnet;
layers = [
    imageInputLayer([227 227 3],"Name","data")
    convolution2dLayer([11 11],96,"Name","conv1","BiasLearnRateFactor",2,"Stride",[4 4])
    reluLayer("Name","relu1")
    crossChannelNormalizationLayer(5,"Name","norm1","K",1)
    maxPooling2dLayer([3 3],"Name","pool1","Stride",[2 2])
    groupedConvolution2dLayer([5 5],128,2,"Name","conv2","BiasLearnRateFactor",2,"Padding",[2 2 2 2])
    reluLayer("Name","relu2")
    crossChannelNormalizationLayer(5,"Name","norm2","K",1)
    maxPooling2dLayer([3 3],"Name","pool2","Stride",[2 2])
    convolution2dLayer([3 3],384,"Name","conv3","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    reluLayer("Name","relu3")
    groupedConvolution2dLayer([3 3],192,2,"Name","conv4","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    reluLayer("Name","relu4")
    groupedConvolution2dLayer([3 3],128,2,"Name","conv5","BiasLearnRateFactor",2,"Padding",[1 1 1 1])
    reluLayer("Name","relu5")
    maxPooling2dLayer([3 3],"Name","pool5","Stride",[2 2])
    fullyConnectedLayer(4096,"Name","fc6","BiasLearnRateFactor",2)
    reluLayer("Name","relu6")
    dropoutLayer(0.5,"Name","drop6")
    fullyConnectedLayer(4096,"Name","fc7","BiasLearnRateFactor",2)
    reluLayer("Name","relu7")
    dropoutLayer(0.5,"Name","drop7")
    fullyConnectedLayer(4,"Name","fc")
    softmaxLayer("Name","prob")
    classificationLayer("Name","output")];
%% Train the network
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( 'RandXReflection',true, 'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainingSet, ...
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),validationSet);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',64, ...
    'MaxEpochs',15, ... 
    'InitialLearnRate',1e-4, ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',50, ...
    'Verbose',true, ...
    'Plots','training-progress');

[trainedNet, traininfo] = trainNetwork(augimdsTrain,layers,options);

%% Classify Validation Images

[YPred,scores] = classify(trainedNet,testSet);
accuracy = mean(YPred == testSet.Labels);
confusionmat(YPred,testSet.Labels)
confusionchart(YPred,testSet.Labels)