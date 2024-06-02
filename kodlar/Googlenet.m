clc ; clearvars;
% Veri seti yolu
dataPath = 'C:\Users\onugu\OneDrive\Masaüstü\AlzheimerSınıflandırma\archive';

% Veri setini oku
imageDatastore = imageDatastore(dataPath, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

inputSize = [224 224];
%imageDatastore.ReadFcn = @(loc)imresize(imread(loc),inputSize);
imageDatastore.ReadFcn = @(loc) repmat(imresize(imread(loc), [224, 224]), [1, 1, 3]);

% Veri setini eğitim ve test setlerine ayır
[trainingSet,validationSet, testSet] = splitEachLabel(imageDatastore, 0.7,0.15, 'randomize');


% CNN modeli için uygun hale getir

%targetSize = [224, 224, 1];  % Gri tonlamalı görüntüler için

net = googlenet;
lgraph = layerGraph(net);
net.Layers(1)
inputSize = net.Layers(1).InputSize;

% Replacing last three layers for transfer learning / retraining

lgraph = removeLayers(lgraph, {'loss3-classifier','prob','output'});

numClasses = numel(categories(trainingSet.Labels));
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',12,'BiasLearnRateFactor',1)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);

lgraph = connectLayers(lgraph,'pool5-drop_7x7_s1','fc');

%% Train the network
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( 'RandXReflection',true, 'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),trainingSet, ...
    'DataAugmentation',imageAugmenter);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),validationSet);
%augimdsTest = augmentedImageDatastore(inputSize(1:2),testSet,'DataAugmentation',imageAugmenter,'ColorProcessing' );

options = trainingOptions('sgdm', ...
    'MiniBatchSize',64, ...
    'MaxEpochs',15, ... 
    'InitialLearnRate',1e-4, ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',50, ...
    'Verbose',true, ...
    'Plots','training-progress');

[trainedNet, traininfo] = trainNetwork(augimdsTrain,lgraph,options);

%% Classify Validation Images

[YPred,scores] = classify(trainedNet,testSet);
accuracy = mean(YPred == testSet.Labels);
confusionmat(YPred,testSet.Labels)
confusionchart(YPred,testSet.Labels)