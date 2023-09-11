trainingData = imageDatastore('Dataset/Training', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
validationData = imageDatastore('Dataset/Validation', 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% AlexNet modelini oluşturmak
net = alexnet;
analyzeNetwork(net);

inputLayer = net.Layers(1);
featureLayer = net.Layers(end-2);
classificationLayer = net.Layers(end);

inputSize = inputLayer.InputSize;

layerGraphNet = layerGraph(net);

numClasses = numel(categories(trainingData.Labels));

newFeatureLayer = fullyConnectedLayer(numClasses, ...
    'Name', 'fc8', ...
    'WeightLearnRateFactor', 10, ...
    'BiasLearnRateFactor', 10);
newClassificationLayer = classificationLayer('Name', 'classification');

layerGraphNet = replaceLayer(layerGraphNet, featureLayer.Name, newFeatureLayer);
layerGraphNet = replaceLayer(layerGraphNet, classificationLayer.Name, newClassificationLayer);

analyzeNetwork(layerGraphNet)

resizedTrainingData = augmentedImageDatastore(inputSize(1:2), trainingData);
resizedValidationData = augmentedImageDatastore(inputSize(1:2), validationData);

miniBatchSize = 250;
validationFrequency = floor(numel(resizedTrainingData.Files) / miniBatchSize);

trainingOptions = trainingOptions('sgdm',...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs', 6,...
    'InitialLearnRate', 3e-4,...
    'Shuffle', 'every-epoch', ...
    'ValidationData', resizedValidationData, ...
    'ValidationFrequency', validationFrequency, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

net = trainNetwork(resizedTrainingData, layerGraphNet, trainingOptions);