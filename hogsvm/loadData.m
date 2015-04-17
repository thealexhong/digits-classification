function [trainingImages, trainingLabels, ...
          testImages, testAnswers] = loadData
    load('train_32x32.mat');
    trainingImages = X;
    trainingLabels = y;
    load('test_32x32.mat');
    testImages = X;
    testAnswers = y;
end