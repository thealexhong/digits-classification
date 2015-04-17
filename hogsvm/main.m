clear; clc; close all; fclose all;

% Load data
[trainingImages_, trainingLabels_, ...
      testImages, testLabels] = loadData;
    
% Partition Training set into training set and validation set
numTrainingImages_ = size(trainingImages_, 4);
numTrainingImages = round(numTrainingImages_ * 0.80);
numValidationImages = numTrainingImages_ - numTrainingImages;
trainingImages = trainingImages_(:,:,:,1:numTrainingImages);
trainingLabels = trainingLabels_(1:numTrainingImages);
validationImages = trainingImages_(:,:,:,numTrainingImages + 1:numTrainingImages_, :);
validationLabels = trainingLabels_(numTrainingImages + 1:numTrainingImages_);

img = trainingImages(:,:,:,1);
[hog_4x4, vis4x4] = extractHOGFeatures(img,'CellSize',[4 4]);
cellSize = [4 4];
hogFeatureSize = length(hog_4x4);
trainingFeatures = [];

% Train SVM classifier for each digit
% digits = [char('1'):char('9'),char('0')]; % SVHN labels
% for d = 1:numel(digits)
%     % Pre-allocate trainingFeatures array
%     n = sum((trainingLabels(1:numTrainingImages)) == d);
%     features = zeros(n, hogFeatureSize, 'single');
%     j = 1;
%     % Extract HOG features from each training image.
%     s = [find(trainingLabels(1:numTrainingImages) == d)];
%     for i = 1:size(s,1)
%         img = trainingImages(:,:,:,s(i));
%         img = preProcess(img);
%         features(j,:) = extractHOGFeatures(img,'CellSize',cellSize);
%         j = j + 1;
%     end
%     %for i = 1:numTrainingImages
%     %   img = trainingImages(:,:,:,i);
%     %   img = preProcess(img);
%     %   features(i,:) = extractHOGFeatures(img,'CellSize',cellSize);
%     %end
%     trainingFeatures = [trainingFeatures; features];
%     %svm(d) = svmtrain(features, trainingLabels == d);
% end

%for d = 1:numel(digits)
%    svm(d) = svmtrain(trainingFeatures((numTrainingImages*(d-1)+1):(numTrainingImages*d)), trainingLabels == d);
%end

    for i = 1:numTrainingImages
        img = trainingImages(:,:,:,i);
        img = preProcess(img);
        trainingFeatures(i,:) = extractHOGFeatures(img,'CellSize',cellSize);
        fprintf('Training %d/%d\n',i,numTrainingImages);
    end
    save('training_hog.mat');
    
    fprintf('Creating classifier...');
    classifier = fitcecoc(trainingFeatures, ...
                          trainingLabels);
    save('classifier.mat');
                      
    for i = 1:numValidationImages
        img_v = validationImages(:,:,:,i);
        img_v = preProcess(img_v);
        validationFeatures(i,:) = extractHOGFeatures(img_v, 'CellSize', cellSize);
        fprintf('Testing %d/%d\n',i,numValidationImages);
    end
    save('validation.mat');
    
    predictedLabels = predict(classifier, validationFeatures);
    confMat = confusionmat(validationLabels(1:numValidationImages), predictedLabels);
    helperDisplayConfusionMatrix(confMat);               
    fprintf('Validation Success: %f\n',sum(diag(confMat))/sum(sum(confMat)));
                    
    
%% VALIDATION SET    

%     digit  | 0        1        2        3        4        5        6        7        8        9        
% ---------------------------------------------------------------------------------------------------
% 0      | 0.87     0.02     0.02     0.02     0.01     0.01     0.01     0.01     0.01     0.02     
% 1      | 0.05     0.83     0.02     0.02     0.01     0.01     0.02     0.02     0.02     0.01     
% 2      | 0.06     0.05     0.70     0.02     0.06     0.02     0.02     0.03     0.02     0.02     
% 3      | 0.06     0.03     0.02     0.80     0.01     0.02     0.01     0.02     0.01     0.01     
% 4      | 0.03     0.01     0.07     0.02     0.74     0.06     0.01     0.03     0.02     0.01     
% 5      | 0.05     0.02     0.04     0.03     0.06     0.69     0.01     0.05     0.01     0.03     
% 6      | 0.08     0.05     0.02     0.01     0.02     0.01     0.78     0.00     0.01     0.01     
% 7      | 0.06     0.03     0.08     0.02     0.03     0.07     0.01     0.64     0.03     0.03     
% 8      | 0.05     0.04     0.04     0.01     0.03     0.01     0.01     0.04     0.72     0.05     
% 9      | 0.07     0.02     0.02     0.01     0.01     0.03     0.02     0.03     0.04     0.75     
% Success: 0.772848


%% TEST SET
numTestImages = size(testImages, 4);

for i = 1:numTestImages
        img_t = testImages(:,:,:,i);
        img_t = preProcess(img_t);
        testFeatures(i,:) = extractHOGFeatures(img_t, 'CellSize', cellSize);
        fprintf('Testing %d/%d\n',i,numTestImages);
end
save('test.mat');
predictedLabels2 = predict(classifier, testFeatures);
confMat2 = confusionmat(testLabels(1:numTestImages), predictedLabels2);
helperDisplayConfusionMatrix(confMat2);
fprintf('Test Success: %f\n',sum(diag(confMat2))/sum(sum(confMat2)));



% Run each SVM classifier on test images
%numTestImages = size(testImages, 4);
% for d = 1:numel(digits) 
%     testFeatures = zeros(numTestImages, hogFeatureSize, 'single');
%     for i = 1:numTestImages
%         img = testImages(:,:,:,i);
%         img = preProcess(img);
%         testFeatures(i,:) = extractHOGFeatures(img,'CellSize',cellSize);
%     end
%     for digit = 1:numel(svm)
%         predictedLabels(:,digit,d) = svmclassify(svm(digit), testFeatures);
%     end
% end
% displayTable(predictedLabels);
