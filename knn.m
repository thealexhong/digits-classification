load('train_32x32.mat');
num_points = size(X,4);       %X is (X,Y,RGB,index)
crop = [1 32 1 32];
new_X = zeros((crop(2)-crop(1)+1)*(crop(4)-crop(3)+1), num_points);

fprintf('Start preprocessing data...\n');

for i = 1:num_points;
    
    fprintf('Processing image %i/%i...\n', i, num_points);
    im = X(:,:,:,i);
    im = rgb2gray(im); 
    im = im(crop(1):crop(2),crop(3):crop(4)); % only keep center
    im = im2double(im);
    im_1d = im';
    im_1d = im_1d(:)'; %transform im to 1d array;
    new_X(:,i) = im_1d; %? 576*num_points????????

end
new_X=new_X';

% split_point = round(num_points*0.8);
% seq = randperm(num_points);
X_train = new_X;     %train set data 
Y_train = y;           %train set label

load('test_32x32.mat');
num_test_points = size(X,4);       %X is (X,Y,RGB,index)
crop = [1 32 1 32];
% new_X = zeros((crop(2)-crop(1)+1),(crop(4)-crop(3)+1), num_points);
new_test_X = zeros((crop(2)-crop(1)+1)*(crop(4)-crop(3)+1), num_test_points);

fprintf('Start preprocessing testing data...\n');

for i = 1:num_test_points;
    
    fprintf('Processing image %i/%i...\n', i, num_test_points);
    im = X(:,:,:,i);
    im = rgb2gray(im); 
    im = im(crop(1):crop(2),crop(3):crop(4)); % only keep center
    im = im2double(im);
    im_1d = im';
    im_1d = im_1d(:)'; %transform im to 1d array;
    new_test_X(:,i) = im_1d; %? 576*num_points????????
    
end
new_test_X=new_test_X';

X_test = new_test_X;  %test set data
Y_test = y;        %test set label

train_y = zeros(10,size(Y_train,1));
for i=1:size(Y_train,1);
    train_y(Y_train(i),i) = 1;
end
train_y=train_y';

test_y = zeros(10,size(Y_test,1));
for i=1:size(Y_test,1);
    test_y(Y_test(i),i) = 1;
end
test_y=test_y';
% first, 0-mean data
X_train = bsxfun(@minus, X_train, mean(X_train,1));           
X_test  = bsxfun(@minus, X_test, mean(X_test,1));           

% Compute PCA
covariancex = (X_train'*X_train)./(size(X_train,1)-1);                 
[V D] = eigs(covariancex, 500);   % reduce to 10 dimension




pcatrain = X_train*V;
pcatest = X_test*V;

fprintf('pca done!\n');

%show eigen digit

% eigenDigit = 255*reshape(pcatrain,[size(pcatrain,1),10,10]);




%%
% Save result to file
% save mydata/patchData_8x8.mat patchData -v7.3
%save('patchData_8x8.mat', 'patchData' );

nDim = size(pcatrain,2);
nTrainPoints = size(pcatrain,1);
nClasses = max(Y_train);
% nTestData = size(pcatest,1);
nTestPoints = size(pcatest,1);
% train_targets2 = zeros(nTrainPoints,nClasses);
% for i=1:nTrainPoints,
%     train_targets2(i,Y_train(i)+1) = 1;
% end;
k=5;

Category = zeros(nTestPoints,1);
classifications = zeros(nTestPoints, nClasses);

for i = 1 : nTestPoints,
  fprintf('Working on test image %i/%i...\n', i, nTestPoints);

      % find squared Euclidean distances to all training points
  v1 = pcatrain(:,1:nDim);
  % to make the matrix equal to the raw
  v2 = repmat(pcatest(i,:),[nTrainPoints 1]);
  difference = v1 - v2;
  distances = sum(difference.^2,2);
  [sorted_distances, ind] = sort(distances);
  classamounts = zeros(1, nClasses);
  
  
  % get the label
  for j=1:k,
    classamounts = classamounts + train_y(ind(j),:);
  end;
    % choose find the max class vote
    
  indices = find(classamounts == max(classamounts));
   classifications(i,indices(1)) = 1;
   Category(i) = indices(1);
end;

Categoryresult = Category - Y_test;

correct = sum(Categoryresult(:)==0);

error = size(Y_test,1) - correct;

accuracy = correct / size(Y_test,1);

fprintf('Accuracy is %i\n', accuracy);



% for i=1:nTestData,
%     arrayK = zeros(k,2);
%     fprintf('training data %i\n',i);
% 
%     for j=1:nTrainData,
% %         distance = 0;
% %         for d=1:nDim,
% %             distance = distance + (test_data(i,d)-train_data(j,d))^2;
% %         end
% 
%         distance = sumsqr(pcatest(i,:)-pcatrain(j,:));
%         distance = sqrt(distance);
%     
%         if j<=k
%             arrayK(j,1) = distance;
%             arrayK(j,2) = Y_train(j);
%         else
%             arrayK = sortrows(arrayK);
%             if distance<arrayK(k,1)
%                 arrayK(k,1) = distance;
%                 arrayK(k,2) = Y_train(j);
%             end 
%         end
%     end
%     arrayK = sortrows(arrayK);
% 
%     mat = arrayK(:,2);
%     class = 0;
%     class_count = sum(mat==0);
%     for a = 1:nClasses
%         if sum(mat==a)>class_count
%             class_count = sum(mat==a);
%             class = a;
%         end
%     end
%         
%         
%     classifications(i) = class;
% end
% 
% % result = classifications - Y_test;
% result = classifications - Y_test(1:nTestData,:);
% 
% correct = sum(result(:)==0);
% 
% % error = size(Y_test,1) - correct;
% 
% 
% 
% %classerror = error / size(Y_test,1);
% accuracy = correct / nTestData;