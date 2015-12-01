function [ confusion_matrix, class_metrics, acr ] = PerformEstiNoisy( top, trainFnc, param1, param2, param3 )
%Section VII

load('noisydata_students');
[x2, y2] = ANNdata(x, y);

%Set seed (all nets initialised the same)
RandStream.setGlobalStream(RandStream('mt19937ar','seed',1));

CROSS_VALIDATION_NUM = 10;
N_EPOCHS = 100;

con_matricies = cell(CROSS_VALIDATION_NUM, 1);
num_examples = size(x2, 2);
base_fold_size = floor(num_examples/CROSS_VALIDATION_NUM);

%Works out optimal fold sizes. Takes remainder after dividing number of
%examples by fold size, and distributes extra examples evenly into
%other folds.
remainder = rem(num_examples, base_fold_size);
fold_sizes(1:10) = base_fold_size;
i = 1;
while(remainder > 0)
    fold_sizes(i) = fold_sizes(i) + 1;
    remainder = remainder - 1;
    i = i + 1;
end

fold_start = 1;
for j = 1:CROSS_VALIDATION_NUM
    fold_end = fold_start + fold_sizes(j);
    if(fold_end > num_examples);
        fold_end = num_examples;
    end

    %train/test examples
    testx = x2(:, fold_start:fold_end);
    trainx_2 = x2(:, fold_end+1:num_examples);

    %train/test Labels
    testy = y2(:, fold_start:fold_end);
    trainy_2 = y2(:, fold_end+1:num_examples);

    if(j == 1);
        train_valx = trainx_2;
        train_valy = trainy_2;
    elseif(j == CROSS_VALIDATION_NUM);
        train_valx = x2(:, 1:fold_start-1);
        train_valy = y2(:, 1:fold_start-1);
    else
        tstx_1 = x2(:, 1:fold_start-1);
        train_valx = horzcat(tstx_1, trainx_2);
        tsty_1 = y2(:, 1:fold_start-1);
        train_valy = horzcat(tsty_1, trainy_2);
    end

    %Split train_valx into train and validation sets using 90/10% split
    num_examples = size(train_valx, 2);
    split_ind = floor(num_examples*0.9);
    trainx = train_valx(:, 1:split_ind);
    trainy = train_valy(:, 1:split_ind);

    valx = train_valx(:, split_ind+1:num_examples);
    valy = train_valy(:, split_ind+1:num_examples);
    
    %Train net with optimal params
    switch trainFnc
        case 'GD'
            [ net, tr ] = traingdNet(top, trainx, trainy, param1, N_EPOCHS);
        case 'GDA'
            [ net, tr ] = traingdaNet(top, trainx, trainy, param1, param2, param3, N_EPOCHS);
        case 'GDM'
            [ net, tr ] = traingdmNet(top, trainx, trainy, param1, param2, N_EPOCHS);
        case 'RP'
            [ net, tr ] = trainrpNet(top, trainx, trainy, param1, param2, N_EPOCHS);
    end
    
    %Test with test fold!
    [ predictions ] = testANN( net, testx );
    con_matricies{j} = confusionMatrixNN(predictions, testy);
	
end

%Sum upp confusion matrices
con_matrix_total = zeros(6);
for k = 1:CROSS_VALIDATION_NUM
    con_matrix_total = con_matrix_total + con_matricies{k};
end
%confusion_matrix = con_matrix_total/CROSS_VALIDATION_NUM;
confusion_matrix = con_matrix_total;

%Get metrics per class

for j = 1:6;
    class_metrics(j) = calculateMetrics(confusion_matrix, j);
end

%Calculate Average Classification Rate
sum_diag = 0;
for k = 1:6;
    sum_diag = sum_diag + confusion_matrix(k, k);
end

sum_total = sum(sum(confusion_matrix));
acr = sum_diag/sum_total;

end

