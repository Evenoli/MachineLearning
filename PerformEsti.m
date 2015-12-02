function [ confusion_matrix, class_metrics, acr ] = PerformEsti( )
%Section VII

load('cleandata_students');
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
    num_train_examples = size(train_valx, 2);

    split_ind = floor(num_train_examples*0.9);
    trainx = train_valx(:, 1:split_ind);
    trainy = train_valy(:, 1:split_ind);

    valx = train_valx(:, split_ind+1:num_train_examples);
    valy = train_valy(:, split_ind+1:num_train_examples);
    
    %topology ranges
     num_layers = [1,2,3];
     neurons_per_layer = [10:20:90];

    %Parameter ranges
     learningRate = [0.01:0.02:0.1];
     lr_increase = [1.05:0.05:1.2];
     lr_decrease =[0.6:0.2:0.9];
     momentum = [0.1:0.2:1];
     delt_increase = [1:0.25:1.5];
     delt_decrease =[0.3:0.15:0.9];

    
    optiParams = cell(1, 4);
    
    %GD
    GDres = {OptimiseGDHelper(j, trainx, trainy, valx, valy,...
                             num_layers, neurons_per_layer, learningRate, N_EPOCHS )};
    optiParams{1} = getOptimalParametersGD(GDres);
                         
    %GDA
    GDAres = {OptimiseGDAHelper(j, trainx, trainy, valx, valy,...
            num_layers, neurons_per_layer, learningRate, lr_increase, lr_decrease, N_EPOCHS )};
    optiParams{2} = getOptimalParametersGDA(GDAres);
        
    %GDM
    GDMres = {OptimiseGDMHelper(j, trainx, trainy, valx, valy,...
            num_layers, neurons_per_layer, learningRate, momentum, N_EPOCHS )};
    optiParams{3} = getOptimalParametersGDM(GDMres);
        
    %RP
    RPres = {OptimiseRPHelper(j, trainx, trainy, valx, valy,...
            num_layers, neurons_per_layer, delt_increase, delt_decrease, N_EPOCHS )};
    optiParams{4} = getOptimalParametersRP(RPres);
    
    %Choose function with best performance
    bestPerf = 0;
    for i = 1:4
        if(bestPerf < optiParams{i}.metrics.F1)
            bestPerf = optiParams{i}.metrics.F1;
            disp(optiParams{i}.metrics.F1);
            optiParam = optiParams{i};
        end
    end
    
    %Train net with optimal params
    top = [optiParam.layers, optiParam.neurons];
    
    switch optiParam.training_func
        case 'GD'
            lr = optiParam.lRate;
            [ net, tr ] = traingdNet(top, trainx, trainy, lr, N_EPOCHS);
        case 'GDA'
            lr = optiParam.lRate;
            lr_inc = optiParam.lr_inc;
            lr_dec = optiParam.lr_dec;
            [ net, tr ] = traingdaNet(top, trainx, trainy, lr, lr_inc, lr_dec, N_EPOCHS);
        case 'GDM'
            lr = optiParam.lRate;
            mom = optiParam.momentum;
            [ net, tr ] = traingdmNet(top, trainx, trainy, lr, mom, N_EPOCHS);
        case 'RP'
            delt_inc = optiParam.delt_inc;
            delt_dec = optiParam.delt_dec;
            [ net, tr ] = trainrpNet(top, trainx, trainy, delt_inc, delt_dec, N_EPOCHS);
    end
    
    %Test with test fold!
    [ predictions ] = testANN( net, testx );
    labels = NNout2labels(testy);                    
    con_matricies{j} = confusionMatrixNN(predictions, labels);
	
    fold_start = fold_end+1;
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

