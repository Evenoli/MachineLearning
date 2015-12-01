function [ opti_params ] = OptimiseNNParamsGD( x, y )
%This function re-uses a lot of code from our 'decision tree' cross
%validation method, and hence splits the train/validation data in the same
%way as it did when testing our decision trees.
%Builds nueral networks uning "Gradient descent backpropagation (traingd) 
%Parameter: learning rate (lr)

    %Set seed (all nets initialised the same)
    RandStream.setGlobalStream(RandStream('mt19937ar','seed',1));
    
    %Create structures for saving results
    Results = [[],[],[],[],[],[],[],[],[],[]];

    load('cleandata_students');
    [x2, y2] = ANNdata(x, y);

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

        %train/validation examples
        validationx = x2(:, fold_start:fold_end);
        trainx_2 = x2(:, fold_end+1:num_examples);
        
        %validation Labels
        validationy = y2(:, fold_start:fold_end);
        trainy_2 = y2(:, fold_end+1:num_examples);
        
        if(j == 1);
            trainx = trainx_2;
            trainy = trainy_2;
        elseif(j == CROSS_VALIDATION_NUM);
            trainx = x2(:, 1:fold_start-1);
            trainy = y2(:, 1:fold_start-1);
        else
            valx_1 = x2(:, 1:fold_start-1);
            trainx = horzcat(valx_1, trainx_2);
            valy_1 = y(:, 1:fold_start-1);
            trainy = horzcat(valy_1, trainy_2);
        end
        
        %TRAIN AND TEST
        %topology ranges
        num_layers = [1,2,3];
        neurons_per_layer = [10:10:90];
        
        %Parameter ranges
        learningRate = [0.01:0.01:0.1];
        
        perFoldRes = [];
        resCounter = 1;
        
        for l = num_layers
            for n = neurons_per_layer
                top = [];
                for k = 1:l
                    top(k) = n;
                end
                for lr = learningRate
                    %Train net using given params
                    [ net, tr ] = traingdNet(top, x2, y2, lr, N_EPOCHS);
                    %Get predictions from validation data
                    predictions = testANN(net, validationx);
                    %Create confusion matrix from predictions
                    conf_matrix = confusionMatrixNN(predictions, validationy);
                    %Get a struct of performance metrics from conf_matrix
                    metrics = calculateMetrics(conf_matrix);
                    %Take chosen performance metric
                    perf = [metrics.AvgClassificationRate, metrics.F1];
                    
                    %Add results to per-fold vector
                    perFoldRes(resCounter) = struct('num_layers', l, ...
                                                    'neurons_per_layer', n, ...
                                                    'learning_rate', lr, ...
                                                    'performance', perf);
                    resCounter = resCounter + 1;
                end
            end
        end
        
        Results(j) = perFoldRes;
        fold_start = fold_end+1;
    end

    opti_params = Results;
end

