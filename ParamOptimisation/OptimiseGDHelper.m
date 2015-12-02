function [ res ] = OptimiseGDHelper(fold, trainx, trainy, validationx, validationy, num_layers, neurons_per_layer, learningRate, N_EPOCHS )
%OPTIMISEGDHELPER Summary of this function goes here
%   Detailed explanation goes here

resCounter = 1;
res = [];
        
for l = num_layers
    for n = neurons_per_layer
        top = [];
        for k = 1:l
            top(k) = n;
        end
        for lr = learningRate
            %Train net using given params
            [ net, tr ] = traingdNet(top, trainx, trainy, lr, N_EPOCHS);
            %Get predictions from validation data
            predictions = testANN(net, validationx);
            %Create confusion matrix from predictions
            labels = NNout2labels(validationy);                    
            conf_matrix = confusionMatrixNN(predictions, labels);
            disp(conf_matrix);
            disp(size(validationy, 2));
            %Get a struct of performance metrics from conf_matrix
            metrics = calculateAvgMetrics(conf_matrix);
            
            %Add results to per-fold vector
            res{resCounter} = struct('fold', fold, ...
                                    'num_layers', l, ...
                                    'neurons_per_layer', n, ...
                                    'learning_rate', lr, ...
                                    'confusion_matrix', conf_matrix, ...
                                    'Avergae_Classification_Rate', metrics.AvgClassificationRate, ...
                                    'F1_Measure', metrics.F1, ...
                                    'Average_Recall', metrics.Recall, ...
                                    'Avergae_Precision', metrics.Precision);
            
            disp(res{resCounter});        
            resCounter = resCounter + 1;
        end
    end
end


end

