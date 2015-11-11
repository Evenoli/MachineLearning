function [ avgMetrics ] = calculateAvgMetrics( confusion_matrix )
%Returns the average metrics for the whole confusion matrix
    
    num_classes = 6;
    recall = 0;
    precision = 0;
    acr = 0;
    f1 = 0;
    
    for i = 1:num_classes;
        metrics = calculateMetrics(confusion_matrix, i);
        recall = recall + metrics.Recall;
        precision = precision + metrics.Precision;
        acr = acr + metrics.AvgClassificationRate;
        f1 = f1 + metrics.F1;
    end
    
    avgMetrics.Recall = recall/num_classes;
    avgMetrics.Precision = precision/num_classes;
    avgMetrics.AvgClassificationRate = acr/num_classes;
    avgMetrics.F1 = f1/num_classes;

end

