function [ metrics ] = calculateMetrics(confusion_matrix, chosenClass)
%Return a struct containing the metrics for the given confusion matrix and 
%the given class

    metrics = struct('Recall', 'null', 'Precision', 'null','AvgClassificationRate', 'null', 'F1', 'null');

    tp_tn_fp_fn = [0,0,0,0];
    for i=1:6
        for j=1:6
            if(j == chosenClass && i == j)
                tp_tn_fp_fn(1) = confusion_matrix(i,j);
            end
            if(j == chosenClass && i ~= j)
                tp_tn_fp_fn(3) = tp_tn_fp_fn(3) + confusion_matrix(i, j);
            end
            if(i == chosenClass && i ~= j)
                tp_tn_fp_fn(4) = tp_tn_fp_fn(4) + confusion_matrix(i, j);
            end
            if(i ~= chosenClass && j ~= chosenClass)
                tp_tn_fp_fn(2) = tp_tn_fp_fn(2) + confusion_matrix(i, j);
            end
        end
    end
    
    metrics.Recall = calculateRecall(tp_tn_fp_fn);
    metrics.Precision = calculatePrecision(tp_tn_fp_fn);
    metrics.AvgClassificationRate = calculateAvgClassificationRate(tp_tn_fp_fn);
    metrics.F1 = calculateF1Measure(metrics.Recall, metrics.Precision);
    
end

function [ recall ] = calculateRecall( tp_tn_fp_fn )
    tp = tp_tn_fp_fn(1);
    fn = tp_tn_fp_fn(4);
    recall = tp/(tp+fn);
end

function [ precision ] = calculatePrecision( tp_tn_fp_fn )
    tp = tp_tn_fp_fn(1);
    fp = tp_tn_fp_fn(3);
    precision = tp/(tp+fp);
end

function [ f1_measure ] = calculateF1Measure( recall, precision )
    f1_measure = 2 * ((precision * recall) / (precision + recall));
end

function [ avg_classification_rate ] = calculateAvgClassificationRate( tp_tn_fp_fn )
    tp = tp_tn_fp_fn(1);
    tn = tp_tn_fp_fn(2);
    fp = tp_tn_fp_fn(3);
    fn = tp_tn_fp_fn(4);
    avg_classification_rate = (tp + tn)/(tp+tn+fp+fn);
end
