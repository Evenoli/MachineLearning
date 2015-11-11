
%TESTTREES takes your trained trees (all six) T and the features x2 and produces
%a vector of label predictions. Both x2 and predictions should be in the same 
%format as x, y provided to you. Think how you will combine the six trees to get 
%a single output for a given input sample. Try at least 2 different ways of 
%combining the six trees.

function [ predictions ] = testTrees(T, x2)

predictions = zeros(size(x2,1), 1);

for i=1:size(x2, 1)
    accepted = [0,0,0,0,0,0];
    acceptedIndex = [];
    for j=1:size(T, 2)
        result = predictExample(T{j}, x2(i,:));
        if (result == 1)
            accepted(j) = 1;
            acceptedIndex(end+1)=j;
        end
    end
    for k=1:6
        if (accepted(k)== 1)
            
            %chooses random emotion that it selected as true
            predictions(i) = k;%acceptedIndex(floor(rand*size(acceptedIndex)));
            break;
        else 
            %possibly temp hack...
            predictions(i) = floor(rand*6);
        end
    end
end
end

function [ tp_tn_fp_fn ] = calculateMetrics(confusion_matrix, chosenClass)
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
    f1_measure = 2 * ((precision * recall) / (precistion + recall));
end

function [ avg_classification_rate ] = calculateAvgClassificationRate( tp_tn_fp_fn )
    tp = tp_tn_fp_fn(1);
    tn = tp_tn_fp_fn(2);
    fp = tp_tn_fp_fn(3);
    fn = tp_tn_fp_fn(4);
    avg_classification_rate = (tp + tn)/(tp+tn+fp+fn);
end


% Report average cross validation classification results (for both clean and noisy data):
%  Confusion matrix.
% (Hint: you should get a single 6x6 matrix)
% (Hint: you will be asked to produce confusion matrices in almost all the assignments so
%  you may wish to write a general purpose function for computing a confusion matrix)
%  Average recall and precision rates per class.
% (Hint: you can derive them directly from the previously computed confusion matrix)
%  The F1-measures derived from the recall and precision rates of the previous step.
%  Average classification rate (NOTE: classification rate = 1 – classification error)


% see section 4 for more details