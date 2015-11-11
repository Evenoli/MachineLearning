
%TESTTREES takes your trained trees (all six) T and the features x2 and produces
%a vector of label predictions. Both x2 and predictions should be in the same 
%format as x, y provided to you. Think how you will combine the six trees to get 
%a single output for a given input sample. Try at least 2 different ways of 
%combining the six trees.

function [ predictions ] = testTrees(T, x2)

predictions = zeros(size(x2,1), 1);

for i=1:size(x2, 1)
    accepted = [0,0,0,0,0,0];
    for j=1:size(T, 2)
        result = predictExample(T{j}, x2(i,:));
        if (result == 1)
            accepted(j) = 1;
        end
    end
    for k=1:6
        if (accepted(k)== 1)
            %chooses first emotion that it selected as true
            predictions(i) = k;
            break;
        else 
            %possibly temp hack...
            predictions(i) = 1;
        end
    end
end
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