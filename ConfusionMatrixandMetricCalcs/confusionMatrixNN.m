function [ confusion_matrix ] = confusionMatrixNN(predictions, labels)
    confusion_matrix = zeros(6);
    for i=1:size(predictions)
        pred = predictions(i);
        actual = labels(i);
        confusion_matrix(actual, pred) = confusion_matrix(actual, pred) + 1;
    end
end

