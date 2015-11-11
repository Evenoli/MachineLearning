function [ confusion_matrix ] = calculateConfusionMatrix(T, examples, labels)
    confusion_matrix = zeros(6);
    predictions = testTreesAlt(T, examples);
    %predictions = testTrees(T, examples);
    if (size(predictions) ~= size(examples))
        disp('amount of predictions and actuals dont match');
    end
    for i=1:size(predictions)
        pred = predictions(i);
        actual = labels(i);
        confusion_matrix(actual, pred) = confusion_matrix(actual, pred) + 1;
    end
end

