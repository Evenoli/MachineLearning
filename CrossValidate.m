function [ confusion_matrix, class_metrics, acr, acerr_per_fold ] = CrossValidate( x, y )
% Given examples (x) and labels (y) will return a confusion matrix, and
% array of structs containing metrics for each class, and the average
% classification rate accross the final confusion matrix (acr).

    CROSS_VALIDATION_NUM = 10;
    
    con_matricies = cell(CROSS_VALIDATION_NUM, 1);
    acerr_per_fold = zeros(1, CROSS_VALIDATION_NUM);
    num_examples = size(x, 1);
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
        disp(j);
        
        fold_end = fold_start + fold_sizes(j);
        if(fold_end > num_examples);
            fold_end = num_examples;
        end

        %Test/validation examples
        testx = x(fold_start:fold_end, :);
        valx_2 = x(fold_end+1:num_examples, :);
        
        %Test/validation Labels
        testy = y(fold_start:fold_end, :);
        valy_2 = y(fold_end+1:num_examples, :);
        
        if(j == 1);
            validationx = valx_2;
            validationy = valy_2;
        elseif(j == CROSS_VALIDATION_NUM);
            validationx = x(1:fold_start-1, :);
            validationy = y(1:fold_start-1, :);
        else
            valx_1 = x(1:fold_start-1, :);
            validationx = vertcat(valx_1, valx_2);
            valy_1 = y(1:fold_start-1, :);
            validationy = vertcat(valy_1, valy_2);
        end
        
        Trees = createTrees(validationx, validationy);
        con_matricies{j} = calculateConfusionMatrix(Trees, testx, testy);
        avgMetrics = calculateAvgMetrics(con_matricies{j});
        acerr_per_fold(1, j) = 1 - avgMetrics.AvgClassificationRate;
        fold_start = fold_end+1;
    end
    
    %Sum upp confusion matrices
    con_matrix_total = zeros(6);
    for k = 1:CROSS_VALIDATION_NUM
        con_matrix_total = con_matrix_total + con_matricies{k};
    end
%     confusion_matrix = con_matrix_total/CROSS_VALIDATION_NUM;
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

