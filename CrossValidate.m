function [ confusion_matrix, avg_metrics ] = CrossValidate( x, y )
% Given examples (x) and labels (y) will return a confusion matrix and a
% struct containing metrics information taken from the average results
% accross the given data.

    CROSS_VALIDATION_NUM = 10;
    
    con_matricies = cell(6, 1);
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
        fold_end = fold_start + fold_sizes(j);
        
    end

end

