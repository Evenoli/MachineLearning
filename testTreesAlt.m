function [ predictions ] = testTreesAlt(T, x2)

predictions = zeros(size(x2,1), 1);

for i=1:size(x2, 1)
    %accepted = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]];
    accepted = zeros(6,2);
    max_nodes = 0;
    min_nodes = 1000000000000;
    for j=1:size(T, 2)
        [result, nodes] = predictExampleAlt(T{j}, x2(i,:));
        
        accepted(j,1) = result;
        accepted(j,2) = nodes;
        
    end
    best_mult_opt = 0;
    best_no_opt = 0;
    accept = false;
    for k=1:6
        if (accepted(k,1) == 1)
            accept = true;
            if(accepted(k,2) < min_nodes)
                best_mult_opt = k;
            end
        else
            if(accepted(k,2) > max_nodes)
                best_no_opt = k;
            end
        end
    end
    
    if (accept)
        predictions(i) = best_mult_opt;
    else
        predictions(i) = best_no_opt;
    end
    
end
end
