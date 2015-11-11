function [ prediction ] = predictExample( tree, ex )
%Given tree 'tree' and example 'ex' (1x45) will return a 1 or a 0
%corresponding to whether the example appears to match the emotion of the
%given tree.

%will return a 1x2 vector of prediction, num_nodes_visited

    if(tree.op == 'null');
        prediction = tree.class;
    else
        if(ex(1, tree.op))
            prediction = predictExample(tree.kids{2}, ex);
        else
            prediction = predictExample(tree.kids{1}, ex);
        end
    end
end

