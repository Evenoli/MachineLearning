function [ prediction, nodes_visited ] = predictExampleAlt( tree, ex )
%Given tree 'tree' and example 'ex' (1x45) will return a 1 or a 0
%corresponding to whether the example appears to match the emotion of the
%given tree.

%will return a 1x2 vector of prediction, num_nodes_visited

    if(tree.op == 'null');
        prediction = tree.class;
        nodes_visited = 0;
    else
        if(ex(1, tree.op))
            %nodes_visited = nodes_visited_so_far + 1;
            [prediction, nodes_visited_yeh] = predictExampleAlt(tree.kids{2}, ex);
            nodes_visited = nodes_visited_yeh + 1;
        else
            %nodes_visited = nodes_visited_so_far + 1;
            [prediction, nodes_visited_yeh] = predictExampleAlt(tree.kids{1}, ex);
            nodes_visited = nodes_visited_yeh + 1;
        end
    end
end