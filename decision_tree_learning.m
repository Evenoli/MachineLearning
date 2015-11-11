function [ decision_tree ] = decision_tree_learning(examples, attributes, binary_targets)
    num_examples = size(examples, 1);
    decision_tree = struct('op', 'null', 'kids', {{struct(), struct()}}, 'class', 'null');
    if (all(binary_targets == binary_targets(1)))
        decision_tree.class = binary_targets(1);
        decision_tree.kids = [];
    elseif (isempty(attributes))
        decision_tree.class = majority_value(binary_targets);
        decision_tree.kids = [];
    else
        best_attr = choose_best_decision_attribute(examples, attributes, binary_targets);
        decision_tree.op = best_attr;
        new_attributes = attributes(attributes~=best_attr);
        examples_1 = []; examples_0 = [];
        examples_1_i = 1; examples_0_i = 1;
        b_targets_1 = []; b_targets_0 = [];
        for i=1:num_examples;
            if(examples(i, best_attr) == 1)
                examples_1(examples_1_i, :) = examples(i, :);
                b_targets_1 =[b_targets_1; binary_targets(i)];
                examples_1_i = examples_1_i + 1;
            else
                examples_0(examples_0_i, :) = examples(i, :);
                b_targets_0 =[b_targets_0; binary_targets(i)];
                examples_0_i = examples_0_i + 1;
            end
        end
        if(isempty(examples_0))
            leaf = struct('op', 'null', 'kids', [], 'class', 'null');
            leaf.class = majority_value(binary_targets);
            left_node = leaf;
        else
            left_node = decision_tree_learning( examples_0, new_attributes, b_targets_0);
        end
        
        if(isempty(examples_1))
            leaf = struct('op', 'null', 'kids', [], 'class', 'null');
            leaf.class = majority_value(binary_targets);
            right_node = leaf;
        else
            right_node = decision_tree_learning( examples_1, new_attributes, b_targets_1);
        end
        decision_tree.kids = {left_node, right_node};
    end
end

function [ mode ] = majority_value( binary_targets )
%MAJORITY_VALUE returns the mode of the binary_targets
% x counts number of 1s, while y counts 0s
    x = 0;
    y = 0;
    for b = binary_targets
        if (b==1)
            x = x + 1;
        else
            y = y + 1;
        end
    end
    if (x > y)
            mode = 1;
        else
            mode = 0;
    end
end

% Returns index of best attribute from 'attributes'
function [ best_attribute ] = choose_best_decision_attribute(examples,attributes,binary_targets)
%chooses the attribute that results in the thighest information gain
%Gain(attribute) = I(p,n) - Remainder(attribute)
    cur_best_gain = 0;
    num_attributes = size(attributes, 2);
    best_att = attributes(1);
    for i = 1:num_attributes;
        cur_attr = attributes(i);
        cur_gain = gain(examples, cur_attr, binary_targets);
        if(cur_gain > cur_best_gain);
            cur_best_gain = cur_gain;
            best_att = cur_attr;
        end
    end
    best_attribute = best_att;
end

function [ result ] = gain(examples, attribute, binary_targets)
    p = 0;
    n = 0;
    for i=1:size(examples, 1);
        if(examples(i, attribute));
            p = p+1;
        else
            n = n+1;
        end
    end
    result = I(p,n) - Remainder(examples, attribute, binary_targets);
end

%Returns entropy for p positive values and n negative values
function [ result ] = I(p,n)
%I(p,n) = - p/(p+n).log2(p/(p+n)) - n/(p+n).log2(n/(p+n))
    pr = p / (p+n);
    nr = n / (p+n);
    result = - pr*log2(pr) - nr*log2(nr);
end

function [ result ] = Remainder(examples, attribute, binary_targets)
    p0 = 0;
    n0 = 0;
    p1 = 0;
    n1 = 0;
    num_bts = size(binary_targets, 1);
    for i = 1:num_bts;
        if(binary_targets(i, 1));
            if(examples(i, attribute));
                p1 = p1+1;
            else
                p0 = p0+1;
            end
        else
            if(examples(i, attribute));
                n1 = p1+1;
            else
                n0 = n0+1;
            end
        end
    end
    result = Rem(p0, n0, p1, n1);
end

function [ result ] = Rem(p0, n0, p1, n1)
    t = p0 + p1 + n0 + n1;
    result = (p0+n0)/t * I(p0, n0) + (p1+n1)/t * I(p1, n1); 
end