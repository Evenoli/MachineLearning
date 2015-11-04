function [ decision_tree ] = decision_tree_learning( examples, attributes, binary_targets )
%DECISION_TREE_LEARNING returns a decision tree for a given target label
% if all examples have the same value of binary_targets
% then return a leaf node with this value
% else if attributes is empty
% then return a leaf node with value = MAJORITY-VALUE(binary_targets)
% else
% best_attribute  CHOOSE-BEST-DECISION-ATTRIBUTE(examples,attributes, binary_targets)
% tree  a new decision tree with root as best_attribute
% for each possible value υi of best_attribute do (note that there are 2 values: 0 and 1)
% add a branch to tree corresponding to best_attibute = υi
% {examplesi , binary_targets i} {elements of examples with best_attribute = υi and the corresponding binary_targetsi }
% if examplesi is empty
% then return a leaf node with value = MAJORITY-VALUE(binary_targets)
% else subtree  DECISION-TREE-LEARNING(examplesi ,attributes-{best_attribute}, binary_targetsi)
% return tree

%must return a MATLAB struct with fields:

% tree.op: label for corresponding node (empty for leaf)

% tree.kids: a cell array containing the subtrees that initiat from
% corresponding node (resulting tree will be niary, the size of the cell
% array must be 1 x 2 where entries contain left and right subtrees
% respoectively) must be empty for leaf i.e tree.kids = []

% tree.class: a label for the leaf node, possible values:
%             0 - 1:the value of the examples (negative-positive, respectively) 
%                   if it is the same for all examples, or with value as it is 
%                   defined by the MAJORITY-VALUE function (in the case attributes is empty).

%             it must be empty for an internal node since the tree returns
%             a label only in the leaf node

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

function [ best_attribute ] = choose_best_decision_attribute(examples,attributes,binary_targets)
%chooses the attribute that results in the thighest information gain
%Gain(attribute) = I(p,n) - Remainder(attribute)
    
end

%Returns entropy for p positive values and n negative values
function [ result ] = I(p,n)
%I(p,n) = - p/(p+n).log2(p/(p+n)) - n/(p+n).log2(n/(p+n))
    pr = p / (p+n);
    nr = n / (p+n);
    result = - pr*log2(pr) - nr*log2(nr); 
end

function [ result ] = Remainder(attribute, binary_targets)
%p0+n0/p+n . I(p0,n0) + p1+n1/p+n . I(p1,n1)
    p0 = 0;
    n0 = 0;
    p1 = 0;
    n1 = 0;
    n = size(attribute, 1);
    for i = 1:n
        if (binary_targets(i) == 0)
            if (attribute(i) == 1)
                p0 = p0 + 1;
            else 
                n0 = n0 + 1;
            end
        else
            if (attribute(i) == 1)
                p1 = p1 + 1;
            else
                n1 = n1 + 1;
            end
        end
    end
    result = (p0+n0/n) * I(p0, n0) + (p1+n1/n) * I(p1, n1);

end