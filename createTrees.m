function [ T ] = createTrees( x, y )
%Given inputs x (Nx45 matrix of examples) and y (Nx1 row vector of target
%data) producest a 1x6 vector of decision trees (one for each emotion)

T = [];
attributes = 1:45;

%happy
happy_bts = remap_target(y, 'happy');
T(1) = decision_tree_learning(x, attributes, happy_bts);

%saddness
sad_bts = remap_target(y, 'sadness');
T(2) = decision_tree_learning(x, attributes, sad_bts);

%fear
fear_bts = remap_target(y, 'fear');
T(3) = decision_tree_learning(x, attributes, fear_bts);

%anger
anger_bts = remap_target(y, 'anger');
T(4) = decision_tree_learning(x, attributes, anger_bts);

%disgust
disgust_bts = remap_target(y, 'disgust');
T(5) = decision_tree_learning(x, attributes, disgust_bts);

%surprisde
surprise_bts = remap_target(y, 'surprise');
T(6) = decision_tree_learning(x, attributes, surprise_bts);

end

