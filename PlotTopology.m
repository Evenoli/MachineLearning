function [ fig ] = PlotTopology( trainFnc, a, b, c, x, y )
% This function takes a training function trainFnc, and the set of
% (optimised) parameters which go with it (a, b, c), along with examples 
% and labels (x, y respectively), and will plot graph of performance 
% (measured in UAR) against number of layers, and neurons per layer.

%IF trainFnc TAKES LESS THE 3 PARAMS, LEAVE OTHERS AS 0
%i.e. PlotTopology(traingd, learning_rate, 0, 0, x, y) 

% We take for first 20% of examples as test data, and the other 80% as
% training data, as we have already optimised the parameters a, b, and c, 
% we wont use validation. 

%Set seed (all nets initialised the same)
RandStream.setGlobalStream(RandStream('mt19937ar','seed',1));

NUM_EPOCHS = 100;
layers_range = [1,2,3];
neurons_range = [10:10:90];

num_examples = size(x, 2);
split_ind = floor(num_examples*0.8);
x1 = x(:, 1:split_ind);
y1 = y(:, 1:split_ind);

x2 = x(:, split_ind+1:num_examples);
y2 = y(:, split_ind+1:num_examples);

performance = zeros(size(layers_range, 2), size(neurons_range, 2));

for l = layers_range
    n_counter = 1;
    for n = neurons_range
        top = [];
        for k = 1:l
            top(k) = n;
        end
        switch trainFnc
            case 'traingda'
                [ net, tr ] = traingdaNet(top, x1, y1, a, b, c, NUM_EPOCHS);
            case 'traingdm'
                [ net, tr ] = traingdmNet(top, x1, y1, a, b, NUM_EPOCHS);
            case 'traingd'
                [ net, tr ] = traingdNet(top, x1, y1, a, NUM_EPOCHS);
            case 'trainrp'
                [ net, tr ] = trainrpNet(top, x1, y1, a, b,  NUM_EPOCHS);
        end
        predictions = testANN(net, x2);
        %Create confusion matrix from predictions
        labels = NNout2labels(y2);                    
        conf_matrix = confusionMatrixNN(predictions, labels);
        %Get a struct of performance metrics from conf_matrix
        metrics = calculateAvgMetrics(conf_matrix);
        performance(l, n_counter) = metrics.Recall;
        n_counter = n_counter + 1;
        
        disp(top);
    end
end

t = sprintf('Graph plotting Average recall against topology for training function %s', trainFnc);
fig = figure;
title(t);
xlabel('Number of hidden layers');
ylabel('Number of neurons per layer');
zlabel('Average Recall Rate');
surf(neurons_range, layers_range, performance);


end