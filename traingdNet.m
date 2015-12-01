function [ net, tr ] = traingdNet(layers, inputs, targets, lRate, epochs)
    
    net = feedforwardnet(layers, 'traingd');
    net = configure(net, inputs, targets);
    
    %traingdParams
    %net.trainFunc = traingd;
    net.trainParam.epochs = epochs;
    net.trainParam.lr = lRate;
    net.trainParam.showWindow=0;
    
    %set outlayers activation function to tansig or 'logsig'
    net.layers{size(layers) + 1}.transferFcn = 'tansig';
    
    %division
    net.divideFcn = 'divideind';
    net.divideParam.trainInd = 1;
    net.divideParam.valInd = [];
    net.divideParam.testInd = [];

    %net.trainParam.epochs = epochs;
    [net, tr] = train(net, inputs, targets);
%     best_perf  = tr.best_perf;
    
%     disp('using: ');
%     disp(tr.trainFcn);
%     disp(tr.performFcn);
%     disp('Epochs: ');
%     disp(tr.best_epoch);
%     disp('Network Topology:');
%     disp(layers);
%     disp('learning rate:');
%     disp(lRate);
%     
%     disp('Best performance: ');
%     disp(best_perf);
    
    %results = sim(net, inputs);
    %predictions = NNout2labels(results);
    %predictions2 = NNout2labels(net(inputs));
    %plot(inputs, targets, inputs, results, 'r.');


end

%gradient descent (traingd) - (lr)

