function [ net, tr ] = traingdNet(layers, inputs, targets, lRate, epochs)
    
    net = feedforwardnet(layers, 'traingd');
    net = configure(net, inputs, targets);
    
    %traingdParams
    %net.trainFunc = traingd;
    net.trainParam.epochs = epochs;
    net.trainParam.lr = lRate;
    
    %division
    net.divideFcn = 'divideblock';
    net.divideParam.trainRatio = 0.8;
    net.divideParam.testRatio = 0.1;
    net.divideParam.valRatio = 0.1;

    %net.trainParam.epochs = epochs;
    [net, tr] = train(net, inputs, targets);
    best_perf  = tr.best_perf;
    
    disp('using: ');
    disp(tr.trainFcn);
    disp(tr.performFcn);
    disp('Epochs: ');
    disp(tr.best_epoch);
    disp('Network Topology:');
    disp(layers);
    disp('learning rate:');
    disp(lRate);
    
    disp('Best performance: ');
    disp(best_perf);
    
    %results = sim(net, inputs);
    %predictions = NNout2labels(results);
    %predictions2 = NNout2labels(net(inputs));
    %plot(inputs, targets, inputs, results, 'r.');


end

%gradient descent (traingd) - (lr)

