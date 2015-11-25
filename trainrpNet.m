function [ net, tr ] = trainrpNet( inputs, targets, delt_inc, delt_dec, epochs )
    
    net = feedforwardnet(layers, 'trainrp');
    net = configure(net, inputs, targets);
    
    %traingdParams
    %net.trainFunc = traingd;
    net.trainParam.epochs = epochs;
    net.trainParam.delt_inc = delt_inc;
    net.trainParam.delt_dec = delt_dec;
    
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
    
    disp('delt_inc: ');
    disp(delt_inc);
    disp('delt_dec: ');
    disp(delt_dec);
    
    disp('Best performance: ');
    disp(best_perf);
    
    %results = sim(net, inputs);
    %predictions = NNout2labels(results);
    %predictions2 = NNout2labels(net(inputs));
    %plot(inputs, targets, inputs, results, 'r.');



end

%resilient backpropagation (trainrp) - Increment/Decrement to weight change
%    (delt_inc/delt_dec).
