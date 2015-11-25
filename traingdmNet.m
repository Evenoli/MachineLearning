function [ net, tr ] = traingdmNet( inputs, targets,lRate, momentum, epochs )
    
 
    net = feedforwardnet(layers, 'traingdm');
    net = configure(net, inputs, targets);
    
    %traingdParams
    net.trainParam.epochs = epochs;
    net.trainParam.lr = lRate;
    net.trainParam.mc = momentum;

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
    
    disp('momentum: ');
    disp(momentum);
    
    disp('Best performance: ');
    disp(best_perf);
    
    %results = sim(net, inputs);
    %predictions = NNout2labels(results);
    %predictions2 = NNout2labels(net(inputs));
    %plot(inputs, targets, inputs, results, 'r.');




end

%gradient descent momentum back prop (traingdm) - learning rate (lr), momentum
%    constant (mc)