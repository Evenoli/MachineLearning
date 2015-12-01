function [ net, tr ] = traingdaNet( inputs, targets, lRate, lr_inc, lr_dec, epochs )
    
    net = feedforwardnet(layers, 'traingda');
    net = configure(net, inputs, targets);
    
    %traingdParams
    %net.trainFunc = traingd;
    net.trainParam.epochs = epochs;
    net.trainParam.lr = lRate;
    net.trainParam.lr_inc = lr_inc;
    net.trainParam.lr_dec = lr_dec;
    
    net.trainParam.showWindow=0;
    
    %set outlayers activation function to tansig or 'logsig'
    net.layers{size(layers) + 1}.transferFcn = 'tansig';
        
    %division
    net.divideFcn = 'divideint';
    net.divideParam.trainRatio = 0.9;
    net.divideParam.valRatio = 0.1;
    net.divideParam.testRatio = 0;
    

    %net.trainParam.epochs = epochs;
    [net, tr] = train(net, inputs, targets);
%     best_perf  = tr.best_perf;
%     
%     disp('using: ');
%     disp(tr.trainFcn);
%     disp(tr.performFcn);
%     disp('Epochs: ');
%     disp(tr.best_epoch);
%     disp('Network Topology:');
%     disp(layers);
%     
%     disp('learning rate:');
%     disp(lRate);
%     
%     disp('lr_inc:');
%     disp(lr_inc);
%     disp('lr_dec:');
%     disp(lr_dec);
%     
%     disp('Best performance: ');
%     disp(best_perf);
    
    %results = sim(net, inputs);
    %predictions = NNout2labels(results);
    %predictions2 = NNout2labels(net(inputs));
    %plot(inputs, targets, inputs, results, 'r.');


end

%gradient descent adaptive(traingda) -  learning rate (lr), 
%    ratio increase/decrease learning rate (lr_inc, lr_dec)