function [ net, tr, bleh ] = trainrpNet( net, inputs, targets, epochs )
    
    net.trainParam.epochs = epochs;
    [net, tr] = trainrp(net, inputs, targets);
    
    bleh = sim(net, inputs);
    plot(inputs, targets, inputs, bleh, 'r.');

end

%resilient backpropagation (trainrp) - Increment/Decrement to weight change
%    (delt_inc/delt_dec).
