function [ predictions ] = testANN( net, x2 )

    results = sim(net, x2); %% WHAT DOES IT MEAN IF WE GET ZEROS HERE!?!!?
    predictions = NNout2labels(results);
    
    %predictions = NNout2labels(net(x2));

end

