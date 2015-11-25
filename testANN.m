function [ results, predictions ] = testANN( net, x2 )

    results = sim(net, x2);
    predictions = NNout2labels(results);
    %predictions = NNout2labels(net(x2));

end

