function [ opti_params ] = getOptimalParametersGDA( results )
%GETOPTIMALPARAMETERS Summary of this function goes here
%   Detailed explanation goes here
best_perf = 0;
average_results = [];
cell = results(1);
inner = cell{1};
num_res = size(inner, 2);
for i=1:num_res
    total_perf = 0;
    for j=1:10
        cell = results(j);
        list = cell{1};
        res_struct = list{i};
        perf = res_struct.performance;
        uar = perf(3);
        total_perf = total_perf + uar;
    end
    
        cell = results(1);
        list = cell{1};
        res_struct = list{i};
        lr = res_struct.learning_rate;
        n = res_struct.neurons_per_layer;
        l = res_struct.num_layers;
        inc = res_struct.lr_inc;
        dec = res_struct.lr_dec;
    average_results{i}.performance = total_perf / 10;
    average_results{i}.lRate = lr;
    average_results{i}.neurons = n;
    average_results{i}.layers = l;
    average_results{i}.lr_inc = inc;
    average_results{i}.lr_dec = dec;
end

for i=1:num_res
    perf = average_results{i}.performance(3);
    if (perf > best_perf)
        opti_params = average_results{i};
    end
end

end



