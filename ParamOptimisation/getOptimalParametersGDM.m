function [ opti_params ] = getOptimalParametersGDM( results )
%GETOPTIMALPARAMETERS Summary of this function goes here
%   Detailed explanation goes here
best_perf = 0;
average_results = [];
cell = results(1);
inner = cell{1};
num_res = size(inner, 2);
num_folds = size(results, 2);
for i=1:num_res
    total_cm = zeros(6);
    for j=1:num_folds
        cell = results(j);
        list = cell{1};
        res_struct = list{i};
        cm = res_struct.confusion_matrix;
        total_cm = total_cm + cm;
    end
    
        cell = results(1);
        list = cell{1};
        res_struct = list{i};
        lr = res_struct.learning_rate;
        n = res_struct.neurons_per_layer;
        l = res_struct.num_layers;
        mom = res_struct.momentum;
    average_results{i}.training_func = 'GDM';
    average_results{i}.conf_matrix = total_cm;
    average_results{i}.metrics = calculateAvgMetrics(total_cm);
    average_results{i}.lRate = lr;
    average_results{i}.neurons = n;
    average_results{i}.layers = l;
    average_results{i}.momentum = mom;
end

opti_params = average_results{1};
for i=1:num_res
    perf = average_results{i}.metrics.Recall;
    if (perf > best_perf)
        opti_params = average_results{i};
        best_perf = perf;
    end
end

end

