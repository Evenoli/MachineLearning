function [ opti_params ] = getOptimalParametersRP( results )
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
        n = res_struct.neurons_per_layer;
        l = res_struct.num_layers;
        delt_i = res_struct.delt_inc;
        delt_d = res_struct.delt_dec;
    average_results{i}.training_func = 'RP';
    average_results{i}.conf_matrix = total_cm;
    average_results{i}.metrics = calculateAvgMetrics(total_cm);
    average_results{i}.neurons = n;
    average_results{i}.layers = l;
    average_results{i}.delt_inc = delt_i;
    average_results{i}.delt_dec = delt_d;
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



