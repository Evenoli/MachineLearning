function [ binary_targets ] = remap_target( target_vector, emotion )
% This function takes an Nx1 row vector of the labels of each emotion, and 
% a selected emotion, and returns an Nx1 vector of the remapping of the
% target vector to 1s and 0s (i.e., emotions matching the given emotion go
% to 1, and all others go to 0)
    
t = str2emolab(emotion);
binary_targets = zeros(size(target_vector, 1), 1);
for i=1:size(target_vector, 1);
    binary_targets(i) = (t == target_vector(i));
end
end

