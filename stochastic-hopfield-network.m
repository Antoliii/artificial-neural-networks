clear;clc;close all

% constants 
beta = 2;
N = 200;
p = 45;
T = 1000;
experiments = 100;
order_param_average = zeros(experiments,1);

% loop 100 experiments
for experiment = 1:experiments
    
    % matrix with p random patterns
    patterns = randi([0,1],N,p);
    patterns(patterns == 0) = -1;
    
    % create weight matrix with hebbs rule
    weightMatrix = zeros(N,N);
    for pattern = 1:p
        patternWeightMatrix = 1/N*patterns(:,pattern)*patterns(:,pattern)';
        weightMatrix = weightMatrix + patternWeightMatrix;
    end
    % set diagonal to 0
    weightMatrix(1:1+size(weightMatrix,1):end) = 0;
    
    pattern1 = patterns(:,1);
    s_j = patterns(:,1);
    order_param = zeros(T,1);
    
    for update = 1:T
        m = zeros(N,1);
        
        for bit = 1:N
            
            b_m = weightMatrix(bit,:)*s_j;
            prob = 1/(1+exp(-2*beta*b_m));
            
            % stochastic update neuron
            if rand() < prob
                s_i = 1;
            else
                s_i = -1;
            end
            
            m(bit) = s_i*pattern1(bit);
            
            % reset
            s_j(bit) = s_i;
        end
        
        order_param(update) = sum(m)/N;
    end
    
    % sum
    order_param_average(experiment) = sum(order_param)/T;
    disp(sum(order_param)/T);
    
end

result = sum(order_param_average)/experiments;
disp("order parameter average: " + string(result))
