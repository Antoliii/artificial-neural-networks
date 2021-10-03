clear;clc;close all

% constants
trials = 10^5;
N = 120;
% nPatterns = [12,24,48,70,100,120];
nPatterns = 12;

% loop over number of patterns list
for nPattern = 1:length(nPatterns)
    
    errors = zeros(trials,1);
    p = nPatterns(nPattern);

    % 10^5 trials
    for i = 1:trials

        % matrix with p random patterns
        patterns = randi([0,1],N,p);
        patterns(patterns == 0) = -1;

        % choose random bit from random pattern
        randPattern = patterns(:,randi([1,p]));
        index = randi([1,N]);
        bit = randPattern(index);

        % create weight matrix using hebbs rule
        weightMatrix = zeros(N,N);
        for pattern = 1:p
            patternWeightMatrix = 1/N*patterns(:,pattern)*patterns(:,pattern)';
            weightMatrix = weightMatrix + patternWeightMatrix;
        end

        % set diagonal to 0
        weightMatrix(1:1+size(weightMatrix,1):end) = 0;

        % update one random neuron
        if weightMatrix(index,:)*randPattern ~= 0
            sPrime = sign(weightMatrix(index,:)*randPattern);
        else
            sPrime = 1;
        end

        % count errors
        if sPrime ~= bit
            errors(i,1) = 1;
        end

    end
    
    % error probability
    disp(sum(errors(:) == 1)/length(errors))
end
