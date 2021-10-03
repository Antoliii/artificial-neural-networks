clear;clc;close all

%% patterns
x = zeros(8, 3);
x(1, :) = [1, 1, 1];
x(2, :) = [-1, 1, -1];
x(3, :) = [1, -1, -1];
x(4, :) = [-1, -1, 1];
x(5, :) = [1, -1, 1];
x(6, :) = [-1, 1, 1];
x(7, :) = [1, 1, -1];
x(8, :) = [-1, -1, -1];

%% variables
trials = 1000;
runs = 10;
batchSizeList = [2, 2, 25, 35];
numberOfNeuronsList = [1, 2, 4, 8];
nOuter = 5000;
nInner = 1000;


%% testing
DklAverage = zeros(length(numberOfNeuronsList), 1);
DklTheoretical = zeros(length(numberOfNeuronsList), 1);
pBoltzmannStorage = zeros(length(numberOfNeuronsList), length(x));

tic
for run = 1:runs
    Dkl = zeros(length(numberOfNeuronsList), 1);
    
    for var = 1:length(numberOfNeuronsList)
        batchSize = batchSizeList(var);
        numberOfNeurons = numberOfNeuronsList(var);
        
        % training
        [w, thetaV, thetaH] = train(x, trials, batchSize, numberOfNeurons);
 
        % feed
        pBoltzmann = zeros(length(x), 1);
        for outer = 1:nOuter
            mu = randi(8);
            v = x(mu,:)';
            % update hidden neurons
            bh = (v'*w)'-thetaH;
            h = stochastic_update(bh);

            for inner = 1:nInner
                % update visible neurons
                bv = w*h-thetaV;
                v = stochastic_update(bv);
                % update hidden neurons
                bh = (v'*w)'-thetaH;
                h = stochastic_update(bh);

                for pattern = 1:length(x)
                    if isequal(v', x(pattern, :))
                        pBoltzmann(pattern) = pBoltzmann(pattern) + 1/(nOuter*nInner);
                    end
                end
            end
            
        % feeding ends here
        end
        fprintf('Boltzmann distributuion for %d neurons:\n', numberOfNeurons)
        disp(pBoltzmann')

        
        % KL. Divergence
        for pattern = 1:length(x)
            if (pattern == 1) || (pattern == 2) || (pattern == 3) || (pattern == 4)
                Dkl(var) = Dkl(var) + 0.25*log(0.25/pBoltzmann(pattern));
            else
                Dkl(var) = Dkl(var) + 0;
            end
        end
        
        pBoltzmannStorage(var,:) = pBoltzmannStorage(var,:) + (1/runs)*pBoltzmann';  
    % end loop over numberOfNeuronsList here
    end
    fprintf('---- Dkl divergences for run number %d ----\n', run)
    disp(Dkl')

    plot(numberOfNeuronsList,Dkl','--o','LineWidth',1); hold on
    
    DklAverage = DklAverage + (1/runs)*Dkl;
end
toc

%% plot
plot(numberOfNeuronsList,DklAverage','-o','LineWidth',4); hold on
title('KL divergence as a function of the number of neurons')
xlabel('Neurons') 
ylabel('KL divergence') 
set(gca,'FontSize',12)
grid on;

%% functions
function layer = stochastic_update(localField)
    layer = zeros(size(localField));
    for neuron = 1:length(localField)
        r = rand;
        if r < probability(localField(neuron))
            layer(neuron) = 1;
        else
            layer(neuron) = -1;
        end
    end
end
    
function prob = probability(localField)
    prob = 1./(1+exp(1)^(-2*localField));
end

function [w, thetaV, thetaH] = train(x, trials, batchSize, numberOfNeurons) 
    % weights and threshholds
    w = randn(3, numberOfNeurons);
    thetaV = zeros(3, 1);
    thetaH = zeros(numberOfNeurons, 1);
    
    % constants
    eta = 0.1;
    k = 100;

    for trial = 1:trials
        deltaW = zeros(3, numberOfNeurons);
        deltaThetaV = zeros(3, 1);
        deltaThetaH = zeros(numberOfNeurons, 1);

        % mini batch
        for mu = 1:batchSize
            muIndex = randi(4);
            v = x(muIndex, :)';
            v0 = v;
            bH = (v'*w)'-thetaH;
            h = stochastic_update(bH);

            % CD-k loop
            for j = 1:k
                % update visible neurons
                bV = w*h-thetaV;
                v = stochastic_update(bV);
                % update hidden neurons
                bH = (v'*w)'-thetaH;
                h = stochastic_update(bH);
            end

            % calculate deltas
            bH0 = (v0'*w)'-thetaH;
            deltaW = deltaW + eta*(v0*tanh(bH0)' - v*tanh(bH)');
            deltaThetaV = deltaThetaV - eta*(v0-v);
            deltaThetaH = deltaThetaH - eta*(tanh(bH0)-tanh(bH));
        end

        % update weights and thetas
        w = w + deltaW;
        thetaV = thetaV + deltaThetaV;
        thetaH = thetaH + deltaThetaH;
    end

end
