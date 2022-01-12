clc, clear, close all
rng(200);

%% Učitavanje ulaznih podataka
input = readmatrix('data/Genres.csv');
input = input(:, 1:11)';
output = findgroups(readtable('data/Genres.csv').genre)';

%% Iscrtavanje izlaznih klasa
figure
histogram(output)
title('Žanrovi')
text([1, 2, 3], [1, 2, 3], ["Pop", "Rap", "RnB"], 'horizontalalignment', 'center', 'verticalalignment', 'bottom')

%% Izdvajanje trening i test skupova
rng(200);

N = length(output);

outputOH = zeros(N, 3);
outputOH(output == 1, 1) = 1;
outputOH(output == 2, 2) = 1;
outputOH(output == 3, 3) = 1;
outputOH = outputOH';

Pop = input(:, output == 1);
Rap = input(:, output == 2);
RnB = input(:, output == 3);
outputPop = outputOH(:, output == 1);
outputRap = outputOH(:, output == 2);
outputRnB = outputOH(:, output == 3);

NPop = length(Pop);
NPopT = floor(0.7*NPop);

NRap = length(Rap);
NRapT = floor(0.7*NRap);

NRnB = length(RnB);
NRnBT = floor(0.7*NRnB);

NT = NPopT + NRapT + NRnBT;

inputPopTraining = Pop(:, 1 : NPopT);
inputPopTest = Pop(:, NPopT + 1 : NPop);
outputPopTraining = outputPop(:, 1 : NPopT);
outputPopTest = outputPop(:, NPopT + 1 : NPop);

inputRapTraining = Rap(:, 1 : NRapT);
inputRapTest = Rap(:, NRapT + 1 : NRap);
outputRapTraining = outputRap(:, 1 : NRapT);
outputRapTest = outputRap(:, NRapT + 1 : NRap);

inputRnBTraining = RnB(:, 1 : NRnBT);
inputRnBTest = RnB(:, NRnBT + 1 : NRnB);
outputRnBTraining = outputRnB(:, 1 : NRnBT);
outputRnBTest = outputRnB(:, NRnBT + 1 : NRnB);


inputTraining = [inputPopTraining, inputRapTraining, inputRnBTraining];
inputTest = [inputPopTest, inputRapTest, inputRnBTest];
outputTraining = [outputPopTraining, outputRapTraining, outputRnBTraining];
outputTest = [outputPopTest, outputRapTest, outputRnBTest];

ind = randperm(length(outputTraining));
inputTraining = inputTraining(:, ind);
outputTraining = outputTraining(:, ind);

%% Unakrsna validacija

Abest = 0;
F1best = 0;
architecture = {[10, 5], [12, 6, 3], [4]};
regularization = [0.1, 0.5, 0.9];
learningRates = [0.5, 0.05, 0.005];

for reg = regularization
    for lr = learningRates
        for arh = length(architecture)
            rng(200);
            net = patternnet(architecture{arh});
            
            net.performParam.regularization = reg;
            net.trainFcn = 'traingda';
            net.trainParam.lr = lr;
            net.trainParam.epochs = 1000;
            net.trainParam.goal = 1e-4;
            net.trainParam.max_fail = 20;
            %net.trainParam.showWindow = false;

            [net, info] = train(net, inputTraining, outputTraining);

            pred = sim(net, inputTest);
            pred = round(pred);

            [~, cm] = confusion(outputTest, pred);
            A = 100*sum(trace(cm))/sum(sum(cm));
            F1 = 2*cm(2, 2)/(cm(2, 1)+cm(1, 2)+2*cm(2, 2));

            disp(['Reg = ' num2str(reg) ', ACC = ' num2str(A) ', F1 = ' num2str(F1)])
            disp(['LR = ' num2str(lr) ', epoch = ' num2str(info.best_epoch)])

            if F1 > F1best
                F1best = F1;
                Abest = A;
                %reg_best = reg;
                %w_best = w;
                %lr_best = lr;
                %arh_best = arhitektura{arh};
                %ep_best = info.best_epoch;
            end
        end
    end
end
