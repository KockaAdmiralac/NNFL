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
NPopV = NPopT + floor(0.15*NPop);

NRap = length(Rap);
NRapT = floor(0.7*NRap);
NRapV = NRapT + floor(0.15*NRap);

NRnB = length(RnB);
NRnBT = floor(0.7*NRnB);
NRnBV = NRnBT + floor(0.15*NRnB);

inputPopTraining = Pop(:, 1 : NPopT);
inputPopValidation = Pop(:, NPopT + 1 : NPopV);
inputPopTest = Pop(:, NPopV + 1 : NPop);
outputPopTraining = outputPop(:, 1 : NPopT);
outputPopValidation = outputPop(:, NPopT + 1 : NPopV);
outputPopTest = outputPop(:, NPopV + 1 : NPop);

inputRnBTraining = RnB(:, 1 : NRnBT);
inputRnBValidation = RnB(:, NRnBT + 1 : NRnBV);
inputRnBTest = RnB(:, NRnBV + 1 : NRnB);
outputRnBTraining = outputRnB(:, 1 : NRnBT);
outputRnBValidation = outputRnB(:, NRnBT + 1 : NRnBV);
outputRnBTest = outputRnB(:, NRnBV + 1 : NRnB);

inputRapTraining = Rap(:, 1 : NRapT);
inputRapValidation = Rap(:, NRapT + 1 : NRapV);
inputRapTest = Rap(:, NRapV + 1 : NRap);
outputRapTraining = outputRap(:, 1 : NRapT);
outputRapValidation = outputRap(:, NRapT + 1 : NRapV);
outputRapTest = outputRap(:, NRapV + 1 : NRap);


inputTraining = [inputPopTraining, inputRapTraining, inputRnBTraining];
inputValidation = [inputPopValidation, inputRapValidation, inputRnBValidation];
inputTest = [inputPopTest, inputRapTest, inputRnBTest];
outputTraining = [outputPopTraining, outputRapTraining, outputRnBTraining];
outputValidation = [outputPopValidation, outputRapValidation, outputRnBValidation];
outputTest = [outputPopTest, outputRapTest, outputRnBTest];

ind = randperm(length(outputTraining));
inputTraining = inputTraining(:, ind);
outputTraining = outputTraining(:, ind);

ind = randperm(length(outputValidation));
inputValidation = inputValidation(:, ind);
outputValidation = outputValidation(:, ind);

inputAll = [inputTraining, inputValidation];
outputAll = [outputTraining, outputValidation];

%% Unakrsna validacija

Abest = 0;
F1best = 0;
architecture = {[3, 5], [12, 6], [12,8,4], [20,10]};
regularization = [0, 0.1, 0.5, 0.9];
learningRates = [0.5, 0.05, 0.005];
%weights =  [1,2,5,10];
precision = @(confusionMat) diag(confusionMat)./sum(confusionMat,2);
recall = @(confusionMat) diag(confusionMat)./sum(confusionMat,1)';
f1Scores = @(confusionMat) 2*(precision(confusionMat).*recall(confusionMat))./(precision(confusionMat)+recall(confusionMat));

for r = regularization
    for lr = learningRates
        for arh = [1:length(architecture)]
            rng(200);
            net = patternnet(architecture{arh});
            
            net.performParam.regularization = r;
            net.trainFcn = 'traingda';
            net.trainParam.lr = lr;
            net.trainParam.epochs = 1000;
            net.trainParam.goal = 1e-4;
            net.trainParam.max_fail = 20;
            net.trainParam.showWindow = false;
           
            net.divideFcn = 'divideind';
            net.divideParam.trainInd = 1 : length(inputTraining);
            net.divideParam.valInd = length(inputTraining) + 1 : (length(inputTraining) + length(inputValidation));
            net.divideParam.testInd = [];%(length(inputTraining) + length(inputValidation)) + 1 : length(inputAll);
            
            %weight = (outputTraining(1,:))*(w-1)+1;   

            %[net, info] = train(net, inputTraining, outputTraining,[],[],weight);
            [net, info] = train(net, inputAll, outputAll);

            pred = sim(net, inputTest);
            pred = round(pred);

            [~, cm] = confusion(outputTest, pred);
            A = 100*trace(cm)/sum(sum(cm));
            F1 = mean(f1Scores(cm));
            
            disp(['Arh = ' num2str(architecture{arh})])
            disp(['R = ' num2str(r) ', ACC = ' num2str(A) ])
            disp(['LR = ' num2str(lr) ', epoch = ' num2str(info.best_epoch)])
            disp(['F1 = ', num2str(F1)])

            if F1 > F1best
                Abest = A;
                F1best = F1;
                %w_best = w;
                reg_best = r;
                lr_best = lr;
                arh_best = architecture{arh};
                ep_best = info.best_epoch;
                net_best = net;
                info_best = info;
            end
        end
    end
end

%% Generisanje rezultata
plotperform(info_best), legend('Location', 'southwest')

pred = sim(net_best, inputTraining);
pred = round(pred);
figure, plotconfusion(outputTraining, pred), title('Matrica konfuzije na trening skupu mreže'), xlabel('Očekivana klasa'), ylabel('Dobijena klasa')

pred = sim(net_best, inputTest);
pred = round(pred);
figure, plotconfusion(outputTest, pred), title('Matrica konfuzije na test skupu mreže'), xlabel('Očekivana klasa'), ylabel('Dobijena klasa')

[~, cm] = confusion(outputTest, pred);
precision(cm)
recall(cm)
f1Scores(cm)
