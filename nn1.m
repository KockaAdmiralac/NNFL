clc, clear, close all
rng(200);

%% Vizuelizacija klasa
rng(200);
dataset = load('data/dataset2.mat').pod';

input = dataset(1:2, :);
output = dataset(3, :);

K1 = input(:, output == 1);
K2 = input(:, output == 2);
K3 = input(:, output == 3);

%figure('Name', 'Z1F1', 'NumberTitle', 'off')
figure
hold all
plot(K1(1, :), K1(2, :), 'o')
plot(K2(1, :), K2(2, :), 'x')
plot(K3(1, :), K3(2, :), '*')
legend('K1', 'K2', 'K3')
xlabel('X osa')
ylabel('Y osa')
title('Podaci po klasama')

%% One-Hot Encoding
rng(200);
outputOH = zeros(length(output), 3);
outputOH(output == 1, 1) = 1;
outputOH(output == 2, 2) = 1;
outputOH(output == 3, 3) = 1;
outputOH = outputOH';

%% Podela na trening i test skup
rng(200);

N = length(output);
ind = randperm(N);
ind_training = ind(1 : 0.7 * N);
ind_test = ind(0.7*N + 1 : N);

input_training = input(:, ind_training);
output_training = outputOH(:, ind_training);

input_test = input(:, ind_test);
output_test = outputOH(:, ind_test);

%% Optimalna neuralna mreža
rng(200);

netB = patternnet([6 4]);

netB.divideFcn = '';

netB.trainParam.epochs = 1000;
netB.trainParam.goal = 1e-3;
netB.trainParam.min_grad = 1e-3;

netB.layers{1}.transferFcn = 'poslin';
netB.layers{2}.transferFcn = 'poslin';
netB.layers{3}.transferFcn = 'softmax';

netB = train(netB, input_training, output_training);

figure, plotconfusion(output_training, netB(input_training)), title('Matrica konfuzije na trening skupu optimalne mreže'), xlabel('Očekivana klasa'), ylabel('Dobijena klasa')
figure, plotconfusion(output_test, netB(input_test)), title('Matrica konfuzije na test skupu optimalne mreže'), xlabel('Očekivana klasa'), ylabel('Dobijena klasa')

%% Neprilagođena neuralna mreža
rng(200);

netU = patternnet([1 1]);

netU.divideFcn = '';

netU.trainParam.epochs = 1000;
netU.trainParam.goal = 1e-3;
netU.trainParam.min_grad = 1e-3;

netU.layers{1}.transferFcn = 'poslin';
netU.layers{2}.transferFcn = 'poslin';
netU.layers{3}.transferFcn = 'softmax';

netU = train(netU, input_training, output_training);

figure, plotconfusion(output_training, netU(input_training)), title('Matrica konfuzije na trening skupu neprilagođene mreže'), xlabel('Očekivana klasa'), ylabel('Dobijena klasa')
figure, plotconfusion(output_test, netU(input_test)), title('Matrica konfuzije na test skupu neprilagođene mreže'), xlabel('Očekivana klasa'), ylabel('Dobijena klasa')

%% Preobučavajuća neuralna mreža
rng(200);

netO = patternnet([100 100 100 100 100]);

netO.divideFcn = '';

netO.trainParam.epochs = 100;
netO.trainParam.goal = 1e-3;
netO.trainParam.min_grad = 1e-3;

netO.layers{1}.transferFcn = 'poslin';
netO.layers{2}.transferFcn = 'poslin';
netO.layers{3}.transferFcn = 'softmax';

netO = train(netO, input_training, output_training);

figure, plotconfusion(output_training, netO(input_training)), title('Matrica konfuzije na trening skupu preobučavajuće mreže'), xlabel('Očekivana klasa'), ylabel('Dobijena klasa')
figure, plotconfusion(output_test, netO(input_test)), title('Matrica konfuzije na test skupu preobučavajuće mreže'), xlabel('Očekivana klasa'), ylabel('Dobijena klasa')


%% Granica odlucivanja
titles = ["Optimalna mreža", "Neprilagođena mreža", "Preobučena mreža"];
for netIndex = 1:3
    Ntest = 500;
    inputDB = [];
    x1 = linspace(-1, 1, Ntest);
    x2 = linspace(-1, 1, Ntest);

    net = netB;
    if netIndex == 2
        net = netU;
    end
    if netIndex == 3
        net = netO;
    end
    
    for x11 = x1
        pom = [x11*ones(1, Ntest); x2];
        inputDB = [inputDB, pom];
    end

    predGO = sim(net, inputDB);
    [vr, class] = max(predGO);

    K1db = inputDB(:, class == 1);
    K2db = inputDB(:, class == 2);
    K3db = inputDB(:, class == 3);

    figure, hold all 
    plot(K1db(1, :), K1db(2, :), 'b.')
    plot(K2db(1, :), K2db(2, :), 'r.')
    plot(K3db(1, :), K3db(2, :), 'w.')
    plot(K1(:, 1), K1(:, 2), 'bo')
    plot(K2(:, 1), K2(:, 2), 'r*')
    plot(K3(:, 1), K3(:, 2), 'wd')
    legend('K1', 'K2', 'K3')
    xlabel('X osa'), ylabel('Y osa')
    title(titles(netIndex))
end
