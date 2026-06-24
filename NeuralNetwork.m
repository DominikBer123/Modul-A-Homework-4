%%
close all
clear
clc

pendulum_process_obf([], 0); 

Ts = 0.01;              % Čas vzorčenja (s)
T_sim = 400;             % Skupni čas simulacije (s)
N = ceil(T_sim / Ts);   % Število korakov
t_vec = (0:N-1)' * Ts;  % Časovni vektor

TimeOfPeriod = 1.5;       % Period of alternation (seconds)
uBaseValue = 1.5;         % Baseline value
amplitude = 0.5;        % How much it steps up/down from baseline

u_vec = zeros(N,1);
numSteps = ceil(t_vec(end) / TimeOfPeriod);

for k = 1:numSteps

    idx = (t_vec >= (k-1)*TimeOfPeriod) & (t_vec < k*TimeOfPeriod);

    stepNoise = (2*rand - 1) * amplitude;

    u_vec(idx) = uBaseValue + stepNoise;

end

disp('Začetek simulacije...');
y_vec = pendulum_process_obf(u_vec, Ts);
disp('Simulacija končana.');


kot_rad = y_vec(:, 1);
hitrost_rad_s = y_vec(:, 2);
kot_stopinje = kot_rad * 180 / pi;

figure('Name', 'Simulacija nihala', 'Color', 'w');

subplot(2, 1, 1);
plot(t_vec, u_vec', 'LineWidth', 1.2);
ylabel('Navor (Nm)');
title('Vhodni signal');
grid on;

subplot(2, 1, 2);
plot(t_vec, kot_stopinje, 'LineWidth', 1.2);
hold on;
yline(10, 'r--', 'Spodnja meja (10°)');
yline(100, 'r--', 'Zgornja meja (100°)');
ylabel('Kot (°)');
title('Izhodni kot');
grid on;
ylim([0, 110]);


%%
% 1. Train the network and get back the model weights struct
[trainMse, numParams, modelParams] = run_deep_experiment(u_vec, y_vec(:,1), [16 16], 42);

% 2. Create a smooth evaluation vector across your input range
xTestGrid = linspace(min(xTrain), max(xTrain), 1000);

% 3. Use your new predict function
yPredictions = predict_deep_nn(xTestGrid, modelParams);

% 4. Plot the results to visually verify the curve fit!
figure('Name', 'Model Prediction Test');
plot(xTrain, yTrain, 'o', 'Color', [0.5 0.5 0.5], 'DisplayName', 'Noisy Training Points'); hold on;
plot(xTestGrid, yPredictions, 'r-', 'LineWidth', 2.5, 'DisplayName', 'NN Model Curve');
xlabel('Torque u (Nm)');
ylabel('Angle \phi (rad)');
title('Visualizing Network Output Curve via Prediction Function');
grid on;
legend('Location', 'best');

%%
function [trainMse, numParams] = run_deep_experiment(xTrain, yTrain, hiddenSizes, seed)
    % Deep two-hidden-layer tanh + linear network trained with the
    % SAME full-batch gradient-descent algorithm as the Python script
    % vaja4_part2_neural_network_solution.py (class TwoHiddenLayerNet).
    %
    % Pipeline summary:
    %   - architecture: [1] -> tanh(h1) -> tanh(h2) -> linear(1)
    %   - init scales: W1 ~ N(0, 0.5^2), b1 ~ N(0, 0.5^2),
    %                  W2 ~ N(0, 1/h1),  b2 ~ N(0, 0.5^2),
    %                  W3 ~ N(0, 1/h2),  b3 = 0
    %   - loss: mean squared error
    %   - optimizer: vanilla full-batch gradient descent, lr = 0.05
    %   - iterations: 8000
    %   - no train/val/test split, no mapminmax, no early stopping.

    learningRate = 0.05;
    numIterations = 8000;
    
    xTrain = xTrain(:)';   % row, 1 x N
    yTrain = yTrain(:)';   % row, 1 x N
    
    h1 = hiddenSizes(1);
    h2 = hiddenSizes(2);
    n  = numel(xTrain);
    
    rng(seed);
    W1 = randn(h1, 1) * 0.5;
    b1 = randn(h1, 1) * 0.5;
    W2 = randn(h2, h1) * (1.0 / sqrt(h1));
    b2 = randn(h2, 1) * 0.5;
    W3 = randn(1, h2) * (1.0 / sqrt(h2));
    b3 = 0.0;
    
    lossHistory = zeros(1, numIterations);
    
    for it = 1:numIterations
        % --- forward pass -------------------------------------------------
        z1 = W1 * xTrain + b1;       % h1 x N
        a1 = tanh(z1);
        z2 = W2 * a1 + b2;           % h2 x N
        a2 = tanh(z2);
        out = W3 * a2 + b3;          % 1 x N
        err = out - yTrain;          % 1 x N
        
        lossHistory(it) = mean(err.^2);
        
        % --- backward pass ------------------------------------------------
        gOut = (2.0 / n) * err;                 % 1 x N
        gW3  = gOut * a2';                       % 1 x h2
        gb3  = sum(gOut);                        % scalar
        gA2  = W3' * gOut;                       % h2 x N
        gZ2  = gA2 .* (1.0 - a2.^2);             % h2 x N
        gW2  = gZ2 * a1';                        % h2 x h1
        gb2  = sum(gZ2, 2);                      % h2 x 1
        gA1  = W2' * gZ2;                        % h1 x N
        gZ1  = gA1 .* (1.0 - a1.^2);             % h1 x N
        gW1  = gZ1 * xTrain';                    % h1 x 1
        gb1  = sum(gZ1, 2);                      % h1 x 1
        
        % --- vanilla gradient-descent update -----------------------------
        W3 = W3 - learningRate * gW3;
        b3 = b3 - learningRate * gb3;
        W2 = W2 - learningRate * gW2;
        b2 = b2 - learningRate * gb2;
        W1 = W1 - learningRate * gW1;
        b1 = b1 - learningRate * gb1;
    end
    
    % Final predictions on training data
    yHatTrain = W3 * tanh(W2 * tanh(W1 * xTrain + b1) + b2) + b3;
    trainMse = mean((yTrain - yHatTrain).^2);
    numParams = (h1 * 1 + h1) + (h2 * h1 + h2) + (h2 + 1);
    
    % Setup for clean plotting (sorting xTrain so lines don't criss-cross)
    [xTrainSorted, sortIdx] = sort(xTrain);
    yHatTrainSorted = yHatTrain(sortIdx);
    
    % Plotting results
    figure('Name', 'Deep NN demonstration (vanilla GD, training only)', ...
        'Position', [220 120 1100 420]);
    tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    
    % Tile 1: Data and Fit
    nexttile;
    plot(xTrain, yTrain, 'o', 'Color', [0.45 0.45 0.45], 'MarkerSize', 3.5, ...
        'DisplayName', 'train data'); hold on;
    plot(xTrainSorted, yHatTrainSorted, 'Color', [0.55 0.30 0.75], 'LineWidth', 2.0, ...
        'DisplayName', 'deep NN fit');
    grid on;
    title(sprintf('Deep NN [%s] (params = %d)\\ntrain MSE = %.4f', ...
        num2str(hiddenSizes), numParams, trainMse));
    xlabel('Input x'); ylabel('Output y');
    legend('Location', 'northwest');
    
    % Tile 2: Loss History
    nexttile;
    plot(1:numIterations, lossHistory, 'Color', [0.55 0.30 0.75], 'LineWidth', 1.2);
    set(gca, 'YScale', 'log');
    xlabel('Iteration'); ylabel('Training MSE');
    title('Deep NN training loss (full-batch GD, lr = 0.05)');
    grid on;
    
    exportgraphics(gcf, 'part2_deep_nn.png', 'Resolution', 180);
    fprintf('Deep NN [%s]: params = %d, train MSE = %.5f\n', ...
        num2str(hiddenSizes), numParams, trainMse);

end



function yHat = predict_deep_nn(xInput, modelParams)
    % PREDICT_DEEP_NN Generates predictions for a two-hidden-layer tanh network
    %
    % Inputs:
    %   xInput      - Row vector (1 x N) of input values (e.g., torques)
    %   modelParams - Struct containing trained weights and biases:
    %                 modelParams.W1, modelParams.b1, etc.
    
    xInput = xInput(:)'; % Ensure it is a row vector (1 x N)
    
    % Extract parameters
    W1 = modelParams.W1;   b1 = modelParams.b1;
    W2 = modelParams.W2;   b2 = modelParams.b2;
    W3 = modelParams.W3;   b3 = modelParams.b3;
    
    % Forward pass
    z1 = W1 * xInput + b1;
    a1 = tanh(z1);
    z2 = W2 * a1 + b2;
    a2 = tanh(z2);
    
    yHat = W3 * a2 + b3; % Output predictions (1 x N row vector)
end