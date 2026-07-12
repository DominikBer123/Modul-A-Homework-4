%% Full Code with Modified Neural Network Input Structure
% Input: [u(k-1), u(k-2), y(k-1), y(k-2)]
% Output: y(k) - current angle prediction

close all
clear
clc

%% 1. Generate Training Data
pendulum_process_obf([], 0); 

Ts = 0.05;              % Čas vzorčenja (s)
T_sim = 500;            % Skupni čas simulacije (s)
N = ceil(T_sim / Ts);   % Število korakov
t_vec = (0:N-1)' * Ts;  % Časovni vektor

% Generate input signal (random step)
TimeOfPeriod = 1.5;       % Period of alternation (seconds)
uBaseValue = 1.7;       % Baseline value
amplitude = 0.6;        % How much it steps up/down from baseline

u_vec = zeros(N, 1);
numSteps = ceil(t_vec(end) / TimeOfPeriod);

for k = 1:numSteps
    idx = (t_vec >= (k-1)*TimeOfPeriod) & (t_vec < k*TimeOfPeriod);
    stepNoise = (2*rand - 1) * amplitude;
    u_vec(idx) = uBaseValue + stepNoise;
end

disp('Začetek simulacije...');
y_vec = pendulum_process_obf(u_vec, Ts);
disp('Simulacija končana.');

% Extract angle from y_vec (first column)
angle = y_vec(:, 1);
angle_deg = angle * 180 / pi;

% Plot training data
figure('Name', 'Training Data', 'Color', 'w');

subplot(2, 1, 1);
plot(t_vec, u_vec', 'LineWidth', 1.2);
ylabel('Navor (Nm)');
title('Vhodni signal - Random Step');
grid on;

subplot(2, 1, 2);
plot(t_vec, angle_deg, 'LineWidth', 1.2);
hold on;
yline(10, 'r--', 'Spodnja meja (10°)');
yline(100, 'r--', 'Zgornja meja (100°)');
ylabel('Kot (°)');
title('Izhodni kot');
grid on;
ylim([0, 110]);

%% 2. Create Training Data with New Input Structure
% Input: [u(k-1), u(k-2), y(k-1), y(k-2)]
% Output: y(k) - current angle

% We need to start from index 3 (need k-2) and go to end
nSamples = length(u_vec) - 2;  % We lose 2 samples at the beginning

% Initialize input matrix (4 x nSamples) and target vector
xTrain = zeros(4, nSamples);
yTrain = zeros(1, nSamples);

for k = 3:length(u_vec)  % k represents current time index
    idx = k - 2;  % Index in the training arrays (starts at 1)
    xTrain(:, idx) = [u_vec(k-1);   % u(k-1)
                      u_vec(k-2);   % u(k-2)
                      angle(k-1);   % y(k-1)
                      angle(k-2)];  % y(k-2)
    yTrain(idx) = angle(k);         % y(k) - predict current angle
end

% Display data dimensions
fprintf('Training data dimensions:\n');
fprintf('xTrain: %d x %d (features x samples)\n', size(xTrain, 1), size(xTrain, 2));
fprintf('yTrain: %d x %d\n', size(yTrain, 1), size(yTrain, 2));

%% 3. Train the Network
disp('Training neural network...');
[trainMse, numParams, modelParams] = run_deep_experiment(xTrain, yTrain, [16 16], 42);
disp('Training complete!');

%% 4. Test the Network on Training Data
yPredictions = predict_deep_nn(xTrain, modelParams);

% Plot training results
figure('Name', 'Training Results');
subplot(2, 1, 1);
plot(angle(3:end), 'b-', 'LineWidth', 1.5);  % angle(3:end) corresponds to y(k) for k>=3
hold on;
plot(yPredictions', 'r--', 'LineWidth', 1.5);
xlabel('Time Step');
ylabel('Angle (rad)');
title('Training Data: Actual vs Predicted');
legend('Actual y(k)', 'Predicted y(k)');
grid on;

subplot(2, 1, 2);
error = rad2deg(yPredictions') - rad2deg(angle(3:end));
plot(error, 'k-', 'LineWidth', 1);
xlabel('Time Step');
ylabel('Error (deg)');
title('Prediction Error on Training Data');
grid on;

fprintf('Training MSE: %.6f\n', trainMse);

%% 5. Generate Test Data (Different Input Pattern)
disp('Generating test data...');
pendulum_process_obf([], 0); 

Ts_test = 0.05;              % Čas vzorčenja (s)
T_sim_test = 100;            % Skupni čas simulacije (s)
N_test = ceil(T_sim_test / Ts_test);
t_vec_test = (0:N_test-1)' * Ts_test;

% Generate step input with different parameters
TimeOfPeriod = 1.5;       % Period of alternation (seconds)
uBaseValue = 1.7;         % Baseline value
amplitude = 0.5;          % How much it steps up/down from baseline

u_test = zeros(N_test, 1);
numSteps = ceil(t_vec_test(end) / TimeOfPeriod);

for k_step = 1:numSteps
    idx = (t_vec_test >= (k_step-1)*TimeOfPeriod) & (t_vec_test < k_step*TimeOfPeriod);
    stepNoise = (2*rand - 1) * amplitude;
    u_test(idx) = uBaseValue + stepNoise;
end

disp('Running test simulation...');
y_test = pendulum_process_obf(u_test, Ts_test);
disp('Test simulation complete.');

angle_test = y_test(:, 1);
angle_test_deg = angle_test * 180 / pi;

% Plot test data
figure('Name', 'Test Data', 'Color', 'w');

subplot(2, 1, 1);
plot(t_vec_test, u_test', 'LineWidth', 1.2);
ylabel('Navor (Nm)');
title('Test Vhodni signal - Step with Random Amplitude');
grid on;

subplot(2, 1, 2);
plot(t_vec_test, angle_test_deg, 'LineWidth', 1.2);
hold on;
yline(10, 'r--', 'Spodnja meja (10°)');
yline(100, 'r--', 'Zgornja meja (100°)');
ylabel('Kot (°)');
title('Test Izhodni kot');
grid on;
ylim([0, 110]);

%% 6. Prepare Test Data with Same Input Structure
nSamples_test = length(u_test) - 2;
xTest = zeros(4, nSamples_test);
yTest = zeros(1, nSamples_test);

for k = 3:length(u_test)
    idx = k - 2;
    xTest(:, idx) = [u_test(k-1);   % u(k-1)
                     u_test(k-2);   % u(k-2)
                     angle_test(k-1);   % y(k-1)
                     angle_test(k-2)];  % y(k-2)
    yTest(idx) = angle_test(k);    % y(k)
end

%% 7. Make Predictions on Test Data
yPredictions_test = predict_deep_nn(xTest, modelParams);

% Plot test results
figure('Name', 'Test Results');
subplot(2, 1, 1);
plot(angle_test(3:end), 'b-', 'LineWidth', 1.5);
hold on;
plot(yPredictions_test', 'r--', 'LineWidth', 1.5);
xlabel('Time Step');
ylabel('Angle (rad)');
title('Test Data: Actual vs Predicted');
legend('Actual y(k)', 'Predicted y(k)');
grid on;

subplot(2, 1, 2);
error_test = rad2deg(yPredictions_test') - rad2deg(angle_test(3:end));
plot(error_test, 'k-', 'LineWidth', 1);
xlabel('Time Step');
ylabel('Error (deg)');
title('Prediction Error on Test Data');
grid on;

testMse = mean(error_test.^2);
fprintf('Test MSE: %.6f\n', testMse);

%% 8. Compare Training and Test Performance
figure('Name', 'Performance Comparison');
subplot(2, 1, 1);
histogram(error, 'FaceColor', 'b', 'FaceAlpha', 0.5, 'Normalization', 'pdf');
hold on;
histogram(error_test, 'FaceColor', 'r', 'FaceAlpha', 0.5, 'Normalization', 'pdf');
xlabel('Prediction Error (rad)');
ylabel('Probability Density');
title('Error Distribution: Training vs Test');
legend('Training', 'Test');
grid on;

subplot(2, 1, 2);
bar([trainMse, testMse]);
set(gca, 'XTickLabel', {'Training', 'Test'});
ylabel('Mean Squared Error');
title('MSE Comparison');
grid on;

%% 9. Scatter Plot of Predictions vs Actual
figure('Name', 'Prediction Scatter');
subplot(1, 2, 1);
scatter(angle(3:end), yPredictions', 20, 'b', 'filled');
hold on;
plot([0, max(angle)], [0, max(angle)], 'r--', 'LineWidth', 2);
xlabel('Actual Angle (rad)');
ylabel('Predicted Angle (rad)');
title('Training Data');
grid on;
axis equal;

subplot(1, 2, 2);
scatter(angle_test(3:end), yPredictions_test', 20, 'r', 'filled');
hold on;
plot([0, max(angle_test)], [0, max(angle_test)], 'r--', 'LineWidth', 2);
xlabel('Actual Angle (rad)');
ylabel('Predicted Angle (rad)');
title('Test Data');
grid on;
axis equal;

%% ========================================================================
%% FUNCTION DEFINITIONS (Same as before, but updated comments)
%% ========================================================================

function [trainMse, numParams, modelParams] = run_deep_experiment(xTrain, yTrain, hiddenSizes, seed)
    % Deep two-hidden-layer tanh + linear network
    % Input: xTrain - 4 x N matrix [u(k-1); u(k-2); y(k-1); y(k-2)]
    % Output: yTrain - 1 x N vector [y(k)]
    
    learningRate = 0.05;
    numIterations = 8000;
    
    % Get dimensions
    [nX, nSamples] = size(xTrain);  % nX = 4 (number of features)
    h1 = hiddenSizes(1);
    h2 = hiddenSizes(2);
    n = nSamples;
    
    rng(seed);
    
    % Initialize weights and biases
    W1 = randn(h1, nX) * 0.5;       % h1 x 4
    b1 = randn(h1, 1) * 0.5;        % h1 x 1
    W2 = randn(h2, h1) * (1.0 / sqrt(h1));
    b2 = randn(h2, 1) * 0.5;
    W3 = randn(1, h2) * (1.0 / sqrt(h2));
    b3 = 0.0;
    
    lossHistory = zeros(1, numIterations);
    
    for it = 1:numIterations
        % --- Forward pass ------------------------------------------------
        z1 = W1 * xTrain + b1;       % h1 x N
        a1 = tanh(z1);
        z2 = W2 * a1 + b2;           % h2 x N
        a2 = tanh(z2);
        out = W3 * a2 + b3;          % 1 x N
        err = out - yTrain;          % 1 x N
        
        lossHistory(it) = mean(err.^2);
        
        % --- Backward pass -----------------------------------------------
        gOut = (2.0 / n) * err;                 % 1 x N
        gW3  = gOut * a2';                      % 1 x h2
        gb3  = sum(gOut);                       % scalar
        gA2  = W3' * gOut;                      % h2 x N
        gZ2  = gA2 .* (1.0 - a2.^2);            % h2 x N
        gW2  = gZ2 * a1';                       % h2 x h1
        gb2  = sum(gZ2, 2);                     % h2 x 1
        gA1  = W2' * gZ2;                       % h1 x N
        gZ1  = gA1 .* (1.0 - a1.^2);            % h1 x N
        gW1  = gZ1 * xTrain';                   % h1 x 4
        gb1  = sum(gZ1, 2);                     % h1 x 1
        
        % --- Gradient descent update -------------------------------------
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
    numParams = (h1 * nX + h1) + (h2 * h1 + h2) + (h2 + 1);
    
    % Sort for plotting
    [~, sortIdx] = sort(xTrain(3, :));  % Sort by y(k-1) - previous output
    xTrainSorted = xTrain(:, sortIdx);
    yHatTrainSorted = yHatTrain(sortIdx);
    yTrainSorted = yTrain(sortIdx);
    
    % Plotting results
    figure('Name', 'Deep NN Training', 'Position', [220 120 1100 420]);
    tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
    
    % Tile 1: Data and Fit (show previous output vs predicted current output)
    nexttile;
    plot(xTrain(3, :), yTrain, 'o', 'Color', [0.45 0.45 0.45], 'MarkerSize', 3.5, ...
        'DisplayName', 'Training data'); 
    hold on;
    plot(xTrainSorted(3, :), yHatTrainSorted, 'Color', [0.55 0.30 0.75], 'LineWidth', 2.0, ...
        'DisplayName', 'NN fit');
    grid on;
    title(sprintf('Deep NN [%s] (params = %d)\nTrain MSE = %.6f', ...
        num2str(hiddenSizes), numParams, trainMse));
    xlabel('y(k-1) - Previous Output');
    ylabel('y(k) - Current Angle');
    legend('Location', 'northwest');
    
    % Tile 2: Loss History
    nexttile;
    plot(1:numIterations, lossHistory, 'Color', [0.55 0.30 0.75], 'LineWidth', 1.2);
    set(gca, 'YScale', 'log');
    xlabel('Iteration');
    ylabel('Training MSE');
    title('Training Loss (full-batch GD, lr = 0.05)');
    grid on;
    
    fprintf('Deep NN [%s]: params = %d, train MSE = %.8f\n', ...
        num2str(hiddenSizes), numParams, trainMse);
    
    % Store model parameters
    modelParams.W1 = W1;
    modelParams.b1 = b1;
    modelParams.W2 = W2;
    modelParams.b2 = b2;
    modelParams.W3 = W3;
    modelParams.b3 = b3;
    modelParams.nInputs = nX;
    modelParams.inputNames = {'u(k-1)', 'u(k-2)', 'y(k-1)', 'y(k-2)'};
end

function yHat = predict_deep_nn(xInput, modelParams)
    % PREDICT_DEEP_NN Generates predictions for the trained network
    %
    % Inputs:
    %   xInput      - Matrix (4 x N) where each column is [u(k-1); u(k-2); y(k-1); y(k-2)]
    %   modelParams - Struct containing trained weights and biases
    
    % Ensure xInput has correct dimensions (features x samples)
    if size(xInput, 1) ~= modelParams.nInputs
        if size(xInput, 2) == modelParams.nInputs
            xInput = xInput';  % Transpose if needed
        else
            error('Input dimensions incorrect. Expected %d features, got %d', ...
                modelParams.nInputs, size(xInput, 1));
        end
    end
    
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