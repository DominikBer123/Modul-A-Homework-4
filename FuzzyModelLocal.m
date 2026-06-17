%%
close all
clear
clc

pendulum_process_obf([], 0); 

Ts = 0.06;              % Čas vzorčenja (s)
T_sim = 100;             % Skupni čas simulacije (s)
N = ceil(T_sim / Ts);   % Število korakov
t_vec = (0:N-1)' * Ts;  % Časovni vektor

n = (0:N-1);
k = 3/N;
u_vec = k * n;


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


figure
plot(u_vec,rad2deg(y_vec(:,1)) ,"*")
hold on;
yline(10, 'r--', 'Spodnja meja (10°)');
yline(100, 'r--', 'Zgornja meja (100°)');
ylabel('Kot (°)');
title('Izhodni kot');
grid on

%% GK Optimization Loop (Updating given variables directly)

xTrain = u_vec';
yTrain = y_vec(:,1);

numRules = 5;
joint = [xTrain, yTrain];
[numSamples, numDims] = size(joint);
fuzziness = 2.0;
clusterVolume = 1.0;
regularisation = 1e-6;
maxIterations = 200;
tolerance = 1e-6;
exponent = 2.0 / (fuzziness - 1.0);

rng(0);
membership = rand(numRules, numSamples);
membership = membership ./ sum(membership, 1);
centersXY = zeros(numRules, numDims);
covariances = repmat(eye(numDims), 1, 1, numRules);
squaredDistance = zeros(numRules, numSamples);

for iter = 1:maxIterations
    old_centers = centersXY;
    
    % 1. Compute Fuzzy Covariances
    for i = 1:numRules
        v = centersXY(i, :)';
        diffs = joint' - v; % joint data is (numSamples x 2), transpose it to (2 x N)
        weights = membership(i, :) .^ fuzziness;
        
        numerator = (diffs .* weights) * diffs';
        denominator = sum(weights);
        
        covariances(:, :, i) = numerator / denominator;
    end
    
    % 2. Compute Norm Matrices & Squared Distances
    for i = 1:numRules
        v = centersXY(i, :)';
        Fi = covariances(:, :, i);
        
        Fi_reg = Fi + regularisation * eye(numDims); 
        Ai = (clusterVolume * det(Fi_reg))^(1/numDims) * inv(Fi_reg);
        
        diffs = joint' - v;
        squaredDistance(i, :) = sum((diffs' * Ai) .* diffs', 2)';
    end
    
    % 3. Update Membership Matrix
    dist_mod = squaredDistance + 1e-10;
    for i = 1:numRules
        membership(i, :) = sum((dist_mod(i, :) ./ dist_mod) .^ (1 / (fuzziness - 1)), 1);
    end
    membership = 1 ./ membership;
    
    % 4. Update Cluster Centers
    for i = 1:numRules
        weights = membership(i, :) .^ fuzziness;
        centersXY(i, :) = (joint' * weights' / sum(weights))';
    end
    
    % 5. Check Convergence
    if max(abs(centersXY - old_centers), [], 'all') < tolerance
        fprintf('GK converged at iteration %d\n', iter);
        break;
    end
end

% ===== Membership on training data only =====

% centers (1D projection from GK clusters)
centers = centersXY(:,1);

% estimate sigma (same idea as yours)
sorted_centers = sort(centers);
sigma = mean(diff(sorted_centers)) / 2;

numRules = length(centers);
numSamples = length(xTrain);

% Gaussian memberships (unnormalized)
muTrain = zeros(numRules, numSamples);

for i = 1:numRules
    muTrain(i,:) = exp(-(xTrain - centers(i)).^2 / (2*sigma^2));
end

% normalize memberships
phiTrain = muTrain ./ sum(muTrain,1);

% ===== PLOT memberships =====

figure('Name','Train Memberships','Color','w');
hold on;

ruleColors = lines(numRules);

for i = 1:numRules
    plot(xTrain, phiTrain(i,:), 'LineWidth', 1.2, ...
        'Color', ruleColors(i,:), ...
        'DisplayName', sprintf('Rule %d', i));
end

xlabel('xTrain');
ylabel('Membership degree');
title('Normalized Gaussian Memberships (Training Set)');
grid on;
legend('Location','best');

%
% ===== Hard assignment from memberships =====
[~, hardAssign] = max(phiTrain, [], 1);
ruleColors = lines(numRules);

% ===== Plot xTrain vs yTrain colored by rule =====
figure('Name','Training data colored by membership','Color','w');
hold on;

% 1. Plot the training data points
for i = 1:numRules
    idx = (hardAssign == i);
    
    scatter(xTrain(idx), yTrain(idx), 20, ...
        ruleColors(i,:), 'filled', ...
        'DisplayName', sprintf('Rule %d Data', i));
end

% 2. Overlay the GK cluster centers (centersXY)
for i = 1:numRules
    % Extract the X and Y coordinate for the i-th cluster center
    centerX = centersXY(i, 1);
    centerY = centersXY(i, 2);
    
    % Plot center as a larger, distinct marker with a black edge
    plot(centerX, centerY, 'd', ...
        'MarkerSize', 12, ...
        'MarkerFaceColor', ruleColors(i,:), ...
        'MarkerEdgeColor', 'k', ...
        'LineWidth', 1.5, ...
        'DisplayName', sprintf('Center %d', i));
end

xlabel('xTrain');
ylabel('yTrain');
title('Training data and GK Cluster Centers (centersXY)');
grid on;
legend('Location','best');

%% =======================================================
%           LOCAL LS MODEL PREDICTION
%% =======================================================

function normalized_mu = NormaliationOfMu(mu)
    sumMu = sum(mu,1);
    normalized_mu = mu ./sumMu;

end
%% Training data

Ts = 0.05;              % Čas vzorčenja (s)
T_sim = 100;             % Skupni čas simulacije (s)
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

%% DATA PREPERATION

u = u_vec(:);
y = y_vec(:,1);
N = length(u);

% ===== Antecedents (Used ONLY to calculate the weights) =====
% We evaluate the weights at time step k, using current u and current y
u_current = u(3:N); 
y_current = y(3:N);
joint_inputs_for_weights = [u_current, y_current]; % Size: (N-2) x 2

% ===== Consequents (The dynamic linear regressors) =====
u_k   = u(3:N);     % u(k)
u_k1  = u(2:N-1);   % u(k-1)
y_k1  = y(2:N-1);   % y(k-1)
y_k2  = y(1:N-2);   % y(k-2)

% Your true target output to fit against
yTrain_aligned = y(3:N);



%% CALCULATES THE INVERSE DISTANCE FOR X AND Y
% Compute the multi-dimensional distance between the 2D data points and 2D centers
% If using your inverse distance function, ensure it handles 2D coordinates:
distances = pdist2(centersXY, joint_inputs_for_weights); % Returns a (numRules x (N-2)) matrix

% Calculate memberships and normalize them
mu_inv = 1 ./ (distances.^fuzziness + 1e-6);
weights = NormaliationOfMu(mu_inv); % Size: (numRules x (N-2))

numSamples_aligned = length(yTrain_aligned); % This is N-2

% Create the expanded regressor matrix
Xe = [u_k, u_k1, y_k1, y_k2, ones(numSamples_aligned, 1)]; % Size: (N-2) x 5

% Preallocate theta_local (Each rule now needs 5 parameters)
theta_local = zeros(numRules, 5);


%% CALCULATES THE THETA
for i = 1:numRules
    wi = weights(i, :); 
    Wi = diag(wi); 
    
    % Solve the 5-parameter WLS system for Rule i stably
    % Resulting row is: [slope_uk, slope_uk1, slope_yk1, slope_yk2, intercept]
    theta_local(i, :) = (Xe' * Wi * Xe) \ (Xe' * Wi * yTrain_aligned(:));
end

fprintf("Current theta dimention:")
size(theta_local)

%% Prediction test

% 2. Re-form the exact 5-column extended regressor matrix used in training
u_k   = u(3:end);
u_k1  = u(2:end-1);
y_k1  = y(2:end-1);
y_k2  = y(1:end-2);

Xe = [u_k, u_k1, y_k1, y_k2, ones(numSamples_aligned, 1)]; % Size: (N-2) x 5

% Initialize prediction vector for the N-2 steps
y_pred = zeros(numSamples_aligned, 1);

% 3. Compute the blended TS prediction
for i = 1:numRules
    % theta_local(i, :) now contains 5 parameters: 
    % [slope_uk, slope_uk1, slope_yk1, slope_yk2, intercept]
    theta_i = theta_local(i, :);
    
    % Evaluate the local linear model for Rule i across all aligned steps
    y_local = Xe * theta_i';
    
    % Extract the matching membership weights for Rule i (steps 3 to end)
    wi = weights(i,:)';
    
    % Accumulate the weighted contribution of this rule
    y_pred = y_pred + (wi .* y_local);
end

% 4. Evaluate the fit quality using the aligned true output
mse = mean((yTrain_aligned(:) - y_pred).^2);
fprintf('Model Dynamic Prediction MSE: %.6f\n', mse);

% 5. Plot the result vs True Data
figure('Name', 'Dynamic TS Model Prediction vs Actual Data', 'Color', 'w');
plot(yTrain_aligned, 'b', 'LineWidth', 1.5, 'DisplayName', 'True Pendulum Angle');
hold on;
plot(y_pred, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Dynamic TS Model Prediction');
xlabel('Sample index (k)');
ylabel('Angle \phi (rad)');
title('Dynamic Fuzzy Model Output Verification (One-Step-Ahead)');
legend('Location', 'best');
grid on;

%% Free-Run Forecasting for the Entire Simulation

N_sim = length(u); % Total length of your original data
y_forecast = zeros(N_sim, 1);

% 1. Seed the first 2 positions with true data to kickstart the system's memory
y_forecast(1) = y(1);
y_forecast(2) = y(2);

% 2. Dynamic multi-step simulation loop
for k = 3:N_sim
    
    % --- Step A: Grab current and past inputs ---
    uk   = u(k);
    uk1  = u(k-1);
    
    % CRITICAL: Look back at your OWN previous forecasted outputs
    yk1  = y_forecast(k-1); 
    yk2  = y_forecast(k-2);
    
    % --- Step B: Evaluate weights dynamically for step k ---
    % Since weights are computed based on [u(k), y(k)], we use the forecasted y
    current_operating_point = [uk, yk1]; 
    distances_k = pdist2(centersXY, current_operating_point);
    
    mu_k = 1 ./ (distances_k.^fuzziness + 1e-6);
    w_k = mu_k / sum(mu_k); % Normalized weights vector for step k (numRules x 1)
    
    % --- Step C: Build the single-row 5-column regressor vector ---
    Xe_k = [uk, uk1, yk1, yk2, 1]; % Size: 1 x 5
    
    % --- Step D: Blend the local linear rule outputs ---
    yk_pred = 0;
    for i = 1:numRules
        % Evaluate rule i line equation (Xe_k * theta_i')
        y_local_rule = Xe_k * theta_local(i, :).';
        
        % Accumulate the contribution scaled by this rule's weight
        yk_pred = yk_pred + w_k(i) * y_local_rule;
    end
    
    % Save this prediction so step k+1 and k+2 can use it as history
    y_forecast(k) = yk_pred;
end

% 3. Align the vectors to skip the initial seed values for proper error tracking
yTrain_aligned = y(3:end);
y_forecast_aligned = y_forecast(3:end);

% 4. Evaluate forecasting fit quality
mse_forecast = mean((yTrain_aligned(:) - y_forecast_aligned(:)).^2);
fprintf('Model Free-Run Forecast MSE: %.6f\n', mse_forecast);

% 5. Plot the full simulation forecasting response
figure('Name', 'TS Model Free-Run Forecast vs Actual Data', 'Color', 'w');
plot(yTrain_aligned, 'b', 'LineWidth', 1.5, 'DisplayName', 'True Pendulum Angle (Sensor)');
hold on;
plot(y_forecast_aligned, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Fuzzy Free-Run Forecast');
xlabel('Sample index (k)');
ylabel('Angle \phi (rad)');
title('Dynamic Fuzzy Model Output Verification (Free-Run Simulation)');
legend('Location', 'best');
grid on;


%% =======================================================
%           LOCAL LS MODEL Validation
%% =======================================================


Ts = 0.05;              % Čas vzorčenja (s)
T_sim = 100;             % Skupni čas simulacije (s)
N = ceil(T_sim / Ts);   % Število korakov
t_vec = (0:N-1)' * Ts;  % Časovni vektor


TimeOfPeriod = 3;       % Period of alternation (seconds)
uBaseValue = 1.3;         % Baseline value
amplitude = 0.4;        % How much it steps up/down from baseline

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

%% DATA PREPERATION

u = u_vec(:);
y = y_vec(:,1);
N = length(u);

% ===== Antecedents (Used ONLY to calculate the weights) =====
% We evaluate the weights at time step k, using current u and current y
u_current = u(3:N); 
y_current = y(3:N);
joint_inputs_for_weights = [u_current, y_current]; % Size: (N-2) x 2


% Your true target output to fit against
yTrain_aligned = y(3:N);
%% Prediction test

% 2. Re-form the exact 5-column extended regressor matrix used in training
u_k   = u(3:end);
u_k1  = u(2:end-1);
y_k1  = y(2:end-1);
y_k2  = y(1:end-2);

Xe = [u_k, u_k1, y_k1, y_k2, ones(numSamples_aligned, 1)]; % Size: (N-2) x 5

% Initialize prediction vector for the N-2 steps
y_pred = zeros(numSamples_aligned, 1);

% 3. Compute the blended TS prediction
for i = 1:numRules
    % theta_local(i, :) now contains 5 parameters: 
    % [slope_uk, slope_uk1, slope_yk1, slope_yk2, intercept]
    theta_i = theta_local(i, :);
    
    % Evaluate the local linear model for Rule i across all aligned steps
    y_local = Xe * theta_i';
    
    % Extract the matching membership weights for Rule i (steps 3 to end)
    wi = weights(i,:)';
    
    % Accumulate the weighted contribution of this rule
    y_pred = y_pred + (wi .* y_local);
end

% 4. Evaluate the fit quality using the aligned true output
mse = mean((yTrain_aligned(:) - y_pred).^2);
fprintf('Model Dynamic Prediction MSE: %.6f\n', mse);

% 5. Plot the result vs True Data
figure('Name', 'Dynamic TS Model Prediction vs Actual Data', 'Color', 'w');
plot(yTrain_aligned, 'b', 'LineWidth', 1.5, 'DisplayName', 'True Pendulum Angle');
hold on;
plot(y_pred, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Dynamic TS Model Prediction');
xlabel('Sample index (k)');
ylabel('Angle \phi (rad)');
title('Dynamic Fuzzy Model Output Verification (One-Step-Ahead)');
legend('Location', 'best');
grid on;

%% Free-Run Forecasting for the Entire Simulation

N_sim = length(u); % Total length of your original data
y_forecast = zeros(N_sim, 1);

% 1. Seed the first 2 positions with true data to kickstart the system's memory
y_forecast(1) = y(1);
y_forecast(2) = y(2);

% 2. Dynamic multi-step simulation loop
for k = 3:N_sim
    
    % --- Step A: Grab current and past inputs ---
    uk   = u(k);
    uk1  = u(k-1);
    
    % CRITICAL: Look back at your OWN previous forecasted outputs
    yk1  = y_forecast(k-1); 
    yk2  = y_forecast(k-2);
    
    % --- Step B: Evaluate weights dynamically for step k ---
    % Since weights are computed based on [u(k), y(k)], we use the forecasted y
    current_operating_point = [uk, yk1]; 
    distances_k = pdist2(centersXY, current_operating_point);
    
    mu_k = 1 ./ (distances_k.^fuzziness + 1e-6);
    w_k = mu_k / sum(mu_k); % Normalized weights vector for step k (numRules x 1)
    
    % --- Step C: Build the single-row 5-column regressor vector ---
    Xe_k = [uk, uk1, yk1, yk2, 1]; % Size: 1 x 5
    
    % --- Step D: Blend the local linear rule outputs ---
    yk_pred = 0;
    for i = 1:numRules
        % Evaluate rule i line equation (Xe_k * theta_i')
        y_local_rule = Xe_k * theta_local(i, :).';
        
        % Accumulate the contribution scaled by this rule's weight
        yk_pred = yk_pred + w_k(i) * y_local_rule;
    end
    
    % Save this prediction so step k+1 and k+2 can use it as history
    y_forecast(k) = yk_pred;
end

% 3. Align the vectors to skip the initial seed values for proper error tracking
yTrain_aligned = y(3:end);
y_forecast_aligned = y_forecast(3:end);

% 4. Evaluate forecasting fit quality
mse_forecast = mean((yTrain_aligned(:) - y_forecast_aligned(:)).^2);
fprintf('Model Free-Run Forecast MSE: %.6f\n', mse_forecast);

% 5. Plot the full simulation forecasting response
figure('Name', 'TS Model Free-Run Forecast vs Actual Data', 'Color', 'w');
plot(yTrain_aligned, 'b', 'LineWidth', 1.5, 'DisplayName', 'True Pendulum Angle (Sensor)');
hold on;
plot(y_forecast_aligned, 'r--', 'LineWidth', 1.5, 'DisplayName', 'Fuzzy Free-Run Forecast');
xlabel('Sample index (k)');
ylabel('Angle \phi (rad)');
title('Dynamic Fuzzy Model Output Verification (Free-Run Simulation)');
legend('Location', 'best');
grid on;