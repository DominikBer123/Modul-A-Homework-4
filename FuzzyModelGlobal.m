%%
close all
clear
clc

pendulum_process_obf([], 0); 

Ts = 0.05;              % Čas vzorčenja (s)
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

%%
figure('Name', 'Simulacija nihala', 'Color', 'w');

subplot(2, 1, 1);
plot(t_vec(50:end), u_vec(50:end)', 'LineWidth', 1.2);
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
plot(u_vec(50:end),rad2deg(y_vec(50:end,1)) ,"*")
hold on;
yline(10, 'r--', 'Spodnja meja (10°)');
yline(100, 'r--', 'Zgornja meja (100°)');
ylabel('Kot (°)');
title('Izhodni kot');
grid on

%% GK Optimization Loop (Updating given variables directly)

xTrain = u_vec(50:end)';
yTrain = y_vec(50:end,1);

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

% Use the same fuzziness parameter from GK algorithm
fuzziness = 2.0;  % Make sure this matches your GK fuzziness value

numRules = size(centersXY, 1);  % Get number of rules from centersXY
numSamples = length(xTrain);

% Inverse distance memberships (unnormalized)
muTrain = zeros(numRules, numSamples);

for i = 1:numRules
    % Calculate distance between each point and the center
    distances = abs(xTrain - centers(i));
    
    % Inverse distance with fuzziness parameter
    % Add small epsilon to avoid division by zero
    muTrain(i,:) = 1 ./ (distances.^fuzziness + 1e-10);
end

% Normalize memberships (so they sum to 1 across rules)
phiTrain = muTrain ./ sum(muTrain, 1);

% ===== PLOT memberships =====

figure('Name','Train Memberships (Inverse Distance)','Color','w');
hold on;

ruleColors = lines(numRules);

for i = 1:numRules
    plot(xTrain, phiTrain(i,:), 'LineWidth', 1.2, ...
        'Color', ruleColors(i,:), ...
        'DisplayName', sprintf('Rule %d', i));
end

xlabel('xTrain');
ylabel('Membership degree');
title('Normalized Inverse Distance Memberships (Training Set)');
grid on;
legend('Location','best');

% ===== Hard assignment from memberships =====
[~, hardAssign] = max(phiTrain, [], 1);
ruleColors = lines(numRules);

% ===== Plot xTrain vs yTrain colored by rule =====
figure('Name','Training data colored by membership (Inverse Distance)','Color','w');
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
title('Training data and GK Cluster Centers (Inverse Distance)');
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
uBaseValue = 1.7;         % Baseline value
amplitude = 0.6;        % How much it steps up/down from baseline

u_vec = zeros(N,1);
numSteps = ceil(t_vec(end) / TimeOfPeriod);

for k = 1:numSteps

    idx = (t_vec >= (k-1)*TimeOfPeriod) & (t_vec < k*TimeOfPeriod);

    stepNoise = (2*rand - 1) * amplitude;

    u_vec(idx) = uBaseValue + stepNoise;

end

pendulum_process_obf([], 0);

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

%% GLOBAL ORDINARY LEAST SQUARES (OLS) THETA ESTIMATION

numSamples_aligned = length(yTrain_aligned); % N-2
numParamsPerRule = 5;                        % [u_k, u_k1, y_k1, y_k2, 1]

% 1. Preallocate the wide global regressor matrix 
% Size: (N-2) x (5 * numRules)
X_global = zeros(numSamples_aligned, numParamsPerRule * numRules);

% 2. Build the global matrix by weighting the regressors for each rule
for i = 1:numRules
    % Extract membership weights for Rule i as a column vector (Size: (N-2) x 1)
    wi = weights(i, :)'; 
    
    % Determine the column indices for rule i in the global matrix
    col_start = (i - 1) * numParamsPerRule + 1;
    col_end   = i * numParamsPerRule;
    
    % Element-wise multiply the entire Xe matrix by this rule's weight vector
    X_global(:, col_start:col_end) = wi .* Xe;
end

% 3. Solve the Global OLS problem using the standard backslash operator
% This finds all global parameters simultaneously
theta_global_vector = X_global \ yTrain_aligned(:); % Returns a vector of size (25 x 1)

% 4. Reshape back into a clean (numRules x 5) matrix to match your local structure
% Each row i contains: [slope_uk, slope_uk1, slope_yk1, slope_yk2, intercept]
theta_global = reshape(theta_global_vector, numParamsPerRule, numRules)';

fprintf('\nGlobal theta dimensions:\n');
disp(size(theta_global)); % Should output: [numRules, 5]

%% Prediction test (GLOBAL OLS)
% 2. Re-form the exact 5-column extended regressor matrix used in training
u_k   = u(3:end);
u_k1  = u(2:end-1);
y_k1  = y(2:end-1);
y_k2  = y(1:end-2);
Xe = [u_k, u_k1, y_k1, y_k2, ones(numSamples_aligned, 1)]; % Size: (N-2) x 5

% Initialize prediction vector for the N-2 steps
y_pred_global = zeros(numSamples_aligned, 1);

% 3. Compute the blended TS prediction using global parameters
for i = 1:numRules
    % FIX: Extract row parameters from theta_global instead of theta_local
    theta_i = theta_global(i, :);
    
    % Evaluate the local linear model for Rule i across all aligned steps
    y_local = Xe * theta_i';
    
    % Extract the matching membership weights for Rule i
    wi = weights(i,:)';
    
    % Accumulate the weighted contribution of this rule
    y_pred_global = y_pred_global + (wi .* y_local);
end

% 4. Evaluate the fit quality using the aligned true output
mse_global_osa = mean((rad2deg(yTrain_aligned(:)) - rad2deg(y_pred_global)).^2);
fprintf('Global Model Dynamic Prediction MSE (deg): %.6f\n', mse_global_osa);

% 5. Plot the result vs True Data
figure('Name', 'Global Dynamic TS Model Prediction vs Actual Data', 'Color', 'w');
plot(yTrain_aligned, 'b', 'LineWidth', 1.5, 'DisplayName', 'True Pendulum Angle');
hold on;
plot(y_pred_global, 'g--', 'LineWidth', 1.5, 'DisplayName', 'Global TS Model Prediction');
xlabel('Sample index (k)');
ylabel('Angle \phi (rad)');
title('Global Fuzzy Model Output Verification (One-Step-Ahead)');
legend('Location', 'best');
grid on;


%% Free-run Forecast (GLOBAL OLS)

% Initial conditions (use measured values for the first two samples)
y_forecast = zeros(length(y),1);

y_forecast(1) = y(1);
y_forecast(2) = y(2);

% Forecast from k = 3 onward
for k = 3:length(y)

    % Construct regressor using PREVIOUS PREDICTIONS
    xe = [ ...
        u(k), ...
        u(k-1), ...
        y_forecast(k-1), ...
        y_forecast(k-2), ...
        1 ];

    yk = 0;

    % TS model output
    for i = 1:numRules

        theta_i = theta_global(i,:);

        % Local model output
        y_local = xe * theta_i';

        % Membership weight at current sample
        wi = weights(i,k-2);

        % Weighted contribution
        yk = yk + wi * y_local;
    end

    y_forecast(k) = yk;

end

% Remove first two samples for comparison
y_forecast_aligned = y_forecast(3:end);

% MSE
mse_global_forecast = mean((rad2deg(yTrain_aligned(:)) - rad2deg(y_forecast_aligned(:))).^2);
fprintf('Global Model Forecast MSE (deg): %.6f\n', mse_global_forecast);

% Plot
figure('Name','Global TS Model Forecast','Color','w');

plot(yTrain_aligned,'b','LineWidth',1.5,...
    'DisplayName','True Pendulum Angle');
hold on;

plot(y_forecast_aligned,'r--','LineWidth',1.5,...
    'DisplayName','Forecast');

xlabel('Sample index (k)');
ylabel('Angle \phi (rad)');
title('Global Fuzzy Model Free-Run Forecast');
legend('Location','best');
grid on;


%% =======================================================
%           LOCAL LS MODEL Validation
%% =======================================================


Ts = 0.05;              % Čas vzorčenja (s)
T_sim = 100;             % Skupni čas simulacije (s)
N = ceil(T_sim / Ts);   % Število korakov
t_vec = (0:N-1)' * Ts;  % Časovni vektor


TimeOfPeriod = 3;       % Period of alternation (seconds)
uBaseValue = 1.2;         % Baseline value
amplitude = 0.3;        % How much it steps up/down from baseline

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

figure

plot(t_vec,rad2deg(y_vec(:,1)))

%% DATA PREPERATION

u = u_vec(:);
y = y_vec(:,1);
N = length(u);

% ===== Antecedents =====
u_current = u(3:N); 
y_current = y(3:N);
joint_inputs_for_weights = [u_current, y_current]; 

% Your true target output to fit against
yTrain_aligned = y(3:N);
numSamples_aligned = length(yTrain_aligned);

% === ADD THIS FIX: Recalculate weights for validation data ===
distances = pdist2(centersXY, joint_inputs_for_weights); 
mu_inv = 1 ./ (distances.^fuzziness + 1e-6);
weights = NormaliationOfMu(mu_inv); 
% =============================================================

%% Prediction test (GLOBAL OLS)
% 2. Re-form the exact 5-column extended regressor matrix used in training
u_k   = u(3:N);     % u(k)
u_k1  = u(2:N-1);   % u(k-1)
y_k1  = y(2:N-1);   % y(k-1)
y_k2  = y(1:N-2);   % y(k-2)


Xe = [u_k, u_k1, y_k1, y_k2, ones(numSamples_aligned, 1)]; % Size: (N-2) x 5

% Initialize prediction vector for the N-2 steps
y_pred_global = zeros(numSamples_aligned, 1);

% 3. Compute the blended TS prediction using global parameters
for i = 1:numRules
    % FIX: Extract row parameters from theta_global instead of theta_local
    theta_i = theta_global(i, :);
    
    % Evaluate the local linear model for Rule i across all aligned steps
    y_local = Xe * theta_i';
    
    % Extract the matching membership weights for Rule i
    wi = weights(i,:)';
    
    % Accumulate the weighted contribution of this rule
    y_pred_global = y_pred_global + (wi .* y_local);
end

% 4. Evaluate the fit quality using the aligned true output
mse_global_osa = mean((rad2deg(yTrain_aligned(:)) - rad2deg(y_pred_global)).^2);
fprintf('Global Model Dynamic Prediction MSE (deg): %.6f\n', mse_global_osa);

% 5. Plot the result vs True Data
figure('Name', 'Global Dynamic TS Model Prediction vs Actual Data', 'Color', 'w');
plot(yTrain_aligned, 'b', 'LineWidth', 1.5, 'DisplayName', 'True Pendulum Angle');
hold on;
plot(y_pred_global, 'g--', 'LineWidth', 1.5, 'DisplayName', 'Global TS Model Prediction');
xlabel('Sample index (k)');
ylabel('Angle \phi (rad)');
title('Global Fuzzy Model Output Verification (One-Step-Ahead)');
legend('Location', 'best');
grid on;

%% Free-run Forecast (GLOBAL OLS)

% Initial conditions (use measured values for the first two samples)
y_forecast = zeros(length(y),1);

y_forecast(1) = y(1);
y_forecast(2) = y(2);

% Forecast from k = 3 onward
for k = 3:length(y)

    % Construct regressor using PREVIOUS PREDICTIONS
    xe = [ ...
        u(k), ...
        u(k-1), ...
        y_forecast(k-1), ...
        y_forecast(k-2), ...
        1 ];

    yk = 0;

    % TS model output
    for i = 1:numRules

        theta_i = theta_global(i,:);

        % Local model output
        y_local = xe * theta_i';

        % Membership weight at current sample
        wi = weights(i,k-2);

        % Weighted contribution
        yk = yk + wi * y_local;
    end

    y_forecast(k) = yk;

end

% Remove first two samples for comparison
y_forecast_aligned = y_forecast(3:end);

% MSE
mse_global_forecast = mean((rad2deg(yTrain_aligned(:)) - rad2deg(y_forecast_aligned(:))).^2);
fprintf('Global Model Forecast MSE (deg): %.6f\n', mse_global_forecast);

% Plot
figure('Name','Global TS Model Forecast','Color','w');

plot(yTrain_aligned,'b','LineWidth',1.5,...
    'DisplayName','True Pendulum Angle');
hold on;

plot(y_forecast_aligned,'r--','LineWidth',1.5,...
    'DisplayName','Forecast');

xlabel('Sample index (k)');
ylabel('Angle \phi (rad)');
title('Global Fuzzy Model Free-Run Forecast');
legend('Location','best');
grid on;