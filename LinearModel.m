% Primer uporabe funkcije pendulum_process.m
close all
clear
% 1. Ponastavitev notranjega stanja procesa
pendulum_process_obf([], 0); 

% 2. Nastavitve simulacije
Ts = 0.1;              % Čas vzorčenja (s)
T_sim = 100;             % Skupni čas simulacije (s)
N = ceil(T_sim / Ts);   % Število korakov
t_vec = (0:N-1)' * Ts;  % Časovni vektor

% 3. Generiranje vhodnega signala vlak impulzov
%u_vec = (sin(t_vec) > 0) + 1.2;

NumberOfSteps = 10;
SamplesPerStep = N/NumberOfSteps;
stepSize = 0.3;
u_vec = zeros(N,1);

for stepCount = 1:NumberOfSteps

    u_vec((stepCount-1)*SamplesPerStep + 1:stepCount*SamplesPerStep) = stepCount*stepSize; 

end


% 4. Simulacija procesa v vektorskem načinu
% Ker pendulum_process sprejema vektorske vhode, lahko posredujemo u_vec neposredno.
disp('Začetek simulacije...');

% Klic procesa z vhodnim vektorjem, in vzorčno frekvenco Ts
y_vec = pendulum_process_obf(u_vec, Ts);

disp('Simulacija končana.');

% 5. Analiza in prikaz rezultatov
kot_rad = y_vec(:, 1);
hitrost_rad_s = y_vec(:, 2);
kot_stopinje = kot_rad * 180 / pi;

figure('Name', 'Simulacija nihala', 'Color', 'w');

subplot(2, 1, 1);
plot(t_vec, u_vec, 'LineWidth', 1.2);
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
figure
plot(u_vec,rad2deg(y_vec(:,1)) ,"*")
hold on;
yline(10, 'r--', 'Spodnja meja (10°)');
yline(100, 'r--', 'Zgornja meja (100°)');
ylabel('Kot (°)');
title('Izhodni kot');
grid on

%% Plot Step repsonse:

figure
plot(u_vec,rad2deg(y_vec(:,1)),"*")
grid on;
xlabel('Angle');
ylabel('U Response');
hold on;
yline(10, 'r--', 'Spodnja meja (10°)');
yline(100, 'r--', 'Zgornja meja (100°)');

yline(20, '--', 'Spodnja delovna meja (20°)', 'Color', [0, 0.5, 0]);
yline(80, '--', 'Zgornja delovna meja (80°)', 'Color', [0, 0.5, 0]);
grid on;

% Moje delovno območje je med U vrednostjo 0.9 in 

%% System identification 

% Reset
pendulum_process_obf([], 0); 

Ts = 0.07;              % Čas vzorčenja (s)
T_sim = 100;             % Skupni čas simulacije (s)
N = ceil(T_sim / Ts);   % Število korakov
t_vec = (0:N-1)' * Ts;  % Časovni vektor


TimeOfPeriod = 1.2;       % Period of alternation (seconds)
uBaseValue = 1.3;         % Baseline value
amplitude = 0.4;        % How much it steps up/down from baseline

u_vec = zeros(N,1);
numSteps = ceil(t_vec(end) / TimeOfPeriod);

%for k = 1:N
%    phase = mod(t_vec(k), TimeOfPeriod);

%    if phase < TimeOfPeriod/2
%        noise = (2*rand - 1) * amplitude;   % [-amplitude, +amplitude]
%        u_vec(k) = uBaseValue + noise;
%    else
%        noise = (2*rand - 1) * amplitude;
%        u_vec(k) = uBaseValue - noise;
%    end
%end



for k = 1:numSteps

    idx = (t_vec >= (k-1)*TimeOfPeriod) & (t_vec < k*TimeOfPeriod);

    % random amplitude for THIS step only
    stepNoise = (2*rand - 1) * amplitude;

    % assign constant value over the whole interval
    u_vec(idx) = uBaseValue + stepNoise;

end


disp('Začetek simulacije...');

% Klic procesa z vhodnim vektorjem, in vzorčno frekvenco Ts
y_vec = pendulum_process_obf(u_vec, Ts);


disp('Simulacija končana.');


% 5. Analiza in prikaz rezultatov
kot_rad = y_vec(:, 1);
hitrost_rad_s = y_vec(:, 2);
kot_stopinje = kot_rad * 180 / pi;

figure('Name', 'Simulacija nihala', 'Color', 'w');

subplot(2, 1, 1);
plot(t_vec, u_vec, 'LineWidth', 1.2);
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


% Linear model theta

y_train = y_vec(3:end,1);

x_train = [ ...
    y_vec(2:end-1,1), ...
    y_vec(1:end-2,1), ...
    u_vec(2:end-1), ...
    u_vec(1:end-2)];

theta = x_train \ y_train;

y_hat = x_train * theta;


figure;
plot(rad2deg(y_train),'b','LineWidth',1.5);
hold on;
plot(rad2deg(y_hat),'r--','LineWidth',1.5);
grid on;

xlabel('Sample k');
ylabel('Output');
title('Measured vs Predicted Output');
legend('Measured y','Predicted \\hat{y}','Location','best');


%

function y_hat = predictModel(theta, u, y_init)
    N = length(u);
    y_hat = zeros(N, 1);
    
    % Initialize the first two steps with the actual measured positions
    y_hat(1) = y_init(1);
    y_hat(2) = y_init(2);
    
    % Run the free-run (IIR) simulation
    for k = 3:N
        x = [ ...
            y_hat(k-1);
            y_hat(k-2);
            u(k-1);
            u(k-2)];
        y_hat(k) = x' * theta;
    end
end

% --- VALIDATION TEST ---             
T_sim = 30;             
N = ceil(T_sim / Ts);   
t_vec = (0:N-1)' * Ts;  
u_vec = (sin(t_vec) > 0) * amplitude+ uBaseValue; % Square wave around your operating point

fprintf('\nZačetek verifikacijske simulacije...');
pendulum_process_obf([], 0); % Reset process
y_vec = pendulum_process_obf(u_vec, Ts);
fprintf('\nSimulacija končana.');

% 1. Pass the true initial measurements (y_vec(1:2)) to the prediction function
y_hat = predictModel(theta, u_vec, y_vec(:,1));

% 2. Convert both to degrees for plotting and analysis
y_true_deg = rad2deg(y_vec(:, 1));
y_pred_deg = rad2deg(y_hat);

% Plot the entire horizon
figure;
plot(t_vec, y_true_deg, 'b', 'LineWidth', 1.5); hold on;
plot(t_vec, y_pred_deg, 'r--', 'LineWidth', 1.5);
grid on;
xlabel('Time (s)');
ylabel('Output y (°)');
legend('True process output', 'Free-run Model prediction');
title('Model Validation (Free-Run Simulation)');

% Accurate RMSE Calculation
rmse = sqrt(mean((y_true_deg - y_pred_deg).^2));
fprintf('\n\nValidation RMSE = %.4f degrees\n\n', rmse);




%% --- MULTI-STEP VALIDATION TEST ---             
T_sim = 30;             
N = ceil(T_sim / Ts);   
t_vec = (0:N-1)' * Ts;  

% Define 3 different step values to test
% (Adjust these based on your baseline, e.g., low, medium, high torque)
step_values = [1.3, 1.6, 1.9]; 

% Create a figure to display the 3 tests side-by-side
figure('Name', 'Step Response Comparison', 'Color', 'w', 'Position', [100, 100, 1200, 400]);

for i = 1:3
    current_u = step_values(i);
    
    % Create a constant step input vector
    u_vec = ones(N, 1) * current_u;
    
    % 1. Run the actual physical non-linear process
    disp(['Začetek simulacije za u = ', num2str(current_u), '...']);
    pendulum_process_obf([], 0); % Reset internal state
    y_vec = pendulum_process_obf(u_vec, Ts);
    
    % 2. Run your free-run linear model prediction
    y_hat = predictModel(theta, u_vec, y_vec(:,1));
    
    % 3. Convert outputs to degrees
    y_true_deg = rad2deg(y_vec(:, 1));
    y_pred_deg = rad2deg(y_hat);
    
    % 4. Plot the comparison
    subplot(1, 3, i);
    plot(t_vec, y_true_deg, 'b', 'LineWidth', 1.5); hold on;
    plot(t_vec, y_pred_deg, 'r--', 'LineWidth', 1.5);
    
    % Calculate steady-state values (average of last 5 seconds)
    ss_idx = t_vec > (T_sim - 5);
    ss_true = mean(y_true_deg(ss_idx));
    ss_pred = mean(y_pred_deg(ss_idx));
    
    % Visual indicators for where they stabilize
    yline(ss_true, 'b:', ['Real: ', num2str(ss_true, '%.1f'), '°']);
    yline(ss_pred, 'r:', ['Model: ', num2str(ss_pred, '%.1f'), '°']);
    
    grid on;
    xlabel('Time (s)');
    ylabel('Output y (°)');
    title(['Step Input U = ', num2str(current_u)]);
    if i == 1
        legend('True Process', 'Linear Model', 'Location', 'best');
    end
    ylim([20, 90]); % Adjusted to capture your working limits
end
disp('Vse simulacije končane.');











