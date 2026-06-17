%%
close all
clear
clc

pendulum_process_obf([], 0); 

Ts = 0.01;              % Čas vzorčenja (s)
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

