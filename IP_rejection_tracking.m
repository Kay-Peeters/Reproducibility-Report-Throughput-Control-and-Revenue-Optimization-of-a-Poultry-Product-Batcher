%% IP type policies comparison and tuning for K bins

% write data to file: csvwrite('q_B250_R4.txt',q_plot)

clc
clear all
close all

% Standard policy combinations:
% IP    greedy
% IPP   differential
% PR    ratio
% PRE   ratio

%% Variables
Duration = tic;
ALGORITHM = 1;      % 1 = IPP, 2 = PRE

B = 350;   % range of batch sizes
AR = 0.5;           % fraction of all items accepted
AR_ = AR/1e2;       % relative tolerated deviation from the target AR
K = 8;              % number of modeled bins
N_bins = 10000;    % number of simulated bins per parameter setting

MU = 100;           % mean item weight in discrete normal distribution
VARIANCE = 15^2;      % variance of item weight in discrete normal distribution
p = normpdf([1:2*MU-1],MU,sqrt(VARIANCE));
p = p./sum(p);      % normalize

W = find(p);    % item space
P = length(W);  % number of different item sizes
p_ = p(W);      % probability of items
w_max = W(end);
series = 0:w_max;
Pcum = cumsum([0 p]);

%9 5 1
%10 7 4 1
%10 6 2
temp2 = 1;
if ALGORITHM == 1
    %temp = [0.094726563];  % vector of tuned parameter values per RR setting
    %parameter = temp(temp2);
    parameter = 0.09375;
    G = 3;
elseif ALGORITHM == 2
    temp = [];  % vector of tuned parameter values per RR setting
    parameter = temp(temp2);
    G = 2;
end

%% Simulation
R = 0;
temp = [1e-6];      % vector of used RR settings
RR = temp(temp2);
AR_matrix = 0;  % number of items accepted
RR_matrix = 0;  % number of items rejected
G_matrix = 0;   % total giveaway generated

WB_matrix = 0;  % total throughput
WG_matrix = 0;  % total giveaway
WR_matrix = 0;  % total rejected

IP_temp = zeros(1,K);
R_plot = zeros(1,N_bins*(ceil(B/MU)+1));
WB_plot = R_plot;
WG_plot = R_plot;

% set cost function f
if ALGORITHM == 1
    IP = [zeros(1,B),series.^parameter];                             % IPP
elseif ALGORITHM == 2
    IP = [zeros(1,B),1 - parameter.^series];                         % PRE
end        
        
% recursively calculate indices c
for v = B:-1:1
    IP(v) = sum(p.*IP(v+1:v+w_max));
end

% generate enough items to fill N_bins bins
i = 0;  % number of filled bins
j = 1;  % number of allocated items
v_ = zeros(1,K);
while i < N_bins
    [~,w] = histc(rand,Pcum);

    if G == 1
        IP_temp = IP(v_+1+w);
    elseif G == 2
        IP_temp = IP(v_+1)./IP(v_+1+w);
    elseif G == 3
        IP_temp = IP(v_+1) - IP(v_+1+w);
    end

    if G == 2
        if sum(isnan(IP_temp)) > 0
            IP_temp(k) = IP(v_+1) - IP(v_+1+w);
        end
        IP_max = max(IP_temp);

        if IP_max >= R   % ACCEPT
            k_star = find(IP_temp==IP_max,1);
            v_(k_star) = v_(k_star) + w;
        end
    else
        IP_max = max(IP_temp);
        if IP_max >= R   % ACCEPT
            k_star = find(IP_temp==IP_max,1);
            v_(k_star) = v_(k_star) + w;
        end
    end

    if IP_max >= R       % ACCEPT
        GG = max([0,v_(k_star) - B]);
        R = R - GG*RR;
        R = R + (w - GG)*RR*(1/AR - 1);
        
        WB_matrix = WB_matrix + (w - GG);
        WG_matrix = WG_matrix + GG;
        
        AR_matrix = AR_matrix + 1;

        r = (v_(k_star) - B);
        if r >= 0
            G_matrix = G_matrix + r;
            v_(k_star) = 0;
            i = i + 1;
        end
    else                % REJECT
        R = R - w*RR;
        WR_matrix = WR_matrix + w;
        RR_matrix = RR_matrix + 1;
    end
    j = j + 1;
    R_plot(j) = R;
    %q_plot(j) = AR_matrix/j;
    WB_plot(j) = WB_matrix/(WB_matrix + WG_matrix + WR_matrix);
    WG_plot(j) = WG_matrix/(WB_matrix + WG_matrix + WR_matrix);
end
g_star = G_matrix/AR_matrix;  % giveaway per item as performance measure
disp(g_star)

j = 1000;

R_plot = R_plot(1:j);
WB_plot = WB_plot(1:j);
WG_plot = WG_plot(1:j);

output = [WB_plot', WG_plot', R_plot'];

figure
subplot(1,3,1)
hold on
plot(WB_plot)
plot([1 j],[AR AR],'r--')
axis([0 length(WB_plot) AR-0.05 AR+0.05])
xlabel('Item #')
ylabel('WB values')

subplot(1,3,2)
hold on
plot(WG_plot)
plot([1 j],[AR AR],'r--')
axis([0 length(WG_plot) 0 0.05])
xlabel('Item #')
ylabel('WG values')

subplot(1,3,3)
hold on
plot(R_plot)
xlabel('Item #')
ylabel('R values')