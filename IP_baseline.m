%% IP type policies comparison and automated tuning for K bins

clc
clear all
close all

% Standard policy combinations:
% IPP   differential
% PRE   ratio

%% Variables
Duration = tic;
ALGORITHM = 1;      % 1 = IPP, 2 = PRE

B = [350];      % range of batch sizes
L = length(B);      % number of batch sizes
K = 8;              % number of modeled bins
N_bins = 10000;     % number of simulated bins per parameter setting

DISTRIBUTION = 2;   % set DISTRIBUTION to 1 for uniform, and 2 for a Normal distribution
ITEM_MIN = 7;       % minimum item weight in uniform distribution
ITEM_MAX = 13;     % maximum item weight in uniform distribution
MU = 100;           % mean item weight in discrete normal distribution
VARIANCE = (MU*0.15)^2;      % variance of item weight in discrete normal distribution

if DISTRIBUTION == 1        % UNIFORM DISTRIBUTION
    M = ITEM_MIN;
    N = ITEM_MAX;
    p = [zeros(1,M-1) 1/(N-M+1) * ones(1,N-M+1)];
elseif DISTRIBUTION == 2    % NORMAL DISTRIBUTION
    p = normpdf((1:2*MU-1),MU,sqrt(VARIANCE));
    p = p./sum(p);      % normalize
end

W = find(p);    % item space
P = length(W);  % number of different item sizes
p_ = p(W);      % probability of items
w_max = W(end);
Pcum = cumsum([0 p]);

MEAN_D = sum(p.*(1:w_max));
STD_D = sqrt(sum(p.*(((1:w_max)-MEAN_D).^2)));
disp([MEAN_D STD_D])

if  ALGORITHM == 1
    parameter = 0.5; % initial parameter value
    stepsize = 0.5; % initial step size value
    nsteps = 9;      % number of steps to take
    G = 3;
elseif ALGORITHM == 2
    parameter = 0.5; % initial parameter value
    stepsize = 0.5; % initial step size value
    nsteps = 9;      % number of steps to take
    G = 2;
end

%% Simulation

S_matrix = zeros(L,1+2*nsteps);     % matrix with parameter values
S_vector = zeros(1,L);
WG_N_matrix = zeros(L,1+2*nsteps);     % matrix with giveaway values
WG_N_vector = zeros(1,L);
N_matrix = zeros(L,1+2*nsteps);     % matrix with number of items allocated
N_vector = zeros(1,L);
WB_N_matrix = zeros(L,1+2*nsteps);     % matrix with total throughput (batched) weight
WB_N_vector = zeros(1,L);

WB_bar_matrix = zeros(L,1+2*nsteps);
WB_bar_vector = zeros(1,L);
WG_bar_matrix = zeros(L,1+2*nsteps);
WG_bar_vector = zeros(1,L);

IP_temp = zeros(1,K);
for b = 1:length(B)  
    
    % initialize algorithm with starting parameter value
    s = 1;
    series = 0:w_max;
    stepsize_ = stepsize;
    parameter_star = parameter;
    
    % set cost function f
    if ALGORITHM == 1
        IP = [zeros(1,B(b)),series.^parameter];                             % IPP
    elseif ALGORITHM == 2
        IP = [zeros(1,B(b)),1 - parameter.^series];                         % PRE
    end

    % recursively calculate indices
    for v = B(b):-1:1
        IP(v) = sum(p.*IP(v+1:v+w_max));
    end

    % generate enough items to fill N_bins bins
    i = 0;
    v_ = zeros(1,K);
    while i < N_bins
        [~,w] = histc(rand,Pcum);
        
        if G == 1                               % greedy policy
            IP_temp = IP(v_+1+w);
        elseif G == 2                           % ratio policy
            IP_temp = IP(v_+1+w)./IP(v_+1);
        elseif G == 3                           % differential policy
            IP_temp = IP(v_+1+w) - IP(v_+1);
        end

        if G == 2                               % check if a tiebreaker is necessary (when dividing by 0)
            if sum(isnan(IP_temp)) > 0
                for k = 1:K
                    IP_temp(k) = IP(v_(k)+1+w) - IP(v_(k)+1);
                end
            end
            k_star = find(IP_temp==min(IP_temp),1);
            v_(k_star) = v_(k_star) + w;
            N_matrix(b,s) = N_matrix(b,s) + 1;
        else
            k_star = find(IP_temp==min(IP_temp),1);
            v_(k_star) = v_(k_star) + w;
            N_matrix(b,s) = N_matrix(b,s) + 1;
        end

        % check if bin is filled and empty if needed
        r = (v_(k_star) - B(b));
        if r >= 0
            WG_N_matrix(b,1) = WG_N_matrix(b,1) + r;
            WB_N_matrix(b,1) = WB_N_matrix(b,1) + B(b);
            v_(k_star) = 0;
            i = i + 1;
        end
    end
    S_matrix(b,1) = parameter;
    WB_bar_matrix(b,1) = WB_N_matrix(b,1)/(WG_N_matrix(b,1) + WB_N_matrix(b,1));
    WG_bar_matrix(b,1) = WG_N_matrix(b,1)/(WG_N_matrix(b,1) + WB_N_matrix(b,1));
    g_star = WG_bar_matrix(b,1);
    
    % Further tuning of the index policy
    for h = 2:nsteps+1 % parameter setting
        parameter_temp = [parameter_star-stepsize^h, parameter_star+stepsize^h];
        S_matrix(b,s+1:s+2) = parameter_temp';
        
        for parameter_ = parameter_temp
            s = s + 1;
            
            % set cost function f
            if ALGORITHM == 1
                IP = [zeros(1,B(b)),series.^parameter_];                                % IPP
            elseif ALGORITHM == 2
                IP = [zeros(1,B(b)),1 - parameter_.^series];                            % PRE
            end

            % recursively calculate indices
            for v = B(b):-1:1
                IP(v) = sum(p.*IP(v+1:v+w_max));
            end

            % generate enough items to fill N_bins bins
            i = 0;
            v_ = zeros(1,K);
            while i < N_bins
                [~,w] = histc(rand,Pcum);

                if G == 1
                    IP_temp = IP(v_+1+w);
                elseif G == 2
                    IP_temp = IP(v_+1+w)./IP(v_+1);
                elseif G == 3
                    IP_temp = IP(v_+1+w) - IP(v_+1);
                end

                if G == 2
                    if sum(isnan(IP_temp)) > 0
                        for k = 1:K
                            IP_temp(k) = IP(v_(k)+1+w) - IP(v_(k)+1);
                        end
                    end
                    k_star = find(IP_temp==min(IP_temp),1);
                    v_(k_star) = v_(k_star) + w;
                    N_matrix(b,s) = N_matrix(b,s) + 1;
                else
                    k_star = find(IP_temp==min(IP_temp),1);
                    v_(k_star) = v_(k_star) + w;
                    N_matrix(b,s) = N_matrix(b,s) + 1;
                end

                r = (v_(k_star) - B(b));
                if r >= 0
                    WG_N_matrix(b,s) = WG_N_matrix(b,s) + r;
                    WB_N_matrix(b,s) = WB_N_matrix(b,s) + B(b);
                    v_(k_star) = 0;
                    i = i + 1;
                end

            end
        end
        parameter_temp = [parameter_star, S_matrix(b,s-1:s)];
        WG_bar_matrix(b,s-1:s) = WG_N_matrix(b,s-1:s)./(WG_N_matrix(b,s-1:s)+WB_N_matrix(b,s-1:s));
        WB_bar_matrix(b,s-1:s) = WB_N_matrix(b,s-1:s)./(WG_N_matrix(b,s-1:s)+WB_N_matrix(b,s-1:s));
        
        g_temp = [g_star, WG_bar_matrix(b,s-1:s)];          % list of w^g
        g_star = g_temp(find(g_temp==min(g_temp),1));               % minimum giveaway of list
        parameter_star = parameter_temp(g_temp==g_star);    % parameter that yielded minimum giveaway
        
        disp([B(b) parameter_star])
    end
    S_temp = S_matrix(b,:);
    N_temp = N_matrix(b,:);
    WG_N_temp = WG_N_matrix(b,:);
    WB_N_temp = WB_N_matrix(b,:);
    WB_bar_temp = WB_bar_matrix(b,:);
    
    WG_bar_vector(b) = min(WG_bar_matrix(b,:));
    S_index = find(WG_bar_matrix(b,:)==WG_bar_vector(b),1); % index of parameter with least giveaway
    S_vector(b) = S_temp(S_index);
    N_vector(b) = N_temp(S_index);
    WG_N_vector(b) = WG_N_temp(S_index);
    WB_N_vector(b) = WB_N_temp(S_index);
    WB_bar_vector(b) = WB_bar_temp(S_index);
end

output = [S_vector
          WB_N_vector
          WG_N_vector
          WB_bar_vector
          WG_bar_vector];
      
output2 = WG_bar_vector;
%WG_vector = WG_vector./N_vector;  % Giveaway per item

%WG_vector
%S_vector
