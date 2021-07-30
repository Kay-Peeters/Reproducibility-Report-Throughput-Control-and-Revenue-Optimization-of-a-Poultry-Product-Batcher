%% IP type policies comparison and tuning for K bins with:
% 1) rejection
% 2) a throughput constraint (fraction of weight batched per product)

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
L = length(B);      % number of batch sizes considered
%AR = 8/10;           % weight per gram batched
AR = 0.5;
AR_ = AR/1e3;       % relative tolerated deviation from the target AR
K = 8;              % number of modeled bins
N_bins = 10000;    % number of simulated bins per parameter setting
R_init = 1e-10;         % initial value of step size constant C
R0 = 0;             % initial value of R

MU = 100;           % mean item weight in discrete normal distribution
VARIANCE = 15^2;      % variance of item weight in discrete normal distribution
p = normpdf([1:2*MU-1],MU,sqrt(VARIANCE));

% MU = 10;
% p = zeros(1,13);
% p(7:13) = 1;

p = p./sum(p);

W = find(p);    % item space
P = length(W);  % number of different item sizes
p_ = p(W);      % probability of items
w_max = W(end);
series = 0:w_max;
Pcum = cumsum([0 p]);

if ALGORITHM == 1
    parameter = 0.5; % initial parameter value
    stepsize = 0.5; % initial step size value
    nsteps = 9;      % number of steps to take
    G = 3;
elseif ALGORITHM == 2
    parameter = 0.5; % initial parameter value
    stepsize = 0.5; % initial step size value
    nsteps = 9; 
    G = 2;
end

%% Simulation
zero_vector = zeros(1,1+2*nsteps);

% rejection data
NA_matrix = cell(1,L);      % number of allocated items
NA_vector = zeros(1,L);
NR_matrix = cell(1,L);      % number of rejected items
NR_vector = zeros(1,L);

WA_N_matrix = cell(1,L);    % total weight put in batches
WA_N_vector = zeros(1,L);
WB_N_matrix = cell(1,L);    % total throughput weight
WB_N_vector = zeros(1,L);
WG_N_matrix = cell(1,L);    % total weight given away
WG_N_vector = zeros(1,L);
WR_N_matrix = cell(1,L);    % total weight rejected
WR_N_vector = zeros(1,L);

WB_bar_matrix = cell(1,L);  % normalized throughput weight
WB_bar_vector = zeros(1,L);
WG_bar_matrix = cell(1,L);  % normalized weight given away
WG_bar_vector = zeros(1,L);
WR_bar_matrix = cell(1,L);  % normalized weight rejected
WR_bar_vector = zeros(1,L);

% parameter and giveaway data
S_matrix = cell(1,L);
S_vector = zeros(1,L);

RR_matrix = cell(1,L);
RR_vector = zeros(1,L);







% 1) set batch size B(b)
% 2) set parameter value R
% 3) iterate over algorithm parameter values for B(b) and R
% Stop when deviating too far from target q

for b = 1:L 
    
    % rejection parameters
    RR = R_init;
    AR_temp = -1;
    ss = 0;
 
    RR_vector_temp = [];
    S_vector_temp = [];
    NA_vector_temp = [];
    NR_vector_temp = [];
    WA_N_vector_temp = [];
    WB_N_vector_temp = [];
    WG_N_vector_temp = [];
    WR_N_vector_temp = [];
    WB_bar_vector_temp = [];
    WG_bar_vector_temp = [];
    WR_bar_vector_temp = [];
        
    %while AR_temp < 0
        
        % initialize algorithm with starting parameter value
        ss = ss + 1;
        S_matrix{b}(ss,:) = zero_vector;
        NA_matrix{b}(ss,:) = zero_vector;
        NR_matrix{b}(ss,:) = zero_vector;
        WA_N_matrix{b}(ss,:) = zero_vector;
        WB_N_matrix{b}(ss,:) = zero_vector;
        WG_N_matrix{b}(ss,:) = zero_vector;
        WR_N_matrix{b}(ss,:) = zero_vector;
        WB_bar_matrix{b}(ss,:) = zero_vector;
        WG_bar_matrix{b}(ss,:) = zero_vector;
        WR_bar_matrix{b}(ss,:) = zero_vector;
        
        %RR = RR/10;
        RR_matrix{b}(ss) = RR;
        
        s = 1;
        parameter_star = parameter;
        
        % set cost function f
        if ALGORITHM == 1
            IP = [zeros(1,B(b)),series.^parameter];                             % IPP
        elseif ALGORITHM == 2
            IP = [zeros(1,B(b)),1 - parameter.^series];                         % PRE
        end        
        
        % recursively calculate indices c
        for v = B(b):-1:1
            IP(v) = sum(p.*IP(v+1:v+w_max));
        end

        % generate enough items to fill N_bins bins
        i = 0;
        R = R0;
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
                IP_min = min(IP_temp);

                if IP_min < R   % ACCEPT
                    k_star = find(IP_temp==min(IP_temp),1);
                    v_(k_star) = v_(k_star) + w;
                end
            else
                IP_min = min(IP_temp);
                if IP_min < R   % ACCEPT
                    k_star = find(IP_temp==IP_min,1);
                    v_(k_star) = v_(k_star) + w;
                end
            end

            if IP_min < R
                GG = max([0,v_(k_star) - B(b)]);
                R = R + GG*RR;                      % giveaway
                R = R - (w - GG)*RR*(1/AR(b) - 1);  % throughput
                NA_matrix{b}(ss,s) = NA_matrix{b}(ss,s) + 1;
                WA_N_matrix{b}(ss,s) = WA_N_matrix{b}(ss,s) + w;

                r = (v_(k_star) - B(b));
                if r >= 0
                    WG_N_matrix{b}(ss,s) = WG_N_matrix{b}(ss,s) + r;
                    WB_N_matrix{b}(ss,s) = WB_N_matrix{b}(ss,s) + B(b);
                    v_(k_star) = 0;
                    i = i + 1;
                end
            else
                R = R + w*RR;                                               % rejected
                NR_matrix{b}(ss,s) = NR_matrix{b}(ss,s) + 1;
                WR_N_matrix{b}(ss,s) = WR_N_matrix{b}(ss,s) + w;
            end
        end
        S_matrix{b}(ss,s) = parameter;
        WB_bar_matrix{b}(ss,s) = WB_N_matrix{b}(ss,s)/(WB_N_matrix{b}(ss,s) + WG_N_matrix{b}(ss,s) + WR_N_matrix{b}(ss,s));
        WG_bar_matrix{b}(ss,s) = WG_N_matrix{b}(ss,s)/(WB_N_matrix{b}(ss,s) + WG_N_matrix{b}(ss,s) + WR_N_matrix{b}(ss,s));
        WR_bar_matrix{b}(ss,s) = WR_N_matrix{b}(ss,s)/(WB_N_matrix{b}(ss,s) + WG_N_matrix{b}(ss,s) + WR_N_matrix{b}(ss,s));
        g_star = WG_bar_matrix{b}(ss,s);  % giveaway per gram as performance measure
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        for h = 2:nsteps+1
            parameter_temp = [parameter_star-stepsize^h, parameter_star+stepsize^h];
            S_matrix{b}(ss,s+1:s+2) = parameter_temp;
            
            for pp = 1:2
                parameter_ = parameter_temp(pp);
                s = s + 1;
                
                % set cost function f
                if ALGORITHM == 1
                    IP = [zeros(1,B(b)),series.^parameter_];                             % IPP
                elseif ALGORITHM == 2
                    IP = [zeros(1,B(b)),1 - parameter_.^series];                         % PRE
                end               

                % recursively calculate indices c
                for v = B(b):-1:1
                    IP(v) = sum(p.*IP(v+1:v+w_max));
                end
                
                % generate enough items to fill N_bins bins
                i = 0;
                R = R0;
                v_ = zeros(1,K);
                while i < N_bins

                    % generate item
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
                        IP_min = min(IP_temp);

                        if IP_min < R   % ACCEPT
                            k_star = find(IP_temp==min(IP_temp),1);
                            v_(k_star) = v_(k_star) + w;
                        end
                    else
                        IP_min = min(IP_temp);
                        if IP_min < R   % ACCEPT
                            k_star = find(IP_temp==IP_min,1);
                            v_(k_star) = v_(k_star) + w;
                        end
                    end

                    % If accepted:
                    if IP_min < R
                        GG = max([0,v_(k_star) - B(b)]);
                        R = R + GG*RR;              % giveaway
                        R = R - (w - GG)*RR*(1/AR(b) - 1);
                        NA_matrix{b}(ss,s) = NA_matrix{b}(ss,s) + 1;
                        WA_N_matrix{b}(ss,s) = WA_N_matrix{b}(ss,s) + w;

                        r = (v_(k_star) - B(b));
                        if r >= 0
                            WG_N_matrix{b}(ss,s) = WG_N_matrix{b}(ss,s) + r;
                            WB_N_matrix{b}(ss,s) = WB_N_matrix{b}(ss,s) + B(b);
                            v_(k_star) = 0;
                            i = i + 1;
                        end
                        
                    % If rejected:
                    else
                        R = R + w*RR;
                        NR_matrix{b}(ss,s) = NR_matrix{b}(ss,s) + 1;
                        WR_N_matrix{b}(ss,s) = WR_N_matrix{b}(ss,s) + w;
                    end
                end
                WB_bar_matrix{b}(ss,s) = WB_N_matrix{b}(ss,s)/(WB_N_matrix{b}(ss,s) + WG_N_matrix{b}(ss,s) + WR_N_matrix{b}(ss,s));
                WG_bar_matrix{b}(ss,s) = WG_N_matrix{b}(ss,s)/(WB_N_matrix{b}(ss,s) + WG_N_matrix{b}(ss,s) + WR_N_matrix{b}(ss,s));
                WR_bar_matrix{b}(ss,s) = WR_N_matrix{b}(ss,s)/(WB_N_matrix{b}(ss,s) + WG_N_matrix{b}(ss,s) + WR_N_matrix{b}(ss,s));
                disp([B(b) RR s])
            end
             
            % save giveaway and parameter values of best and new parameter settings
            g_temp = WG_bar_matrix{b}(ss,s-1:s);
            parameter_temp2 = [parameter_star, S_matrix{b}(ss,s-1:s)];
            
            % find best giveaway and corresponding parameter per batcher
            g_temp_ = [g_star, g_temp];
            s_star = find(g_temp_==min(g_temp_),1);
            g_star = g_temp_(s_star);
            parameter_star = parameter_temp2(s_star);
            
        end
        
        % end of loop evaluation: if giveaway improves and AR within bounds -> update best known giveaway and save
        NA_temp = NA_matrix{b}(ss,:);
        NR_temp = NR_matrix{b}(ss,:);
        S_temp = S_matrix{b}(ss,:);
        WA_N_temp = WA_N_matrix{b}(ss,:);
        WB_N_temp = WB_N_matrix{b}(ss,:);
        WG_N_temp = WG_N_matrix{b}(ss,:);
        WR_N_temp = WR_N_matrix{b}(ss,:);
        WB_bar_temp = WB_bar_matrix{b}(ss,:);
        WG_bar_temp = WG_bar_matrix{b}(ss,:);
        WR_bar_temp = WR_bar_matrix{b}(ss,:);
        
        s_star = find(WG_bar_temp==min(WG_bar_temp),1);
        
        % calculate furthest distance from target throughput
        AR_temp2 = abs(WB_bar_temp - AR(b)) - AR_(b);
        AR_temp2 = AR_temp2(s_star);
        
        % only continue loop if WB is sufficiently close to target
        %if AR_temp2 < 0

            RR_vector_temp = [RR_vector_temp, RR];
            S_vector_temp = [S_vector_temp, S_temp(s_star)];
            NR_vector_temp = [NR_vector_temp, NR_temp(s_star)/(NA_temp(s_star)+NR_temp(s_star))];
            NA_vector_temp = [NA_vector_temp, NA_temp(s_star)/(NA_temp(s_star)+NR_temp(s_star))];

            WA_N_vector_temp = [WA_N_vector_temp, WA_N_temp(s_star)];
            WB_N_vector_temp = [WB_N_vector_temp, WB_N_temp(s_star)];
            WG_N_vector_temp = [WG_N_vector_temp, WG_N_temp(s_star)];
            WR_N_vector_temp = [WR_N_vector_temp, WR_N_temp(s_star)];
            WB_bar_vector_temp = [WB_bar_vector_temp, WB_bar_temp(s_star)];
            WG_bar_vector_temp = [WG_bar_vector_temp, WG_bar_temp(s_star)];
            WR_bar_vector_temp = [WR_bar_vector_temp, WR_bar_temp(s_star)];
            
            disp([RR, WG_bar_vector_temp(ss)])

        % otherwise update AR_temp to break loop   
        %else
        %     AR_temp = AR_temp2;
        %end        
        
    end
    WG_bar_vector(b) = min(WG_bar_vector_temp);
    S_star = find(WG_bar_vector_temp==WG_bar_vector(b));
    
    RR_vector(b) = RR_vector_temp(S_star);
    S_vector(b) = S_vector_temp(S_star);
    NR_vector(b) = NR_vector_temp(S_star);
    NA_vector(b) = NA_vector_temp(S_star);
    WA_N_vector(b) = WA_N_vector_temp(S_star);
    WB_N_vector(b) = WB_N_vector_temp(S_star);
    WG_N_vector(b) = WG_N_vector_temp(S_star);
    WR_N_vector(b) = WR_N_vector_temp(S_star);
    WB_bar_vector(b) = WB_bar_vector_temp(S_star);
    WR_bar_vector(b) = WR_bar_vector_temp(S_star);
%end

output = [S_vector
          RR_vector
          %NA_vector
          %NR_vector
          WB_N_vector
          WG_N_vector
          WR_N_vector
          WB_bar_vector
          WG_bar_vector
          WR_bar_vector];

output2 = [abs(WB_bar_vector-AR) WG_bar_vector]';      
      
%s_star = find(G_vector==min(G_vector),1);
%w_temp = WA_N_vector(s_star) + WR_N_vector(s_star);
%output = [S_vector(s_star), NR_vector(s_star), WA_N_vector(s_star), WG_N_vector(s_star), WR_N_vector(s_star), (WA_N_vector(s_star) - WG_N_vector(s_star))/w_temp, WG_N_vector(s_star)/w_temp, WR_N_vector(s_star)/w_temp];

%G_vector
%S_vector

% s_star = find(G_vector==min(G_vector),1);
% 
% output = [S_vector(s_star) WA_vector(s_star) WG_vector(s_star) WR_vector(s_star) G_vector(s_star)];
% 
% disp('Fraction of weight allocated:')
% disp((WA_vector(s_star)+WG_vector(s_star))/(WA_vector(s_star)+WG_vector(s_star)+WR_vector(s_star)))
% 
% disp('Fraction of weight given away:')
% disp(WG_vector(s_star)/(WA_vector(s_star)+WG_vector(s_star)+WR_vector(s_star)))
% 
% disp('Fraction of weight rejected:')
% disp(WR_vector(s_star)/(WA_vector(s_star)+WG_vector(s_star)+WR_vector(s_star)))


%disp([AR accept/(accept+reject) reject/(accept+reject)])

% %% Plotting
% xmin = min(parameter);
% xmax = max(parameter);
% ymin = min(min(G_vector))*0.9;
% ymax = ymin*1.5;%max(max(G_vector))*1.1;

% figure
% if length(B) == 1
%     plot(parameter,G_matrix)
%     xlabel('alpha')
%     ylabel('giveaway')
% else
%     plot(B,G_vector)
%     xlabel('Target weight')
%     ylabel('Giveaway')
% end

%axis([xmin xmax ymin ymax])
% min_val = min(min(G_vector));
% %min_val = min(min(IP_giveaway_final(3,:)));
% title(['x^{\alpha} (' num2str(min_val,'%1.3g') ')'])

%Duration = toc(Duration)