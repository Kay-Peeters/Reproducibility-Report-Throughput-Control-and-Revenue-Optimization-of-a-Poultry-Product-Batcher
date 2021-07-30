%% 1-bounded bin-covering problem with rejection solved through Value Iteration
% Note: number of modeled bins is fixed (1)
% Note: actions are allocating the item to the bin or rejecting it
% Note: the considered problem is unconstrained

% NOTE: alternative reward function formulation (r=B if action closes)

clc
clear all
close all

%% Variables
tic

Bmin = 350;
Bmax = 350;
B = Bmin:Bmax;      % batch size
epsilon = 1e-6;     % VI and G-S stopping criterion
r_B = 0.8;            % normalized revenue of rejected weight per gram
plotting = 0;

DISTRIBUTION = 2;   % type of distribution used
ITEM_MIN = 1;       % minimum item weight
ITEM_MAX = 19;      % maximum item weight
MU = 100;             % mean item weight in discrete normal distribution5
VARIANCE = (MU*0.15)^2;       % variance of item weight in discrete normal distribution
if DISTRIBUTION == 1
    M = ITEM_MIN;
    N = ITEM_MAX;
    p = [zeros(1,M-1) 1/(N-M+1) * ones(1,N-M+1)];
elseif DISTRIBUTION == 2
    p = normpdf([1:2*MU-1],MU,sqrt(VARIANCE));
        %p = p.*(p>1/1e3);
    %p(1:floor(MU-1-4*sqrt(VARIANCE))) = 0;
    %p(ceil(MU+1+4*sqrt(VARIANCE)):end) = 0;
    p = p./sum(p);
elseif DISTRIBUTION == 3
    p = normpdf([1:2*MU-1],MU,sqrt(VARIANCE));
    p = p./sum(p);
    p = max(p) - 0.9*p;
    p = p./sum(p);
elseif DISTRIBUTION == 4
    p = 1:ITEM_MAX;
    p = p(end:-1:1);
    p = p./sum(p);
end
W = find(p);        % item space
P_ = length(W);      % number of item sizes
A = 1:2;            % action space. a=1: reject. a=2: accept.
K = length(A);      % size of the action space
p_ = p(W);          % probability of items
w_max = W(end);

MU_D = sum(p_.*W);
VARIANCE_D = sum(p_.*(W - MU).^2);

disp([MU VARIANCE MU_D VARIANCE_D])

Revenue = zeros(1,length(B));
e_0 = zeros(1,K);
r_0 = zeros(1,K);
V_0 = zeros(1,K);
Revenue_gram = zeros(1,length(B));
for b = 1:length(B)
    % Determine reachable states S & index number of each state

    S = zeros(1,B(b)-1);
    S(W(W<B(b))) = 1;
    W_temp = W;
    for i = 1:floor(B(b)/W(1))
        W_temp = unique(sum(combvec(W_temp,W)));
        for j = 1:length(W_temp)
            if W_temp(j) < B(b)
                S(W_temp(j)) = 1;
            end
        end
    end
    S = [0 find(S)];                    % single bin state space
    %SS = P_*nchoosek(length(S)+K-1,K);   % state space size
    SS = P_*length(S);
    
    % index numbers for bins (to eliminate unused bin levels)
    index_S = zeros(1,B(b));
    s = 1;
    for v = 1:length(S)
        index_S(S(v)+1) = s;
        s = s+1;
    end

    % index numbers (multi -> single dimension array)
    %index = zeros(length(S),length(W));   % index of states
    index = zeros(S(end)+1,W(end));
    S_vector = zeros(SS,2);                         % state space (v1,v2,w)
    s = 1;
    for v = 1:length(S)
        for w = 1:length(W)
            S_vector(s,:) = [S(v) W(w)];
            index(S(v)+1,W(w)) = s;
            s = s+1;
        end
    end

    %% Value Iteration
    %T_MDP = tic;

    span = 10*epsilon;       % convergence value
    h = 2;          % iteration counter
    V0 = zeros(SS,1);
    V{1} = V0;      % value matrix initialization
    A_star = V0;
    while span > epsilon
        V{h} = V0;

        for s = 1:SS
            r_temp = r_0;
            V_temp = V_0;
            
            v = S_vector(s,1);
            w = S_vector(s,2);
        
            % Reject (index 1)
            r_temp(1) = r_B * w;
            s_ = index(v+1,W(1));
            V_temp(1) = p_*V{h-1}(s_:s_+P_-1);
        
            % Accept (index 2)
            v_ = v + w;
            r = v_ - B(b);
            %r_temp(2) = w;
            if r >= 0
                r_temp(2) = B(b);
                v_ = 0;
            end
            s_ = index(v_+1,W(1));
            V_temp(2) = p_*V{h-1}(s_:s_+P_-1);
        
            % Select action that maximizes reward
            V_sum = r_temp + V_temp;
            if V_sum(1) >= V_sum(2)
                V{h}(s) = V_sum(1);
                A_star(s) = 1;
            else
                V{h}(s) = V_sum(2);
                A_star(s) = 2;
            end            
        end
        
        m = min(V{h} - V{h-1});
        M = max(V{h} - V{h-1});
        span = (M-m)/m;
        h = h + 1;
    end
    Revenue_gram(b) = (M+m)/(2*MU);
    
    
    % Translate state values to matrix form
    h = h - 1;
    V_matrix = zeros(length(S),P_)';
    
    plotdata = cell(1,K);
    for v = 1:length(S)
        for w = 1:length(W)
            s = index(S(v)+1,W(w));
            V_matrix(w,v) = V{h}(s); % - V{h-1}(s);
            
            if A_star(s) == 1
                plotdata{1} = [plotdata{1}; S_vector(s,:)];
            else
                plotdata{2} = [plotdata{2}; S_vector(s,:)];
            end
            
%             v_ = S(v) + W(w);
%             g_ = v_ - B(b);
%             r_ = W(w);
%             if g_ >= 0
%                 r_ = r_ - g_;
%                 v_ = 0;
%             end
%             
%             r1 = (W(w)*r_B + p_*V_matrix(:,v));            % rejection value
%             r2 =       (r_ + p_*V_matrix(:,index_S(v_+1)));   % accept value
%             disp([W(w)*r_B, r_])
%             disp([p_*V_matrix(:,v), p_*V_matrix(:,index_S(v_+1))])
%             if r1 > r2
%                 %A_star(s) = 1;
%             else
%                 %A_star(s) = 2;
%             end
        end
    end

%% Plotting
if plotting == 1
    % Value function
    figure
    surf(S,W,V_matrix)
    xlabel('Bin level')
    ylabel('Item weight')
    zlabel('State value')
    
%     V_x = size(V_matrix,1);
%     V_y = size(V_matrix,2);
%     V_temp = zeros(V_x,V_y+1);
%     V_temp(1:V_x,1:V_y) = V_matrix+1;
%     V_temp(1:V_x,2:V_y+1) = V_temp(1:V_x,2:V_y+1) - V_matrix;
%     V_temp = V_temp(:,2:end-1);
    
%     figure
%     surf(S(2:end),W,V_temp)
%     xlabel('Bin level')
%     ylabel('Item weight')
%     zlabel('V(v+1,w) + 1 - V(v,w) >= 0')
    
    % Optimal actions
    plotmarks = {'rx','bo'};
    figure
    hold on
    for k = 1:K
        scatter(plotdata{k}(:,1),plotdata{k}(:,2),1500/length(S),plotmarks{k});
    end
    xlabel('Bin level v')
    ylabel('Weight w')
    axis([0 S(end) W(1) W(end)])
    title('Optimal action per state (v,w)')
    h = legend('Reject','Accept');
    set(h, 'Location', 'NorthWest')
    
%     figure
%     plot(S,p_*V_matrix)
%     xlabel('Bin level v')
%     ylabel('Expected bin level value V(v)')
end
    %% Calculation of the equilibrium distribution
    % S_vector: state space
    % S2_vector: indices of states AFTER allocation
    % A_star_vector: list of optimal actions
    % R_vector: list of costs incurred (only >0 if optimal action closes bin)
    % C_vector: list of actions that close bins (1)
    % P_matrix: transition probabilities between states for the optimal policy
    % E_vector: equilibrium distribution
    % A_matrix: transformed probability matrix for G-S iterations

%     A_star_vector = zeros(SS,1);
%     s = 1;
%     for v1 = 1:length(S)
%         for v2 = v1:length(S)
%             for w = 1:length(W)
%                 A_star_vector(s) = A_star(v1,v2,w);
%                 s = s + 1;
%             end
%         end
%     end


%     R_vector = zeros(SS,1);
%     C_vector = zeros(SS,1);
% 
%     S_input = repmat(1:SS,P_,1);
%     S_input = S_input(:);
%     P_input = zeros(SS*P_,1);
%     p_input = repmat(p_',SS,1);
%     for s = 1:SS
%         v = S_vector(s,1);
%         w = S_vector(s,2);
%         a = A_star(s);
%         
%         if a == 1
%             v_ = v;
%             R_vector(s) = r_B * w;
%         else
%             v_ = v + w;
%             r = (v_ - B(b));
%             if r >= 0
%                 R_vector(s) = w - r;
%                 v_ = 0;
%             else
%                 R_vector(s) = w;
%             end
%         end
% 
%         % Derive P matrix
%         S_index = index(v_+1,W(1));
%         P_input((s-1)*P_+1:s*P_) = S_index:S_index+P_-1;
%     end
%     P_matrix = sparse(S_input,P_input,p_input,SS,SS,SS*P_);
% 
%     clear x
%     x{1} = (1/SS)*ones(1,SS);
%     i = 1;
%     span = 1;
%     while span > epsilon
%         i = i + 1;
%         x{i} = x{i-1}*P_matrix;
%         span = sum(abs(x{i} - x{i-1}))/sum(abs(x{i}));
%     end
%     Revenue(b) = x{end}*R_vector; 
%     %B(b)
    
end

%Revenue_gram = Revenue/MU;
disp(['Optimal revenue per gram is: ', num2str(Revenue_gram)]) %disp(['Optimal revenue per gram is: ', num2str(Revenue_gram)])
disp(['Calculations were completed in ', num2str(toc), ' seconds.'])