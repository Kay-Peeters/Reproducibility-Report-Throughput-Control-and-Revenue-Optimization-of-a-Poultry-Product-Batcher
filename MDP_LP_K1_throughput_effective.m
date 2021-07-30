%% Markov Decision Process formulation of the online 1-bounded bin-covering problem with rejection

clc
clear all
close all

%% Variables

Bmin = 35;
Bmax = 35;
B = Bmin:Bmax;      % batch size
q_vector = 0.8; %0:0.1:0.8;
r_B = 0.9;

DISTRIBUTION = 2;   % type of distribution used
ITEM_MIN = 7;       % minimum item weight
ITEM_MAX = 13;      % maximum item weight
MU = 10;             % mean item weight in discrete normal distribution
VARIANCE = (MU*0.15)^2;       % variance of item weight in discrete normal distribution
if DISTRIBUTION == 1
    M = ITEM_MIN;
    N = ITEM_MAX;
    p = [zeros(1,M-1) 1/(N-M+1) * ones(1,N-M+1)];
elseif DISTRIBUTION == 2
    p = normpdf([1:2*MU-1],MU,sqrt(VARIANCE));
    %p = p.*(p>1/1e6);
    p(1:floor(MU-1-4*sqrt(VARIANCE))) = 0;
    p(ceil(MU+1+4*sqrt(VARIANCE)):end) = 0;
    p = p./sum(p);
elseif DISTRIBUTION == 3
    p = [1/3 0 1/3 0 1/3];
end
W = find(p);        % item space
P_ = length(W);      % number of item sizes
A = 1:2;            % action space. a=1: reject. a=2: accept.
K = length(A);      % size of the action space
p_ = p(W);          % probability of items
w_max = W(end);

output = zeros(length(q_vector),6);
for q_ = 1:length(q_vector);
    q = q_vector(q_);

    G_vector = zeros(1,length(B));
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
        SS = length(S)*P_;

        % index numbers (multi -> single dimension array)
        index = zeros(length(S),length(W));   % index of states
        S_vector = zeros(SS,2);        % state space (v,w)
        s = 1;
        for v = 1:length(S)
            for w = 1:length(W)
                S_vector(s,:) = [S(v) W(w)];
                index(S(v)+1,W(w)) = s;
                s = s+1;
            end
        end

        P = zeros(SS*K,SS); % P(2*i-2+a,j) = the probability of moving from i to j when taking action a
        P2 = zeros(SS,SS,K); % P2(s1,s2,a) is the probability of moving from s1 to s2 if we take action a
        f = zeros(1,SS*K);   % objective function
        Aeq = zeros(SS+1,SS*K);
        beq = zeros(SS+1,1);
        for s = 1:SS
            v = S_vector(s,1);
            w = S_vector(s,2);

            % Accept
            v_ = v;
            v_ = v_ + w;
            r = v_ - B(b);  % giveaway
            if r >= 0
                f(2*s) = min([w,B(b)-v]);%r;
                v_ = 0;
            else
                f(2*s) = w;%0;
            end
            s_ = index(v_+1,W(1));
            P2(s,s_:s_+P_-1,2) = p_;

            % Reject
            v_ = v;
            f(2*s-1) = w*r_B;%0;
            s_ = index(v_+1,W(1));
            P2(s,s_:s_+P_-1,1) = p_;

        end
        f = -f;
        
        % first part of first equality constraint (rate of moving out of a state)
        ss = 0;
        for s = 1:SS
            for a = 1:K
                ss = ss + 1;
                Aeq(s,ss) = Aeq(s,ss) + 1;
            end
        end

        % second equality constraint (normalizing equation)
        Aeq(end,:) = 1;
        beq(end) = 1;

        % second part of first equality constraint (rate of moving into a state)
        for j = 1:SS
            for i = 1:SS
                for a = 1:K
                    Aeq(j,2*i-2+a) = Aeq(j,2*i-2+a) - P2(i,j,a);
                end
            end
        end

        % third equality constraint (throughput constraint)
        Aeq_ = zeros(1,SS*K);
        for i = 1:SS
            v = S_vector(i,1);
            w = S_vector(i,end);
            c = max(0,v+w-B(b));
            Aeq_(2*i) = c - w;
        end
        beq_ = -q*MU;

        Aeq = [Aeq; Aeq_];
        beq = [beq; beq_];
        
        % inequality constraint (nonnegativity of x)
        Ain = [-eye(SS*K)];
        bin = [zeros(SS*K,1)];
        lb = [];
        ub = [];
        x0 = [];
        options = optimoptions('linprog','Algorithm','Dual-Simplex','Maxiter',100000);
        options2 = optimoptions('linprog','Algorithm','Dual-Simplex','Maxiter',100000,'Display','Iter');
        [x,r_item] = linprog(f,Ain,bin,Aeq,beq,lb,ub,x0,options);

        disp('Revenue per gram:')
        disp(-r_item/MU)

        q_accept = sum(x(2:K:end));
        q_reject = 1 - q_accept;
        disp('Minimum fraction and accepted fraction:')
        disp([q q_accept])

        % Optimal actions
        A_star = zeros(SS,1);
        plotdata = cell(1,K);
        w_q = zeros(1,K);   % fraction of weight per gram accepted/rejected
        for s = 1:SS
            x_temp = (2*s-1):2*s;
            
            % accept
            if (x(x_temp(2)) > x(x_temp(1))) && (x(x_temp(1)) == 0)
                A_star(s) = 1;
                plotdata{2} = [plotdata{2}; S_vector(s,:)];
                w_q(1) = w_q(1) + x(x_temp(2))*S_vector(s,2);
                
            % reject
            elseif (x(x_temp(2)) < x(x_temp(1))) && (x(x_temp(2)) == 0)
                A_star(s) = 0;
                plotdata{1} = [plotdata{1}; S_vector(s,:)];
                w_q(2) = w_q(2) + x(x_temp(1))*S_vector(s,2);
            
            % either accept or reject
            elseif (x(x_temp(1)) > 0) && (x(x_temp(2)) > 0)
                A_star(s) = 10;
                plotdata{1} = [plotdata{1}; S_vector(s,:)];
                plotdata{2} = [plotdata{2}; S_vector(s,:)];
                w_q(1) = w_q(1) + x(x_temp(2))*S_vector(s,2);
                w_q(2) = w_q(2) + x(x_temp(1))*S_vector(s,2);
            end

            v = S_vector(s,1);
            w = S_vector(s,2);

        end

        g_temp = w_q(1) + sum(x(2:2:end).*f(2:2:end)');
        
        disp('Fraction of weight per gram accepted/rejected')
        disp(w_q/MU)

        output(q_,:) = [q q_accept g_temp/MU w_q/MU -r_item/MU];
        %output: [target weight% accept, item% accept, giveaway%, weight% accept, weight% reject, value/gram]
    end
end

disp('Output as: [q, item% accept, giveaway%, weight% accept, weight% reject, value/gram]:')
disp(output)
output2 = output(:,end);
verification = [output(:,4)-output(:,3) + output(:,5)*r_B, output(:,6)];

%% Plotting

% plotmarks = {'kx','ko'};
% figure
% hold on
% for k = 1:K
%     scatter(plotdata{k}(:,1),plotdata{k}(:,2),1500/length(S),plotmarks{k});
% end
% xlabel('v')
% ylabel('w')
% axis([0 S(end) W(1) W(end)])
% title(['(v,w)'])
% legend('Reject','Accept')
% 
% figure
% hold on
% plot(output(:,1),output(:,6));
% plot(output(:,1),output(:,4)-output(:,3));
% plot(output(:,1),output(:,4));
% xlabel('Target effective throughput per gram q')
% ylabel('Normalized value per gram')
% legend(['r_B = ' num2str(r_B)], 'Weight% Eff. Batched', 'Weight% Batched')






















