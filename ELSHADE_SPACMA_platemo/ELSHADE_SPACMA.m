classdef ELSHADE_SPACMA < ALGORITHM

    methods
        function main(Algorithm,Problem)
            %% Generate random population
            Population = Problem.Initialization(18 * Problem.D);
            problem_size=Problem.D;
            Archive    = [];
            MCR = zeros(Problem.N,1) + 0.5;
            MF  = zeros(Problem.N,1) + 0.5;
            k   = 1;
            L_Rate= 0.80;
            val_2_reach = 10^(-8);
            RecordFEsFactor = ...
                [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, ...
                0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
            progress = numel(RecordFEsFactor);
            max_nfes = Problem.maxFE;
            rand('seed', sum(100 * clock));
            lu = [-100 * ones(1, Problem.D); 100 * ones(1, Problem.D)];

            %%%%%%%%%%%%%%optimum = Problem. * 100.0;
            outcome = [];

            runs=1;
            allerrorvals = zeros(progress, runs);%不需要存数据
            nfes = 0;
            run_funcvals = [];

            %%   L-SHADE的参数%%%%%%%%%%%%%%%%%%%%%%
            p_best_rate = 0.11;    
            arc_rate = 1.4; 
            memory_size = 5; 
            pop_size = 18 * Problem.D;
            max_pop_size = pop_size;
            min_pop_size = 4.0;

            %%  混合的参数%%%%%%%%%%%%%%%%
            First_calss_percentage=0.5;

            %% 初始化主种群
            popold = Population.decs;
            pop = Population.decs; %初始种群          
            fitness=Population.objs;%适应度

            bsf_fit_var = 1e+30;%%%%%%%%%%%%%%
            bsf_index = 0;
            bsf_solution = zeros(1, problem_size);

            for i = 1 : pop_size
                nfes = nfes + 1;
                if (fitness(i) < bsf_fit_var && isreal(pop(i, :)) && sum(isnan(pop(i, :)))==0 && min(pop(i, :))>=-100 && max(pop(i, :))<=100)
                    bsf_fit_var = fitness(i);
                    bsf_solution = pop(i, :);
                    bsf_index = i;%%初始最优索引
                end
                
                if nfes > max_nfes
                    break; 
                end
            end

            run_funcvals = [run_funcvals;ones(pop_size,1)*bsf_fit_var];
            
            memory_sf = 0.5 .* ones(memory_size, 1);
            memory_cr = 0.5 .* ones(memory_size, 1);
            memory_pos = 1;
            
            archive.NP = arc_rate * pop_size; % 档案尺寸
            archive.pop = zeros(0, problem_size); % 档案
            archive.funvalues = zeros(0, 1); % 档案适应度
            
            memory_1st_class_percentage = First_calss_percentage.* ones(memory_size, 1); % 混合概率 

            %% CMAES 的参数%%%%%%%%%%%%
            sigma = 0.5;          % coordinate wise standard deviation (step size)
            xmean = rand(problem_size,1);    % objective variables initial point
            mu = pop_size/2;               % number of parents/points for recombination
            weights = log(mu+1/2)-log(1:mu)'; % muXone array for weighted recombination
            mu = floor(mu);
            weights = weights/sum(weights);     % normalize recombination weights array
            mueff=sum(weights)^2/sum(weights.^2); % variance-effectiveness of sum w_i x_i
            
            % Strategy parameter setting: Adaptation
            cc = (4 + mueff/problem_size) / (problem_size+4 + 2*mueff/problem_size); % time constant for cumulation for C
            cs = (mueff+2) / (problem_size+mueff+5);  % t-const for cumulation for sigma control
            c1 = 2 / ((problem_size+1.3)^2+mueff);    % learning rate for rank-one update of C
            cmu = min(1-c1, 2 * (mueff-2+1/mueff) / ((problem_size+2)^2+mueff));  % and for rank-mu update
            damps = 1 + 2*max(0, sqrt((mueff-1)/(problem_size+1))-1) + cs; % damping for sigma usually close to 1

            % Initialize dynamic (internal) strategy parameters and constants
            pc = zeros(problem_size,1);
            ps = zeros(problem_size,1);   % evolution paths for C and sigma
            B = eye(problem_size,problem_size);                       % B defines the coordinate system
            D = ones(problem_size,1);                      % diagonal D defines the scaling
            C = B * diag(D.^2) * B';            % covariance matrix C
            invsqrtC = B * diag(D.^-1) * B';    % C^-1/2
            eigeneval = 0;                      % track update of B and D
            chiN=problem_size^0.5*(1-1/(4*problem_size)+1/(21*problem_size^2));  % expectation of
            
            
            %% main loop
            Hybridization_flag=1; % Indicator flag if we need to Activate CMAES Hybridization
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
            %while (nfes < max_nfes)

                pop = popold; %旧种群变成当前种群
                [temp_fit, sorted_index] = sort(fitness, 'ascend');
                
                mem_rand_index = ceil(memory_size * rand(pop_size, 1));
                mu_sf = memory_sf(mem_rand_index);
                mu_cr = memory_cr(mem_rand_index);
                mem_rand_ratio = rand(pop_size, 1);

                 %% 产生交叉率
                cr = normrnd(mu_cr, 0.1);
                term_pos = find(mu_cr == -1);
                cr(term_pos) = 0;
                cr = min(cr, 1);
                cr = max(cr, 0);

                %% 产生比例因子
                if(nfes <= max_nfes/2)
                    sf=0.45+.1*rand(pop_size, 1);
                    pos = find(sf <= 0);
                    
                    while ~ isempty(pos)
                        sf(pos)=0.45+0.1*rand(length(pos), 1);
                        pos = find(sf <= 0);
                    end
                else
                    sf = mu_sf + 0.1 * tan(pi * (rand(pop_size, 1) - 0.5));
                    
                    pos = find(sf <= 0);
                    
                    while ~ isempty(pos)
                        sf(pos) = mu_sf(pos) + 0.1 * tan(pi * (rand(length(pos), 1) - 0.5));
                        pos = find(sf <= 0);
                    end
                end
                sf = min(sf, 1);

                %% 产生杂交概率
                Class_Select_Index=(memory_1st_class_percentage(mem_rand_index)>=mem_rand_ratio);
                if(Hybridization_flag==0)
                    Class_Select_Index=or(Class_Select_Index,~Class_Select_Index);%All will be in class#1
                end

                %%
                r0 = [1 : pop_size];
                popAll = [pop; archive.pop];
                [r1, r2] = gnR1R2(pop_size, size(popAll, 1), r0);
                
                pNP = max(round(p_best_rate * pop_size), 2); %% 选择两个最优方案
                randindex = ceil(rand(1, pop_size) .* pNP); %% select from [1, 2, 3, ..., pNP]
                randindex = max(1, randindex); %% to avoid the problem that rand = 0 and thus ceil(rand) = 0
                pbest = pop(sorted_index(randindex), :); %% 随机方案

                vi=[];
                temp=[];
                if(sum(Class_Select_Index)~=0)
                    vi(Class_Select_Index,:) = pop(Class_Select_Index,:) + sf(Class_Select_Index, ones(1, problem_size)) .* (pbest(Class_Select_Index,:) - pop(Class_Select_Index,:) + pop(r1(Class_Select_Index), :) - popAll(r2(Class_Select_Index), :));
                end
                
                if(sum(~Class_Select_Index)~=0)
                    for k=1:sum(~Class_Select_Index)
                        temp(:,k) = xmean + sigma * B * (D .* randn(problem_size,1)); % m + sig * Normal(0,C)
                    end
                    vi(~Class_Select_Index,:) = temp';
                end
                
                if(~isreal(vi))
                    Hybridization_flag=0;
                    continue;
                end

                
                vi = boundConstraint(vi, pop, lu);

                
                mask = rand(pop_size, problem_size) > cr(:, ones(1, problem_size)); % mask is used to indicate which elements of ui comes from the parent
                rows = (1 : pop_size)'; cols = floor(rand(pop_size, 1) * problem_size)+1; % choose one position where the element of ui doesn't come from the parent
                jrand = sub2ind([pop_size problem_size], rows, cols); mask(jrand) = false;
                ui = vi; ui(mask) = pop(mask);

                children_fitness = SOLUTION(ui).objs;%%%最后用它产生最终子种群

                %%%%%%%%%%%%%%%%%%%%%%%% for out
                for i = 1 : pop_size
                    nfes = nfes + 1;
                    if (children_fitness(i) < bsf_fit_var && isreal(ui(i, :)) && sum(isnan(ui(i, :)))==0 && min(ui(i, :))>=-100 && max(ui(i, :))<=100)
                        bsf_fit_var = children_fitness(i);
                        bsf_solution = ui(i, :);
                        bsf_index = i;
                    end
                    
                    if nfes > max_nfes
                        break;
                    end
                end
                %%%%%%%%%%%%%%%%%%%%%%%% for out

                run_funcvals = [run_funcvals;ones(pop_size,1)*bsf_fit_var];                
                dif = abs(fitness - children_fitness);

                %% I == 1: the parent is better; I == 2: the offspring is better
                Child_is_better_index = (fitness > children_fitness);
                goodCR = cr(Child_is_better_index == 1);
                goodF = sf(Child_is_better_index == 1);
                dif_val = dif(Child_is_better_index == 1);
                dif_val_Class_1 = dif(and(Child_is_better_index,Class_Select_Index) == 1);
                dif_val_Class_2 = dif(and(Child_is_better_index,~Class_Select_Index) == 1);
                
                archive = updateArchive(archive, popold(Child_is_better_index == 1, :), fitness(Child_is_better_index == 1));
                
                [fitness, Child_is_better_index] = min([fitness, children_fitness], [], 2);

                popold = pop;
                popold(Child_is_better_index == 2, :) = ui(Child_is_better_index == 2, :);
                
                num_success_params = numel(goodCR);

                if num_success_params > 0
                    sum_dif = sum(dif_val);
                    dif_val = dif_val / sum_dif;
                    
                    %% 更新缩放因子的内存
                    memory_sf(memory_pos) = (dif_val' * (goodF .^ 2)) / (dif_val' * goodF);
                    
                    %% 更新交叉率的内存
                    if max(goodCR) == 0 || memory_cr(memory_pos)  == -1
                        memory_cr(memory_pos)  = -1;
                    else
                        memory_cr(memory_pos) = (dif_val' * (goodCR .^ 2)) / (dif_val' * goodCR);
                    end
                    
                    if (Hybridization_flag==1)%如果使用了混合
                        memory_1st_class_percentage(memory_pos) = memory_1st_class_percentage(memory_pos)*L_Rate+ (1-L_Rate)*(sum(dif_val_Class_1) / (sum(dif_val_Class_1) + sum(dif_val_Class_2)));
                        memory_1st_class_percentage(memory_pos)=min(memory_1st_class_percentage(memory_pos),0.8);
                        memory_1st_class_percentage(memory_pos)=max(memory_1st_class_percentage(memory_pos),0.2);
                    end
                    
                    memory_pos = memory_pos + 1;
                    if memory_pos > memory_size;  memory_pos = 1; end
                end

                %% for resizing the population size
                plan_pop_size = round((((min_pop_size - max_pop_size) / max_nfes) * nfes) + max_pop_size);
                
                if pop_size > plan_pop_size
                    reduction_ind_num = pop_size - plan_pop_size;
                    if pop_size - reduction_ind_num <  min_pop_size; reduction_ind_num = pop_size - min_pop_size;end
                    
                    pop_size = pop_size - reduction_ind_num;
                    for r = 1 : reduction_ind_num
                        [valBest, indBest] = sort(fitness, 'ascend');
                        worst_ind = indBest(end);
                        popold(worst_ind,:) = [];
                        pop(worst_ind,:) = [];
                        fitness(worst_ind,:) = [];
                        Child_is_better_index(worst_ind,:) = [];
                    end
                    
                    archive.NP = round(arc_rate * pop_size);
                    
                    if size(archive.pop, 1) > archive.NP
                        rndpos = randperm(size(archive.pop, 1));
                        rndpos = rndpos(1 : archive.NP);
                        archive.pop = archive.pop(rndpos, :);
                    end
                    
                    %% update CMA parameters
                    mu = pop_size/2;               % number of parents/points for recombination
                    weights = log(mu+1/2)-log(1:mu)'; % muXone array for weighted recombination
                    mu = floor(mu);
                    weights = weights/sum(weights);     % normalize recombination weights array
                    mueff=sum(weights)^2/sum(weights.^2); % variance-effectiveness of sum w_i x_i
                end

                %% CMAES Adaptation
                if(Hybridization_flag==1)
                    % Sort by fitness and compute weighted mean into xmean
                    [~, popindex] = sort(fitness);  % minimization
                    xold = xmean;
                    xmean = popold(popindex(1:mu),:)' * weights;  % recombination, new mean value
                    
                    % Cumulation: Update evolution paths
                    ps = (1-cs) * ps ...
                        + sqrt(cs*(2-cs)*mueff) * invsqrtC * (xmean-xold) / sigma;
                    hsig = sum(ps.^2)/(1-(1-cs)^(2*nfes/pop_size))/problem_size < 2 + 4/(problem_size+1);
                    pc = (1-cc) * pc ...
                        + hsig * sqrt(cc*(2-cc)*mueff) * (xmean-xold) / sigma;
                    
                    % Adapt covariance matrix C
                    artmp = (1/sigma) * (popold(popindex(1:mu),:)' - repmat(xold,1,mu));  % mu difference vectors
                    C = (1-c1-cmu) * C ...                   % regard old matrix
                        + c1 * (pc * pc' ...                % plus rank one update
                        + (1-hsig) * cc*(2-cc) * C) ... % minor correction if hsig==0
                        + cmu * artmp * diag(weights) * artmp'; % plus rank mu update
                    
                    % Adapt step size sigma
                    sigma = sigma * exp((cs/damps)*(norm(ps)/chiN - 1));
                    
                    % Update B and D from C
                    if nfes - eigeneval > pop_size/(c1+cmu)/problem_size/10  % to achieve O(problem_size^2)
                        eigeneval = nfes;
                        C = triu(C) + triu(C,1)'; % enforce symmetry
                        if(sum(sum(isnan(C)))>0 || sum(sum(~isfinite(C)))>0 || ~isreal(C))
                            Hybridization_flag=0;
                            continue;
                        end
                        [B,D] = eig(C);           % eigen decomposition, B==normalized eigenvectors
                        D = sqrt(diag(D));        % D contains standard deviations now
                        invsqrtC = B * diag(D.^-1) * B';
                    end                   
                end                
                Population=SOLUTION(popold);
            end
        end
    end
end