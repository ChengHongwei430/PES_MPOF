classdef AGSK < ALGORITHM
    methods
        function main(Algorithm,Problem)

            %% Generate random population
            Population = Problem.Initialization();
            Pbest      = Population;
            [~,best]   = min(FitnessSingle(Pbest));
            Gbest      = Pbest(best);

            ConvDisp=0;
            Runs=1;
            val_2_reach=1.00e-8;
            problem_size=Problem.D;
            max_nfes=Problem.maxFE;

            val_2_reach = 10^(-8);%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            max_region = 100.0;
            min_region = -100.0;
            lu = [-100 * ones(1, problem_size); 100 * ones(1, problem_size)];
            analysis= zeros(10,6);
            KF_pool = [0.1 1.0 0.5 1.0];
            KR_pool = [0.2 0.1 0.9 0.9];
            optimumi= [100, 1100 ,700 ,1900 ,1700 ,1600 ,2100 ,2200 ,2400 ,2500];
            %optimum= optimumi(func);
            outcome = [];
            rand('state',sum(100*clock));

            bsf_error_val=[];

            %%  parameter settings for pop-size
            pop_size=Problem.N;
            max_pop_size = pop_size;
            min_pop_size = 12;

            %% Initialize the main population
            popold = Population.decs;
            pop = Population.decs; % the old population becomes the current population

            fitness = Population.objs;

            nfes = 0;
            bsf_fit_var = 1e+300;

            %%%%%%%%%%%%%%%%%%%%%%%% for out
            for i = 1 : pop_size
                nfes = nfes + 1;
                %%      if nfes > max_nfes; exit(1); end
                if nfes > max_nfes; break; end
                if fitness(i) < bsf_fit_var
                    bsf_fit_var = fitness(i);
                end
            end

            %%POSSIBLE VALUES FOR KNOWLEDGE RATE K%%%%
            K=[];
            KF=[];
            KR=[];
            Kind=rand(pop_size, 1);
            %%%%%%%%%%%%%%%%%%%K uniform rand (0,1) with prob 0.5 and unifrom integer [1,20] with prob 0.5
            K(Kind<0.5,:)= rand(sum(Kind<0.5), 1);
            K(Kind>=0.5,:)=ceil(20 * rand(sum(Kind>=0.5), 1));
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            g=0;
            %% main loop

            KW_ind=[];
            All_Imp=zeros(1,4);

            %% Optimization
            while Algorithm.NotTerminated(Population)


                if  (nfes < 0.1*max_nfes)% intial probability values
                    KW_ind=[0.85 0.05 0.05 0.05];
                    K_rand_ind=rand(pop_size, 1);
                    K_rand_ind(K_rand_ind>sum(KW_ind(1:3))&K_rand_ind<=sum(KW_ind(1:4)))=4;
                    K_rand_ind(K_rand_ind>sum(KW_ind(1:2))&K_rand_ind<=sum(KW_ind(1:3)))=3;
                    K_rand_ind(K_rand_ind>KW_ind(1)&K_rand_ind<=sum(KW_ind(1:2)))=2;
                    K_rand_ind(K_rand_ind>0&K_rand_ind<=KW_ind(1))=1;
                    KF=KF_pool(K_rand_ind)';
                    KR=KR_pool(K_rand_ind)';
                else %% updaing probability values
                    KW_ind=0.95*KW_ind+0.05*All_Imp;
                    KW_ind=KW_ind./sum(KW_ind);
                    K_rand_ind=rand(pop_size, 1);
                    K_rand_ind(K_rand_ind>sum(KW_ind(1:3))&K_rand_ind<=sum(KW_ind(1:4)))=4;
                    K_rand_ind(K_rand_ind>sum(KW_ind(1:2))&K_rand_ind<=sum(KW_ind(1:3)))=3;
                    K_rand_ind(K_rand_ind>KW_ind(1)&K_rand_ind<=sum(KW_ind(1:2)))=2;
                    K_rand_ind(K_rand_ind>0&K_rand_ind<=KW_ind(1))=1;
                    KF=KF_pool(K_rand_ind)';
                    KR=KR_pool(K_rand_ind)';

                end

                %%% Junior and Senior Gaining-Sharing phases %%%%%
                D_Gained_Shared_Junior=ceil((problem_size)*(1-nfes / max_nfes).^K);
                D_Gained_Shared_Senior=problem_size-D_Gained_Shared_Junior;
                pop = popold; % the old population becomes the current population

                [valBest, indBest] = sort(fitness, 'ascend');
                [Rg1, Rg2, Rg3] = Gained_Shared_Junior_R1R2R3(indBest);

                [R1, R2, R3] = Gained_Shared_Senior_R1R2R3(indBest);
                R01=1:pop_size;
                Gained_Shared_Junior=zeros(pop_size, problem_size);
                ind1=fitness(R01)>fitness(Rg3);

                if(sum(ind1)>0)
                    Gained_Shared_Junior (ind1,:)= pop(ind1,:) + KF(ind1, ones(1,problem_size)).* (pop(Rg1(ind1),:) - pop(Rg2(ind1),:)+pop(Rg3(ind1), :)-pop(ind1,:)) ;
                end
                ind1=~ind1;
                if(sum(ind1)>0)
                    Gained_Shared_Junior(ind1,:) = pop(ind1,:) + KF(ind1, ones(1,problem_size)) .* (pop(Rg1(ind1),:) - pop(Rg2(ind1),:)+pop(ind1,:)-pop(Rg3(ind1), :)) ;
                end
                R0=1:pop_size;
                Gained_Shared_Senior=zeros(pop_size, problem_size);
                ind=fitness(R0)>fitness(R2);
                if(sum(ind)>0)
                    Gained_Shared_Senior(ind,:) = pop(ind,:) + KF(ind, ones(1,problem_size)) .* (pop(R1(ind),:) - pop(ind,:) + pop(R2(ind),:) - pop(R3(ind), :)) ;
                end
                ind=~ind;
                if(sum(ind)>0)
                    Gained_Shared_Senior(ind,:) = pop(ind,:) + KF(ind, ones(1,problem_size)) .* (pop(R1(ind),:) - pop(R2(ind),:) + pop(ind,:) - pop(R3(ind), :)) ;
                end
                Gained_Shared_Junior = boundConstraint(Gained_Shared_Junior, pop, lu);
                Gained_Shared_Senior = boundConstraint(Gained_Shared_Senior, pop, lu);


                D_Gained_Shared_Junior_mask=rand(pop_size, problem_size)<=(D_Gained_Shared_Junior(:, ones(1, problem_size))./problem_size); % mask is used to indicate which elements of will be gained
                D_Gained_Shared_Senior_mask=~D_Gained_Shared_Junior_mask;

                D_Gained_Shared_Junior_rand_mask=rand(pop_size, problem_size)<=KR(:,ones(1, problem_size));
                D_Gained_Shared_Junior_mask=and(D_Gained_Shared_Junior_mask,D_Gained_Shared_Junior_rand_mask);

                D_Gained_Shared_Senior_rand_mask=rand(pop_size, problem_size)<=KR(:,ones(1, problem_size));
                D_Gained_Shared_Senior_mask=and(D_Gained_Shared_Senior_mask,D_Gained_Shared_Senior_rand_mask);
                ui=pop;

                ui(D_Gained_Shared_Junior_mask) = Gained_Shared_Junior(D_Gained_Shared_Junior_mask);
                ui(D_Gained_Shared_Senior_mask) = Gained_Shared_Senior(D_Gained_Shared_Senior_mask);

                children_fitness = SOLUTION(ui).objs;%%%最后用它产生最终子种群

                %%% Updating individuals %%%
                for i = 1 : pop_size
                    nfes = nfes + 1;
                    if nfes > max_nfes; break; end
                    if children_fitness(i) < bsf_fit_var
                        bsf_fit_var = children_fitness(i);
                        bsf_solution = ui(i, :);
                    end
                end

                %%%%  Calculate the improvemnt of each settings %%%
                dif = abs(fitness - children_fitness);
                %% I == 1: the parent is better; I == 2: the offspring is better
                Child_is_better_index = (fitness > children_fitness);
                dif_val = dif(Child_is_better_index == 1);
                All_Imp=zeros(1,4);% (1,4) delete for 4 cases
                for i=1:4
                    if(sum(and(Child_is_better_index,K_rand_ind==i))>0)
                        All_Imp(i)=sum(dif(and(Child_is_better_index,K_rand_ind==i)));
                    else
                        All_Imp(i)=0;
                    end
                end

                if(sum(All_Imp)~=0)
                    All_Imp=All_Imp./sum(All_Imp);
                    [temp_imp,Imp_Ind] = sort(All_Imp);
                    for imp_i=1:length(All_Imp)-1
                        All_Imp(Imp_Ind(imp_i))=max(All_Imp(Imp_Ind(imp_i)),0.05);
                    end
                    All_Imp(Imp_Ind(end))=1-sum(All_Imp(Imp_Ind(1:end-1)));
                else
                    Imp_Ind=1:length(All_Imp);
                    All_Imp(:)=1/length(All_Imp);
                end
                [fitness, Child_is_better_index] = min([fitness, children_fitness], [], 2);

                popold = pop;
                popold(Child_is_better_index == 2, :) = ui(Child_is_better_index == 2, :);

                %% for resizing the population size %%%%

                plan_pop_size = round((min_pop_size - max_pop_size)* ((nfes / max_nfes).^((1-nfes / max_nfes)))  + max_pop_size);

                if pop_size > plan_pop_size
                    reduction_ind_num = pop_size - plan_pop_size;
                    if pop_size - reduction_ind_num <  min_pop_size; reduction_ind_num = pop_size - min_pop_size;end

                    pop_size = pop_size - reduction_ind_num;
                    for r = 1 : reduction_ind_num
                        [valBest indBest] = sort(fitness, 'ascend');
                        worst_ind = indBest(end);
                        popold(worst_ind,:) = [];
                        pop(worst_ind,:) = [];
                        fitness(worst_ind,:) = [];
                        K(worst_ind,:)=[];
                    end

                end

                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                %bsf_error_val = bsf_fit_var - optimum;
                %bsf_error_val = bsf_fit_var;
%                 if bsf_error_val < val_2_reach
%                     bsf_error_val = 0;
%                     break;
%                 end

                Population=SOLUTION(popold);


            end
        end
    end
end