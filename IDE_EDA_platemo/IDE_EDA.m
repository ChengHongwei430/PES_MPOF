classdef IDE_EDA < ALGORITHM


    methods
        function main(Algorithm,Problem)
           
            dim=Problem.D;
            FEsmax=Problem.maxFE;
            lb=Problem.lower(1);
            ub=Problem.upper(1);
            SearchAgents_no=ceil(75*dim^(2/3));
            
            %% Generate random population
            Population = Problem.Initialization(SearchAgents_no);
            Pbest      = Population;
            [~,best]   = min(FitnessSingle(Pbest));
            Gbest      = Pbest(best);
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
                IDE_EDA_op(SearchAgents_no,FEsmax,lb,ub,dim);
            end
        end
    end
end