classdef PES_MPOF < ALGORITHM

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [minN,aRate] = Algorithm.ParameterSet(5,1.5);
            
            %% Generate random population
            Population = Problem.Initialization();
            Archive    = [];
            MCR = zeros(20*Problem.D,1) + 0.2;
            MF  = zeros(20*Problem.D,1) + 0.2;
            k   = 1;
            MOP = ones(1,3)/3;
            sumrep=inf;
            
            %% Optimization
            while Algorithm.NotTerminated(Population)
               

            end
        end
    end
end