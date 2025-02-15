classdef RUNnn < ALGORITHM

    methods
        function main(Algorithm,Problem)
                        
            %% Generate random population
            Population = Problem.Initialization();

            %% Parameter setting
            nP = Problem.N;
            dim = Problem.D;
            ub = Problem.upper(1);
            lb = Problem.lower(1);
            Xnew2 = zeros(1,dim);
            Cost = zeros(nP,1);
            X = Population.decs;

            % 计算初始种群的适应度
            for i=1:nP
                Cost(i) = Population(i).obj;      % 计算目标函数值
            end

            % 找到初始种群中的最优解
            [Best_Cost,ind] = min(Cost);     % 找到最小适应度值和对应的解
            Best_X = X(ind,:);
            
            %% Optimization
            while Algorithm.NotTerminated(Population)

                f=20.*exp(-(12.*(Problem.FE/Problem.maxFE))); % 计算适应度的适应因子f
                Xavg = mean(X);               % 计算当前种群的平均位置
                SF=2.*(0.5-rand(1,nP)).*f;    % 计算自适应因子

                for i=1:nP
                    [~,ind_l] = min(Cost); % 找到当前种群中最优解的索引
                    lBest = X(ind_l,:);   % lBest表示最优解

                    [A,B,C]=RndX(nP,i);   % 获取三个随机解的索引
                    [~,ind1] = min(Cost([A B C])); % 找到三个随机解中的最优解

                end

                % 计算ΔX
                gama = rand.*(X(i,:)-rand(1,dim).*(ub-lb)).*exp(-4*Problem.FE/Problem.maxFE);
                Stp=rand(1,dim).*((Best_X-rand.*Xavg)+gama);
                DelX = 2*rand(1,dim).*(abs(Stp)); % 更新步长

                % 使用Runge Kutta方法计算搜索机制
                if Cost(i)<Cost(ind1)
                    Xb = X(i,:); % 当前解
                    Xw = X(ind1,:); % 最优解
                else
                    Xb = X(ind1,:); % 最优解
                    Xw = X(i,:); % 当前解
                end

                SM = RungeKutta(Xb,Xw,DelX);   % Runge Kutta方法进行搜索优化

                L=rand(1,dim)<0.5;  % 随机生成一个0-1数组
                Xc = L.*X(i,:)+(1-L).*X(A,:);  % 计算交叉解
                Xm = L.*Best_X+(1-L).*lBest;   % 计算混合解

                vec=[1,-1];
                flag = floor(2*rand(1,dim)+1);
                r=vec(flag);  % 随机选择1或-1
                g = 2*rand;
                mu = 0.5+.1*randn(1,dim);  % 添加随机噪声

                % 基于Runge Kutta方法计算新的解
                if rand<0.5
                    Xnew = (Xc+r.*SF(i).*g.*Xc) + SF(i).*(SM) + mu.*(Xm-Xc);
                else
                    Xnew = (Xm+r.*SF(i).*g.*Xm) + SF(i).*(SM)+ mu.*(X(A,:)-X(B,:));
                end

                % 检查新的解是否越界，如果越界则调整
                FU=Xnew>ub;FL=Xnew<lb;Xnew=(Xnew.*(~(FU+FL)))+ub.*FU+lb.*FL; 

                Offspring = SOLUTION(Xnew); % 计算新的后代适应度值

                % 如果新的适应度值更好，则更新解
                if FitnessSingle(Population(i)) > FitnessSingle(Offspring)
                    Population(i) = Offspring;
                end

                %% 增强解质量（ESQ）
                if rand<0.5

                    EXP=exp(-5*rand*Problem.FE/Problem.maxFE);
                    r = floor(Unifrnd(-1,2,1,1));

                    u=2*rand(1,dim);
                    w=Unifrnd(0,2,1,dim).*EXP;

                    [A,B,C]=RndX(nP,i);
                    Xavg=(X(A,:)+X(B,:)+X(C,:))/3;

                    beta=rand(1,dim);
                    Xnew1 = beta.*(Best_X)+(1-beta).*(Xavg);

                    % 计算增强后的解
                    for j=1:dim
                        if w(j)<1
                            Xnew2(j) = Xnew1(j)+r*w(j)*abs((Xnew1(j)-Xavg(j))+randn);
                        else
                            Xnew2(j) = (Xnew1(j)-Xavg(j))+r*w(j)*abs((u(j).*Xnew1(j)-Xavg(j))+randn);
                        end
                    end

                    FU=Xnew2>ub;FL=Xnew2<lb;Xnew2=(Xnew2.*(~(FU+FL)))+ub.*FU+lb.*FL;  % 检查越界

                    Offspring = SOLUTION(Xnew); % 计算新的后代适应度值

                    % 如果新的适应度值更好，则更新解
                    if FitnessSingle(Population(i)) > FitnessSingle(Offspring)
                        Population(i) = Offspring;
                    else
                        if rand<w(randi(dim))
                            SM = RungeKutta(X(i,:),Xnew2,DelX);
                            Xnew = (Xnew2-rand.*Xnew2)+ SF(i)*(SM+(2*rand(1,dim).*Best_X-Xnew2));  % (公式20)

                            FU=Xnew>ub;FL=Xnew<lb;Xnew=(Xnew.*(~(FU+FL)))+ub.*FU+lb.*FL;  % 检查越界
                            Offspring = SOLUTION(Xnew); % 计算新的后代适应度值

                            if FitnessSingle(Population(i)) > FitnessSingle(Offspring)
                                Population(i) = Offspring;
                            end
                        end
                    end
                end

                %% 更新最优解
                X = Population.decs;
                % 计算初始种群的适应度
                for i=1:nP 
                    Cost(i) = Population(i).obj;   
                end
                if Cost(i)<Best_Cost
                    Best_X=X(i,:);  % 更新最优解
                    Best_Cost=Cost(i);  % 更新最优适应度值
                end

            end

           
        end
    end
end




% 获取三个随机解的索引
function [A,B,C]=RndX(nP,i)
Qi=randperm(nP);Qi(Qi==i)=[];  % 随机生成索引，并确保不等于当前解的索引
A=Qi(1);B=Qi(2);C=Qi(3);
end




% Runge Kutta方法实现搜索机制
function SM=RungeKutta(XB,XW,DelX)

dim=size(XB,2);
C=randi([1 2])*(1-rand);  % 随机选择一个常数
r1=rand(1,dim);
r2=rand(1,dim);

K1 = 0.5*(rand*XW-C.*XB);  % 计算K1
K2 = 0.5*(rand*(XW+r2.*K1.*DelX/2)-(C*XB+r1.*K1.*DelX/2));  % 计算K2
K3 = 0.5*(rand*(XW+r2.*K2.*DelX/2)-(C*XB+r1.*K2.*DelX/2));  % 计算K3
K4 = 0.5*(rand*(XW+r2.*K3.*DelX)-(C*XB+r1.*K3.*DelX));  % 计算K4

XRK = (K1+2.*K2+2.*K3+K4);  % 计算Runge Kutta的结果
SM=1/6*XRK;
end



% 生成均匀分布的随机数
function z=Unifrnd(a,b,c,dim)
a2 = a/2;
b2 = b/2;
mu = a2+b2;
sig = b2-a2;
z = mu + sig .* (2*rand(c,dim)-1);
end
