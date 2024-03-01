% The MATLAB source code of IDE-EDA is developed in MATLAB R2019b with 64 bits. 
% Please run this code on MATLAB2019b with 64 bits or later. 
% Please cite this article as: Yintong Li, Tong Han, Shangqin Tang,Changqiang Huang, Huan Zhou, Yuan Wang, 
% An Improved Differential Evolution by Hybridizing with Estimation-of-Distribution Algorithm, 
% Information Sciences, https://doi.org/10.1016/j.ins.2022.11.029.
% Corresponding author:Yintong Li. E-mail address: yintongli0007@163.com.

function [Global_score,Global_pos,Convergence_curve] = IDE_EDA_op(SearchAgents_no,FEsmax,lb,ub,dim)
lb=lb.*ones(dim,1);
ub=ub.*ones(dim,1);
k=3;%scaling factor responsible for the greediness of the rank selection
F = 0.3;
CR = 0.8;
H = 5;%historical memory size H.
NPinit = SearchAgents_no;%initial population size
NPmin = 4;%the population number at the end of the run
counteval = 0;
countiter = 1;
% Initialize archive
Asize =SearchAgents_no;
A = [];
fA=[];
nA = 0;
MF = F * ones(H, 1);%H :historical memory size H.
MCR = CR * ones(H, 1);
iM = 1;
% Initialize population and Evaluation
X = lb + (ub - lb) .* rand(dim,SearchAgents_no);



fitness = [SOLUTION(X').objs]';



counteval = counteval + SearchAgents_no;
[fitness, fidx] = sort(fitness);
X = X(:, fidx);
V = X;
U = X;
S_CR = zeros(1, SearchAgents_no);
S_F = zeros(1, SearchAgents_no);
S_df = zeros(1, SearchAgents_no);
Chy = cauchyrnd(0, 0.1, SearchAgents_no + 10);
iChy = 1;
rcount=0;
while counteval < FEsmax
    %RSP
    Ri=1:SearchAgents_no;
    Rank=k*(SearchAgents_no-Ri)+1;
    Pr=Rank./sum(Rank);
    p=0.085+0.085*counteval/FEsmax;
    % Memory Indices
    MCR(H)=0.9;
    MF(H)=0.9;
    r = floor(1 + H * rand(1, SearchAgents_no));
    % Crossover rates
    CR = MCR(r)' + 0.1 * randn(1, SearchAgents_no);
    CR((CR < 0) | (MCR(r)' == -1)) = 0;%terminal value=-1
    CR(CR > 1) = 1;
    % Scaling factors
    F = zeros(1, SearchAgents_no);
    for i = 1 : SearchAgents_no
        while F(i) <= 0
            F(i) = MF(r(i)) + Chy(iChy);
            iChy = mod(iChy, numel(Chy)) + 1;
        end
    end
    F(F > 1) = 1;
    %_________________________iSO__________________
    if counteval<0.6*FEsmax
        F(F>0.7) = 0.7;
    end
    if counteval<0.2*FEsmax
        FW = 0.7*F;
    elseif counteval<0.4*FEsmax
        FW = 0.8*F;
    else
        FW = 1.2*F;
    end
    if counteval<0.25*FEsmax
        CR(CR<0.7) = 0.7;
    elseif counteval<0.5*FEsmax
        CR(CR<0.6) = 0.6;
    end
    % pbest index
    pnum=max(2, round(p * SearchAgents_no));
    pbest = randi(pnum,1, SearchAgents_no);
    % Mutation
    r1 =randsample(SearchAgents_no,SearchAgents_no,true,Pr);
    r2 =randsample(SearchAgents_no,SearchAgents_no,true,Pr);
    for i = 1 : SearchAgents_no
        % Generate r1 in RSP
        while i == r1(i)
            if rcount<10
                r1(i) =randsample(SearchAgents_no,1,true,Pr);
                rcount=rcount+1;
            else
                break;
            end
        end
        rcount=0;
        % Generate r2 in RSP
        while i == r2(i) || r1(i) == r2(i)
            if rcount<10
                r2(i) =randsample(SearchAgents_no,1,true,Pr);
                rcount=rcount+1;
            else
                break;
            end
        end
        rcount=0;
        if rand<(nA/(nA+SearchAgents_no))
            r2(i) = randi(nA);
            V(:, i) = X(:, i) + FW(i) .* (X(:, pbest(i)) - X(:, i)) + F(i) .* (X(:, r1(i)) - A(:, r2(i)));
        else
            V(:, i) = X(:, i) + FW(i) .* (X(:, pbest(i)) - X(:, i)) + F(i) .* (X(:, r1(i)) - X(:, r2(i)));
        end
        for j = 1 : dim
            if V(j, i) < lb(j)
                V(j, i) = 0.5 * (lb(j) + X(j, i));
            end
            if V(j, i) > ub(j)
                V(j, i) = 0.5 * (ub(j) + X(j, i));
            end
        end
        jrand = randi(dim);
        for j = 1 : dim
            if rand <= CR(i) || j == jrand
                U(j, i) = V(j, i);
            else
                U(j, i) = X(j, i);
            end
        end
    end


    fu=[SOLUTION(U').objs]';

    % Selection
    nS = 0;
    for i = 1 : SearchAgents_no
        if fu(i) < fitness(i)
            if nA < Asize
                A(:, nA + 1)= X(:, i);
                fA(nA + 1)=fitness(i);
                nA= nA + 1;
            else
                ri= floor(1 + Asize * rand);
                A(:, ri)= X(:, i);
                fA(ri)=fitness(i);
            end
            nS			= nS + 1;
            S_CR(nS)	= CR(i);
            S_F(nS)		= F(i);
            S_df(nS)	= abs(fu(i) - fitness(i)); 
            X(:, i)		= U(:, i);
            fitness(i)	= fu(i);
        elseif fu(i) == fitness(i)
            X(:, i)		= U(:, i);
        end
    end
    % Update MCR and MF
    if nS > 0 
        w = S_df(1 : nS) ./ sum(S_df(1 : nS));
        if all(S_CR(1 : nS) == 0)
            MCR(iM) = -1;
        elseif MCR(iM) ~= -1
            MCR(iM) = (sum(w .* S_CR(1 : nS) .* S_CR(1 : nS)) / sum(w .* S_CR(1 : nS))+MCR(iM))/2;  
        end
        MF(iM) = (sum(w .* S_F(1 : nS) .* S_F(1 : nS)) / sum(w .* S_F(1 : nS))+MF(iM))/2;
        iM = mod(iM, H) + 1;
    end
        % Sort
    [fitness, fidx] = sort(fitness);
        % Update NP and population
    SearchAgents_no = round(NPinit - (NPinit - NPmin) * counteval / FEsmax);
    fitness = fitness(1 : SearchAgents_no);
    X = X(:, fidx(1 : SearchAgents_no));
    U = X;
    [fA, fidxA] = sort(fA);
    A = A(:, fidxA);
    Asize = SearchAgents_no;
    if nA > Asize
        nA = Asize;
        A = A(:,1 : Asize);
        fA=fA(1 : Asize);
    end
    %____________EDA
    PApnum=max(2, ceil(0.5 * SearchAgents_no));
    if PApnum<2*dim
        PApnum=SearchAgents_no;
    end
    newIndividual_num = max(1, ceil(0.9 * pnum)); 
    selection_pos = X(:,1:PApnum)';
    mu = mean(selection_pos, 1);
    SEL=PApnum;
    mu_matrix=repmat(mu,SEL,1);
    temp_C=selection_pos-mu_matrix;
    C_matrix =(temp_C'*temp_C)/(SEL-1);
    C_matrix = triu(C_matrix) + transpose(triu(C_matrix,1));
 
    temp_pos = mvnrnd(mu,C_matrix,newIndividual_num)';
   
    UB=repmat(ub,1,newIndividual_num);LB=repmat(lb,1,newIndividual_num);Ran=rand(dim,newIndividual_num);
    temp_index1 = temp_pos > UB;
    temp_index2 = temp_pos < LB;
    temp_index = temp_index1 | temp_index2;
    temp_pos(temp_index) = Ran(temp_index).*(UB(temp_index)-LB(temp_index))+LB(temp_index);




    fuE=[SOLUTION(temp_pos').objs]';

    

    counteval = counteval + SearchAgents_no+newIndividual_num;
    fuEU=[fuE fitness];
    XU=[temp_pos X];
    [fitnessU, fidxU] = sort(fuEU);
    fitness(1:SearchAgents_no) = fitnessU(1 : SearchAgents_no);
    X(:,1:SearchAgents_no) = XU(:, fidxU(1 : SearchAgents_no));
    Convergence_curve(1,countiter)=fitness(1);
    countiter = countiter + 1;
end
Global_score = fitness(1);
Global_pos = X(:, 1);
end

function r= cauchyrnd(varargin)
% % % % %Generate random numbers from the Cauchy distribution, r= a + b*tan(pi*(rand(n)-0.5)).
% % % % Chy = cauchyrnd(0, 0.1, SearchAgents_no + 10);%(SearchAgents_no + 10)*(SearchAgents_no + 10) rand numbers
% USAGE:       r= cauchyrnd(a, b, n, ...)
% Generate random numbers from the Cauchy distribution, r= a + b*tan(pi*(rand(n)-0.5)).
% ARGUMENTS:
% a (default value: 0.0) must be scalars or size(x).
% b (b>0, default value: 1.0) must be scalars or size(x).
% n and onwards (default value: 1) specifies the dimension of the output.
% EXAMPLE:
% r= cauchyrnd(0, 1, 10); % A 10 by 10 array of random values, Cauchy distributed.
% SEE ALSO:    cauchycdf, cauchyfit, cauchyinv, cauchypdf.
% Copyright (C) Peder Axensten <peder at axensten dot se>
% HISTORY:
% Version 1.0, 2006-07-10.
% Version 1.1, 2006-07-26.
% - Added cauchyfit to the cauchy package.
% Version 1.2, 2006-07-31:
% - cauchyinv(0, ...) returned a large negative number but should be -Inf.
% - Size comparison in argument check didn't work.
% - Various other improvements to check list.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Default values
a=	0.0;
b=	1.0;
n=	1;
% Check the arguments
if(nargin >= 1)
    a=	varargin{1};
    if(nargin >= 2)
        b=			varargin{2};
        b(b <= 0)=	NaN;	% Make NaN of out of range values.
        if(nargin >= 3)
            n=	[varargin{3:end}];		
        end
    end
end
% Generate
r=	cauchyinv(rand(n), a, b);
end

function x= cauchyinv(p, varargin)
% USAGE:       x= cauchyinv(p, a, b)
% Inverse of the Cauchy cumulative distribution function (cdf), x= a + b*tan(pi*(p-0.5)).
% ARGUMENTS:
% p (0<=p<=1) might be of any dimension.
% a (default value: 0.0) must be scalars or size(p).
% b (b>0, default value: 1.0) must be scalars or size(p).
% EXAMPLE:
% p= 0:0.01:1;
% plot(cauchyinv(p), p);
% SEE ALSO:    cauchycdf, cauchyfit, cauchypdf, cauchyrnd.
% Copyright (C) Peder Axensten <peder at axensten dot se>
% HISTORY:
% Version 1.0, 2006-07-10.
% Version 1.1, 2006-07-26.
% - Added cauchyfit to the cauchy package. 
% Version 1.2, 2006-07-31:
% - cauchyinv(0, ...) returned a large negative number but should be -Inf. 
% - Size comparison in argument check didn't work. 
% - Various other improvements to check list. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Default values
	a=	0.0;
	b=	1.0;
	% Check the arguments
	if(nargin >= 2)
		a=	varargin{1};
		if(nargin == 3)
			b=			varargin{2};
			b(b <= 0)=	NaN;	% Make NaN of out of range values.
		end
	end
	if((nargin < 1) || (nargin > 3))
		error('At least one argument, at most three!');
    end
	p(p < 0 | 1 < p)=	NaN;
	% Calculate
	x=			a + b.*tan(pi*(p-0.5));
	% Extreme values. 
	if(numel(p) == 1), 	p= repmat(p, size(x));		end
	x(p == 0)=	-Inf;
	x(p == 1)=	Inf;
end