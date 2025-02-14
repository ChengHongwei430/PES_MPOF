x = [0.2056688,3.256566,9.036931,0.2057393];
P = 6000;
L = 14;
TAUMAX = 13600;
TAUP=P/(sqrt(2)*x(1)*x(2));
M =  P*(L+x(2)/2); 
R = sqrt((x(2)^2)/4+((x(1)+x(3))/2)^2);
J1 = 2*sqrt(2)*x(1)*x(2); 
J2 = (x(2)^2)/4+((x(1)+x(3))/2)^2;
J = J1*J2;
TAUPP = (M*R)/J;
TAU = sqrt(TAUP^2+2*TAUP*TAUPP*x(2)/(2*R)+TAUPP^2);
g1 = TAU-TAUMAX;

SIGMA = (6*P*L)/(x(4)*x(3)^2); 
SIGMAX = 30000;
g2 = SIGMA-SIGMAX;

g3 = x(1)-x(4);

g4 = 0.10471*x(1)^2+0.04811*x(3)*x(4)*(14+x(2))-5;

g5 = 0.125-x(1);

E = 30e6;
DELTMAX = 0.25; 
DELTA = (4*P*L^3)/(E*x(3)^3*x(4));
g6 = DELTA-DELTMAX;

G = 12e6;
tem = 14;
PC = 4.013*E*sqrt((x(3)^2*x(4)^6)/36)*(1-x(3)*sqrt(E/(4*G))/(2*tem))/(tem^2);
g7 = P-PC;
