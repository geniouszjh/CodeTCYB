%Distributed interval estimation for continuous-time linear systems based
%on robust observer and interval 
clc
clear all
close all
%system parameters
N = 6;%nodes
A = kron(eye(N),[-1 0.2132;0 -0.5]);
B = eye(12);
C1 = kron([1 0 0 0 0 0],[1 0]);
C2 = kron([0 1 0 0 0 0],[1 0]);
C3 = kron([0 0 -1 0 0 0],[1 0]);
C4 = kron([0 0 0 1 0 0],[1 0]);
C5 = kron([0 0 0 0 1 0],[1 0]);
C6 = kron([0 0 0 0 0 1],[1 0]);
C = [C1;C2;C3;C4;C5;C6];
E1=1;E2=1;E3=1;E4=1;E5=1;E6=1;
E = blkdiag(E1,E2,E3,E4,E5,E6);

%dimension
nx = size(A,1); 
nm = size(C1,1);
nq = size(E1,2);
nw = size(B,2);

%Laplace matrix
L = [1 0 0 0 0 -1;
     -1 1 0 0 0 0;
     0 -1 1 0 0 0;
     0 0 -1 1 0 0;
     0 0 0 -1 1 0;
     0 0 0 0 -1 1]; 


O1 = obsv(A,C1);
[Q1,Sig1,T1t]=svd(O1);
T1=T1t;
v1=rank(O1);
Z1=T1(:,1:v1);
U1=T1(:,v1+1:nx);


O2 = obsv(A,C2);
[Q2,Sig2,T2t]=svd(O2);
T2=T2t;
v2=rank(O2);
Z2=T2(:,1:v2);
U2=T2(:,v2+1:nx);


O3 = obsv(A,C3);
[Q3,Sig3,T3t]=svd(O3);
T3=T3t;
v3=rank(O3);
Z3=T3(:,1:v3);
U3=T3(:,v3+1:nx);


O4 = obsv(A,C4);
[Q4,Sig4,T4t]=svd(O4);
T4=T4t;
v4=rank(O4);
Z4=T4(:,1:v4);
U4=T4(:,v4+1:nx);


O5 = obsv(A,C5);
[Q5,Sig5,T5t]=svd(O5);
T5=T5t;
v5=rank(O5);
Z5=T5(:,1:v5);
U5=T5(:,v5+1:nx);

O6 = obsv(A,C6);
[Q6,Sig6,T6t]=svd(O6);
T6=T6t;
v6=rank(O6);
Z6=T6(:,1:v6);
U6=T6(:,v6+1:nx);

%(Aio,Cio)
AT1 = T1'*A*T1;
A1o = AT1(1:v1,1:v1);
A1r = AT1(v1+1:nx,1:v1);
A1u = AT1(v1+1:nx,v1+1:nx);
C1T1 = C1*T1;
C1o = C1T1(:,1:v1);

AT2 = T2'*A*T2;
A2o = AT2(1:v2,1:v2);
A2r = AT2(v2+1:nx,1:v2);
A2u = AT2(v2+1:nx,v2+1:nx);
C2T2 = C2*T2;
C2o = C2T2(:,1:v2);

AT3 = T3'*A*T3;
A3o = AT3(1:v3,1:v3);
A3r = AT3(v3+1:nx,1:v3);
A3u = AT3(v3+1:nx,v3+1:nx);
C3T3 = C3*T3;
C3o = C3T3(:,1:v3);

AT4 = T4'*A*T4;
A4o = AT4(1:v4,1:v4);
A4r = AT4(v4+1:nx,1:v4);
A4u = AT4(v4+1:nx,v4+1:nx);
C4T4 = C4*T4;
C4o = C4T4(:,1:v4);

AT5 = T5'*A*T5;
A5o = AT5(1:v5,1:v5);
A5r = AT5(v5+1:nx,1:v5);
A5u = AT5(v5+1:nx,v5+1:nx);
C5T5 = C5*T5;
C5o = C5T5(:,1:v5);

AT6 = T6'*A*T6;
A6o = AT6(1:v6,1:v6);
A6r = AT6(v6+1:nx,1:v6);
A6u = AT6(v6+1:nx,v6+1:nx);
C6T6 = C6*T6;
C6o = C6T6(:,1:v6);

%H_\infty observer design
Tall = blkdiag(T1,T2,T3,T4,T5,T6);

P1o = sdpvar(v1,v1,'symmetric');P2o = sdpvar(v2,v2,'symmetric');P3o = sdpvar(v3,v3,'symmetric');
P4o = sdpvar(v4,v4,'symmetric');P5o = sdpvar(v5,v5,'symmetric');P6o = sdpvar(v6,v6,'symmetric');

P1u=eye(nx-v1);P2u=eye(nx-v2);P3u=eye(nx-v3);P4u=eye(nx-v4);P5u=eye(nx-v5);P6u=eye(nx-v6);

P1 = blkdiag(P1o,P1u);P2 = blkdiag(P2o,P2u);P3 = blkdiag(P3o,P3u);
P4 = blkdiag(P4o,P4u);P5 = blkdiag(P5o,P5u);P6 = blkdiag(P6o,P6u);

W1o = sdpvar(v1,nm);W2o = sdpvar(v2,nm);W3o = sdpvar(v3,nm);
W4o = sdpvar(v4,nm);W5o = sdpvar(v5,nm);W6o = sdpvar(v6,nm);

M1=sdpvar(nx-v1,nx-v1);M2=sdpvar(nx-v2,nx-v2);M3=sdpvar(nx-v3,nx-v3);
M4=sdpvar(nx-v4,nx-v4);M5=sdpvar(nx-v5,nx-v5);M6=sdpvar(nx-v6,nx-v6);

F1 = [zeros(v1,nx); M1*U1'];F2 = [zeros(v2,nx); M2*U2'];F3 = [zeros(v3,nx); M3*U3'];
F4 = [zeros(v4,nx); M4*U4'];F5 = [zeros(v5,nx); M5*U5'];F6 = [zeros(v6,nx); M6*U6'];

P=blkdiag(P1,P2,P3,P4,P5,P6);

Aaug1=[P1o*A1o-W1o*C1o zeros(v1,nx-v1);
       P1u*A1r P1u*A1u];
Aaug2=[P2o*A2o-W2o*C2o zeros(v1,nx-v2);
       P2u*A2r P2u*A2u];
Aaug3=[P3o*A3o-W3o*C3o zeros(v1,nx-v3);
       P3u*A3r P3u*A3u];
Aaug4=[P4o*A4o-W4o*C4o zeros(v1,nx-v4);
       P4u*A4r P4u*A4u];
Aaug5=[P5o*A5o-W5o*C5o zeros(v1,nx-v5);
       P5u*A5r P5u*A5u];
Aaug6=[P6o*A6o-W6o*C6o zeros(v1,nx-v6);
       P6u*A6r P6u*A6u];
 
Aaug=blkdiag(Aaug1,Aaug2,Aaug3,Aaug4,Aaug5,Aaug6)+blkdiag(F1,F2,F3,F4,F5,F6)*kron(L,eye(nx))*Tall;

H1=[-W1o*E1;zeros(nx-v1,nq);];
H2=[-W2o*E2;zeros(nx-v2,nq);];
H3=[-W3o*E3;zeros(nx-v3,nq);];
H4=[-W4o*E4;zeros(nx-v4,nq);];
H5=[-W5o*E5;zeros(nx-v5,nq);];
H6=[-W6o*E6;zeros(nx-v6,nq);];

H=blkdiag(H1,H2,H3,H4,H5,H6);
Theta=[H,P*Tall*kron(ones(N,1),B)];

mu=sdpvar(1,1);

%LMI
LMI1=[(Aaug+Aaug')+eye(nx*N),Theta;
     (Theta)',-mu*eye(nw+nq*N)];
con=[LMI1<=-0.01;P>=0.01;mu>=0.01;];

ops = sdpsettings('solver','sdpt3','verbose',1);
optimize(con,mu,ops);

P1o=value(P1o);P2o=value(P2o);P3o=value(P3o);
P4o=value(P4o);P5o=value(P5o);P6o=value(P6o);

W1o=value(W1o);W2o=value(W2o);W3o=value(W3o);
W4o=value(W4o);W5o=value(W5o);W6o=value(W6o);

L1 = T1*[pinv(P1o)*W1o;zeros(nx-v1,1)];
L2 = T2*[pinv(P2o)*W2o;zeros(nx-v2,1)];
L3 = T3*[pinv(P3o)*W3o;zeros(nx-v3,1)];
L4 = T4*[pinv(P4o)*W4o;zeros(nx-v4,1)];
L5 = T5*[pinv(P5o)*W5o;zeros(nx-v5,1)];
L6 = T6*[pinv(P6o)*W6o;zeros(nx-v6,1)];

M1=value(M1);M2=value(M2);M3=value(M3);

M4=value(M4);M5=value(M5);M6=value(M6);


