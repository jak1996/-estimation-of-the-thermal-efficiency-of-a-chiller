%progetto
clc;
clear all;
close all;
rng(30);
%%--------------------------------------------------------------------------
load Chiller_1.mat
figure(1);
scatter3(Qe,Tcond,COP);
title('Scatter dei dati di Chiller1 ');
xlabel('Qe');
ylabel('Tcond');
zlabel('COP');

%TRAIN E DEV SET
%varianza dei dati
VarQe = var(Qe);
VarTcond = var(Tcond);

%vettori contenenti gli indici per il training set e validation set
[trainInd,valInd,testInd] = dividerand(length(COP),0.7,0.3,0);

%vettori dei dati di training e validation set delle variabili Qe,COP,Tcond
COPtrain = COP(trainInd); 
COPdev = COP(valInd);
Qetrain = Qe(trainInd);
Qedev = Qe(valInd);
Tcondtrain = Tcond(trainInd);
Tconddev = Tcond(valInd);

figure(2);
scatter3(Qedev,Tconddev,COPdev,'r');
hold on 
scatter3(Qetrain,Tcondtrain,COPtrain,'b');
title('Dati di training e di validation');
xlabel('Qe');
ylabel('Tcond');
zlabel('COP');
legend('Dati di validation', 'Dati di training'); 

%%--------------------------------------------------------------------------
%% modello lineare 
phitrainLinear = [ones(length(COPtrain),1) Qetrain Tcondtrain];
betaLinear = lscov(phitrainLinear,COPtrain);
COPtrainLinear = phitrainLinear*betaLinear;
phivalLinear = [ones(length(COPdev),1) Qedev Tconddev];
COPdevLinear = phivalLinear*betaLinear;

[Xg,Yg]=meshgrid(linspace(min(Qetrain),max(Qetrain),10),linspace(min(Tcondtrain),max(Tcondtrain),10));
phigSquare=[ones(length(Xg(:)),1) Xg(:) Yg(:)];
ZgLinear=phigSquare*betaLinear;

figure(3);
scatter3(Qetrain,Tcondtrain,COPtrain, 'b');
hold on 
scatter3(Qedev,Tconddev,COPdev, 'r');
hold on
surf(Xg,Yg,reshape(ZgLinear,size(Xg)));
title('Dati di train e validation sovrapposti con il modello lineare');
xlabel('Qe');
ylabel('Tcond');
zlabel('COP');
legend('Dati di training', 'Dati di validation'); 

%errori
epsilon_devLinear = COPdev - COPdevLinear;
% Root Mean Square Error
RMSELinear = sqrt(mean((epsilon_devLinear).^2));
% SSR
SSR_devLinear = epsilon_devLinear'*epsilon_devLinear;
% FPE
Ndev = length(COPdev);
q = length(betaLinear);
FPE_devLinear = ((Ndev+q)/(Ndev-q))* SSR_devLinear;
% AKAIKE
AIC_devLinear = (2*q/Ndev) + log(SSR_devLinear); 
%MINIMUM DESCRIPTION LENGTH
MDL_devLinear = ((log(Ndev)*q)/Ndev) + log(SSR_devLinear);

%% modello quadratico
phitrainSquare = [ones(length(COPtrain),1) Qetrain Tcondtrain Tcondtrain.*Qetrain Qetrain.^2 Tcondtrain.^2];
betaSquare = lscov(phitrainSquare,COPtrain);
COPtrainSquare = phitrainSquare*betaSquare;

phivaldevSquare = [ones(length(COPdev),1) Qedev Tconddev Tconddev.*Qedev Qedev.^2 Tconddev.^2];
COPdevSquare = phivaldevSquare*betaSquare;

phigSquare=[ones(length(Xg(:)),1) Xg(:) Yg(:) Xg(:).*Yg(:) Xg(:).^2 Yg(:).^2];
ZgSquare=phigSquare*betaSquare;

figure(4);
scatter3(Qetrain,Tcondtrain,COPtrain, 'b');
hold on 
scatter3(Qedev,Tconddev,COPdev, 'r');
hold on
surf(Xg,Yg,reshape(ZgSquare,size(Xg)));
title('Dati di train e validation sovrapposti con il modello polinomiale quadratico');
xlabel('Qe');
ylabel('Tcond');
zlabel('COP');
legend('Dati di training', 'Dati di validation'); 

%errore
epsilon_devSquare = COPdev - COPdevSquare;
% Root Mean Square Error
RMSE_devSquare = sqrt(mean((epsilon_devSquare).^2));
% SSR
SSR_devSquare = epsilon_devSquare'*epsilon_devSquare;
% FPE
Ndev = length(COPdev);
q = length(betaSquare);
FPE_devSquare = ((Ndev+q)/(Ndev-q))* SSR_devSquare;
% AKAIKE
AIC_devSquare = (2*q/Ndev) + log(SSR_devSquare); 
%MINIMUM DESCRIPTION LENGTH
MDL_devSquare = ((log(Ndev)*q)/Ndev) + log(SSR_devSquare);

%% modello cubico
phitrainCubic = [ones(length(COPtrain),1) Qetrain Tcondtrain Tcondtrain.*Qetrain Qetrain.^2 Tcondtrain.^2 Qetrain.^3 Qetrain.^2.*Tcondtrain Qetrain.*Tcondtrain.^2 Tcondtrain.^3];
betaCubic = lscov(phitrainCubic,COPtrain);
COPtrainCubic = phitrainCubic*betaCubic;

phivaldevCubic = [ones(length(COPdev),1) Qedev Tconddev Tconddev.*Qedev Qedev.^2 Tconddev.^2 Qedev.^3 Qedev.^2.*Tconddev Qedev.*Tconddev.^2 Tconddev.^3];
COPdevCubic = phivaldevCubic*betaCubic;

phigCubic=[ones(length(Xg(:)),1) Xg(:) Yg(:) Xg(:).*Yg(:) Xg(:).^2 Yg(:).^2 Xg(:).^3 Xg(:).^2.*Yg(:) Xg(:).*Yg(:).^2 Yg(:).^3];
ZgCubic=phigCubic*betaCubic;

figure(5);
scatter3(Qetrain,Tcondtrain,COPtrain, 'b');
hold on 
scatter3(Qedev,Tconddev,COPdev, 'r');
hold on
surf(Xg,Yg,reshape(ZgCubic,size(Xg)));
title('Dati di train e validation sovrapposti con il modello polinomiale cubico');
xlabel('Qe');
ylabel('Tcond');
zlabel('COP');
legend('Dati di training', 'Dati di validation'); 

%errore
epsilon_devCubic = COPdev - COPdevCubic;
% Root Mean Square Error
RMSE_devCubic = sqrt(mean((epsilon_devCubic).^2));
% SSR
SSR_devCubic = epsilon_devCubic'*epsilon_devCubic;
% FPE
Ndev = length(COPdev);
q = length(betaCubic);
FPE_devCubic = ((Ndev+q)/(Ndev-q))* SSR_devCubic;
% AKAIKE
AIC_devCubic = (2*q/Ndev) + log(SSR_devCubic); 
%MINIMUM DESCRIPTION LENGTH
MDL_devCubic = ((log(Ndev)*q)/Ndev) + log(SSR_devCubic);

%% modello quarto
phitrainFour = [ones(length(COPtrain),1) Qetrain Tcondtrain Tcondtrain.*Qetrain Qetrain.^2 Tcondtrain.^2 Qetrain.^3 Qetrain.^2.*Tcondtrain Qetrain.*Tcondtrain.^2 Tcondtrain.^3 Qetrain.^4 Qetrain.^3.*Tcondtrain Qetrain.*Tcondtrain.^3 Qetrain.^2.*Tcondtrain.^2 Tcondtrain.^4];
betaFour = lscov(phitrainFour,COPtrain);
COPtrainFour = phitrainFour*betaFour;

phivaldevFour = [ones(length(COPdev),1) Qedev Tconddev Tconddev.*Qedev Qedev.^2 Tconddev.^2 Qedev.^3 Qedev.^2.*Tconddev Qedev.*Tconddev.^2 Tconddev.^3 Qedev.^4 Qedev.^3.*Tconddev Qedev.*Tconddev.^3 Qedev.^2.*Tconddev.^2 Tconddev.^4];
COPdevFour = phivaldevFour*betaFour;

phigFour=[ones(length(Xg(:)),1) Xg(:) Yg(:) Xg(:).*Yg(:) Xg(:).^2 Yg(:).^2 Xg(:).^3 Xg(:).^2.*Yg(:) Xg(:).*Yg(:).^2 Yg(:).^3 Xg(:).^4 Xg(:).^3.*Yg(:) Xg(:).*Yg(:).^3 Xg(:).^2.*Yg(:).^2 Yg(:).^4];
ZgFour=phigFour*betaFour;

figure(6);
scatter3(Qetrain,Tcondtrain,COPtrain, 'b');
hold on 
scatter3(Qedev,Tconddev,COPdev, 'r');
hold on
surf(Xg,Yg,reshape(ZgFour,size(Xg)));
title('Dati di train e validation sovrapposti con il modello polinomiale di quarto grado');
xlabel('Qe');
ylabel('Tcond');
zlabel('COP');
legend('Dati di training', 'Dati di validation'); 

%errore
epsilon_devFour = COPdev - COPdevFour;
% Root Mean Square Error
RMSE_devFour = sqrt(mean((epsilon_devFour).^2));
% SSR
SSR_devFour = epsilon_devFour'*epsilon_devFour;
% FPE
Ndev = length(COPdev);
q = length(betaFour);
FPE_devFour = ((Ndev+q)/(Ndev-q))* SSR_devFour;
% AKAIKE
AIC_devFour = (2*q/Ndev) + log(SSR_devFour); 
%MINIMUM DESCRIPTION LENGTH
MDL_devFour = ((log(Ndev)*q)/Ndev) + log(SSR_devFour);

%% modello quinto
phitrainFifth = [ones(length(COPtrain),1) Qetrain Tcondtrain Tcondtrain.*Qetrain Qetrain.^2 Tcondtrain.^2 Qetrain.^3 Qetrain.^2.*Tcondtrain Qetrain.*Tcondtrain.^2 Tcondtrain.^3 Qetrain.^4 Qetrain.^3.*Tcondtrain Qetrain.*Tcondtrain.^3 Qetrain.^2.*Tcondtrain.^2 Tcondtrain.^4 Qetrain.^5 Qetrain.^3.*Tcondtrain.^2 Qetrain.^2.*Tcondtrain.^3 Qetrain.^4.*Tcondtrain.^1 Qetrain.*Tcondtrain.^4 Tcondtrain.^5];
betaFifth = lscov(phitrainFifth,COPtrain);
COPtrainFifth = phitrainFifth*betaFifth;

phivaldevFifth = [ones(length(COPdev),1) Qedev Tconddev Tconddev.*Qedev Qedev.^2 Tconddev.^2 Qedev.^3 Qedev.^2.*Tconddev Qedev.*Tconddev.^2 Tconddev.^3 Qedev.^4 Qedev.^3.*Tconddev Qedev.*Tconddev.^3 Qedev.^2.*Tconddev.^2 Tconddev.^4 Qedev.^5 Qedev.^3.*Tconddev.^2 Qedev.^2.*Tconddev.^3 Qedev.^4.*Tconddev.^1 Qedev.*Tconddev.^4 Tconddev.^5];
COPdevFifth = phivaldevFifth*betaFifth;

phigFifth=[ones(length(Xg(:)),1) Xg(:) Yg(:) Xg(:).*Yg(:) Xg(:).^2 Yg(:).^2 Xg(:).^3 Xg(:).^2.*Yg(:) Xg(:).*Yg(:).^2 Yg(:).^3 Xg(:).^4 Xg(:).^3.*Yg(:) Xg(:).*Yg(:).^3 Xg(:).^2.*Yg(:).^2 Yg(:).^4 Xg(:).^5 Xg(:).^3.*Yg(:).^2 Xg(:).^2.*Yg(:).^3 Xg(:).^4.*Yg(:).^1 Xg(:).^1.*Yg(:).^4 Yg(:).^5];
ZgFifth=phigFifth*betaFifth;

figure(7);
scatter3(Qetrain,Tcondtrain,COPtrain, 'b');
hold on 
scatter3(Qedev,Tconddev,COPdev, 'r');
hold on
surf(Xg,Yg,reshape(ZgFifth,size(Xg)));
title('Dati di train e validation sovrapposti con il modello polinomiale di quinto grado');
xlabel('Qe');
ylabel('Tcond');
zlabel('COP');
legend('Dati di training', 'Dati di validation'); 

%errore
epsilon_devFifth = COPdev - COPdevFifth;
% Root Mean Square Error
RMSE_devFifth = sqrt(mean((epsilon_devFifth).^2));
% SSR
SSR_devFifth = epsilon_devFifth'*epsilon_devFifth;
% FPE
Ndev = length(COPdev);
q = length(betaFifth);
FPE_devFifth = ((Ndev+q)/(Ndev-q))* SSR_devFifth;
% AKAIKE
AIC_devFifth = (2*q/Ndev) + log(SSR_devFifth); 
%MINIMUM DESCRIPTION LENGTH
MDL_devFifth = ((log(Ndev)*q)/Ndev) + log(SSR_devFifth);

%%------------------------------------------------------------------------
%% Stima del modello usando la Stepwise 
mdl = stepwiselm ([Qetrain Tcondtrain], COPtrain, 'poly11');
epsilon_mdl = COPdev - predict(mdl,[Qedev Tconddev]);
q0 = mdl.NumEstimatedCoefficients;
RMSE_mdl = sqrt(mean((epsilon_mdl).^2));
SSR_mdl = epsilon_mdl'*epsilon_mdl;
FPE_mdl =((Ndev+q0)/(Ndev-q0))*SSR_mdl;
AIC_mdl = (2*q0/Ndev) + log(SSR_mdl); 
MDL_mdl = ((log(Ndev)*q0)/Ndev) + log(SSR_mdl);

[Qegrid,Tcondgrid]=meshgrid(linspace(min(Qe),max(Qe),50),linspace(min(Tcond),max(Tcond),50));
COPgrid=[];
for j=linspace(min(Qe),max(Qe),50)
    COPgrid=[COPgrid;predict(mdl,[j*ones(50,1) linspace(min(Tcond),max(Tcond),50)'])];
end
figure(8);
surfc(Qegrid,Tcondgrid,reshape(COPgrid,size(Qegrid)));
hold on
scatter3(Qetrain,Tcondtrain,COPtrain, 'y');
hold on 
scatter3(Qedev,Tconddev,COPdev, 'r');
xlabel('Qe');
ylabel('Tcond');
zlabel('COP');
title('Stepwise modello lineare');

mdl1 = stepwiselm ([Qetrain Tcondtrain], COPtrain, 'poly22');
epsilon_mdl1=COPdev - predict(mdl1,[Qedev Tconddev]);
q1 = mdl1.NumEstimatedCoefficients;
RMSE_mdl1 = sqrt(mean((epsilon_mdl1).^2));
SSR_mdl1 = epsilon_mdl1'*epsilon_mdl1;
FPE_mdl1 = SSR_mdl1 * ((Ndev+q1)/(Ndev-q1));
AIC_mdl1 = (2*q1/Ndev) + log(SSR_mdl1); 
MDL_mdl1 = ((log(Ndev)*q1)/Ndev) + log(SSR_mdl1);

[Qegrid1,Tcondgrid1]=meshgrid(linspace(min(Qe),max(Qe),50),linspace(min(Tcond),max(Tcond),50));
COPgrid1=[];
for j=linspace(min(Qe),max(Qe),50)
    COPgrid1=[COPgrid1;predict(mdl1,[j*ones(50,1) linspace(min(Tcond),max(Tcond),50)'])];
end
figure(9)
surfc(Qegrid1,Tcondgrid1,reshape(COPgrid1,size(Qegrid1)));
title('Stepwise modello quadratico');
hold on
scatter3(Qetrain,Tcondtrain,COPtrain, 'y');
hold on 
scatter3(Qedev,Tconddev,COPdev, 'r');
xlabel('Qe');
ylabel('Tcond');
zlabel('COP');

mdl2 = stepwiselm ([Qetrain Tcondtrain], COPtrain, 'poly33');
epsilon_mdl2=COPdev - predict(mdl2,[Qedev Tconddev]);
q2 = mdl2.NumEstimatedCoefficients;
RMSE_mdl2 = sqrt(mean((epsilon_mdl2).^2));
SSR_mdl2 = epsilon_mdl2'*epsilon_mdl2;
FPE_mdl2 = SSR_mdl2 * ((Ndev+q2)/(Ndev-q2));
AIC_mdl2 = (2*q2/Ndev) + log(SSR_mdl2); 
MDL_mdl2 = ((log(Ndev)*q2)/Ndev) + log(SSR_mdl2);

[Qegrid2,Tcondgrid2]=meshgrid(linspace(min(Qe),max(Qe),50),linspace(min(Tcond),max(Tcond),50));
COPgrid2=[];
for j=linspace(min(Qe),max(Qe),50)
    COPgrid2=[COPgrid2;predict(mdl2,[j*ones(50,1) linspace(min(Tcond),max(Tcond),50)'])];
end
figure(10)
surfc(Qegrid2,Tcondgrid2,reshape(COPgrid2,size(Qegrid2)))
title('Stepwise modello cubico');
hold on
scatter3(Qetrain,Tcondtrain,COPtrain, 'y');
hold on 
scatter3(Qedev,Tconddev,COPdev, 'r');
xlabel('Qe');
ylabel('Tcond');
zlabel('COP');

mdl3 = stepwiselm ([Qetrain Tcondtrain], COPtrain, 'poly44');
epsilon_mdl3=COPdev - predict(mdl3,[Qedev Tconddev]);
q3 = mdl3.NumEstimatedCoefficients;
RMSE_mdl3 = sqrt(mean((epsilon_mdl3).^2));
SSR_mdl3 = epsilon_mdl3'*epsilon_mdl3;
FPE_mdl3 = SSR_mdl3 * ((Ndev+q3)/(Ndev-q3));
AIC_mdl3 = (2*q3/Ndev) + log(SSR_mdl3); 
MDL_mdl3 = ((log(Ndev)*q3)/Ndev) + log(SSR_mdl3);

[Qegrid3,Tcondgrid3]=meshgrid(linspace(min(Qe),max(Qe),50),linspace(min(Tcond),max(Tcond),50));
COPgrid3=[];
for j=linspace(min(Qe),max(Qe),50)
    COPgrid3=[COPgrid3;predict(mdl3,[j*ones(50,1) linspace(min(Tcond),max(Tcond),50)'])];
end
figure(11)
surfc(Qegrid3,Tcondgrid3,reshape(COPgrid3,size(Qegrid3)));
title('Stepwise modello quarto grado');
hold on
scatter3(Qetrain,Tcondtrain,COPtrain, 'y');
hold on 
scatter3(Qedev,Tconddev,COPdev, 'r');
xlabel('Qe');
ylabel('Tcond');
zlabel('COP');

mdl4 = stepwiselm ([Qetrain Tcondtrain], COPtrain, 'poly55');
epsilon_mdl4=COPdev - predict(mdl4,[Qedev Tconddev]);
q4 = mdl4.NumEstimatedCoefficients;
RMSE_mdl4 = sqrt(mean((epsilon_mdl4).^2));
SSR_mdl4 = epsilon_mdl4'*epsilon_mdl4;
FPE_mdl4 = SSR_mdl4 * ((Ndev+q4)/(Ndev-q4));
AIC_mdl4 = (2*q4/Ndev) + log(SSR_mdl4); 
MDL_mdl4 = ((log(Ndev)*q4)/Ndev) + log(SSR_mdl4);

[Qegrid4,Tcondgrid4]=meshgrid(linspace(min(Qe),max(Qe),50),linspace(min(Tcond),max(Tcond),50));
COPgrid4=[];
for j=linspace(min(Qe),max(Qe),50)
    COPgrid4=[COPgrid4;predict(mdl4,[j*ones(50,1) linspace(min(Tcond),max(Tcond),50)'])];
end
figure(12)
surfc(Qegrid4,Tcondgrid4,reshape(COPgrid4,size(Qegrid4)));
title('Stepwise modello quinto grado');
hold on
scatter3(Qetrain,Tcondtrain,COPtrain, 'y');
hold on 
scatter3(Qedev,Tconddev,COPdev, 'r');
xlabel('Qe');
ylabel('Tcond');
zlabel('COP');

%--------------------------------------------------------------------------
%% Rete neurale
net=feedforwardnet(5);
%training della rete
inputdataset_train=[Qetrain Tcondtrain];
net = train (net, inputdataset_train', COPtrain');
%verifica performance della rete con i dati di validation
inputdataset_validation=[Qedev Tconddev];
COPnet = net(inputdataset_validation');
NNrmse=sqrt(mse(net,COPdev',COPnet));
NNsse=sse(net,COPdev',COPnet);

figure(13);
[Xg,Yg]=meshgrid(linspace(min(Qe),max(Qe),10),linspace(min(Tcond),max(Tcond),10));
COP_grid= net([Xg(:) Yg(:)]');
surf(Xg, Yg, reshape(COP_grid , size(Xg)));
hold on
scatter3(Qedev,Tconddev, COPnet,'r');
title('Rete neurale con 5 neuroni');
xlabel('Qe');
ylabel('Tcond');
zlabel('COP');

net2=feedforwardnet(20);
inputdataset_train2=[Qetrain Tcondtrain];
net2 = train (net2, inputdataset_train2', COPtrain');
inputdataset_validation2=[Qedev Tconddev];
COPnet2 = net2(inputdataset_validation2');
NNrmse2=sqrt(mse(net2,COPdev',COPnet2));
NNsse2=sse(net2,COPdev',COPnet2);

figure(14);
COP_grid2= net2([Xg(:) Yg(:)]');
surf(Xg, Yg, reshape(COP_grid2 , size(Xg)));
hold on
scatter3(Qedev,Tconddev, COPnet,'r');
title('Rete neurale con 10 neuroni');
xlabel('Qe');
ylabel('Tcond');
zlabel('COP');
