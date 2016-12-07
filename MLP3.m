clear all
close all
 
% Hyperparameter
submission=0;
method='NN';%NN,SVM,LR,LRlasso
class=3;%1=sex,2=age,3=health
cv=10;%0 for no cv

%read in target-vector
y=csvread('targets.csv');
y_train=y(:,class);

%extract features

%hypocampus
hv=Feature_select('train',[49 71;100 112;53 65],'voxvar',0,0);
hdm=Feature_select('train',[49 71;100 112;53 65],'ivox',0,[20 140]);
hbm=Feature_select('train',[49 71;100 112;53 65],'ivox',0,[420 1420]);
hdv=Feature_select('train',[49 71;100 112;53 65],'ivoxvar',0,[20 140]);
hbv=Feature_select('train',[49 71;100 112;53 65],'ivoxvar',0,[420 1420]);

%inner ventricle
iv=Feature_select('train',[67 108;116 144;77 90],'voxvar',0,0);
idm=Feature_select('train',[67 108;116 144;77 90],'ivox',0,[20 120]);
ibm=Feature_select('train',[67 108;116 144;77 90],'ivox',0,[920 1721]);
idv=Feature_select('train',[67 108;116 144;77 90],'ivoxvar',0,[20 120]);
ibv=Feature_select('train',[67 108;116 144;77 90],'ivoxvar',0,[920 1721]);

%overall
av=Feature_select('train',[1 89;1 208;1 176],'voxvar',0,0);
adm=Feature_select('train',[1 89;1 208;1 176],'ivox',0,[70 320]);
abm=Feature_select('train',[1 89;1 208;1 176],'ivox',0,[920 1420]);
adv=Feature_select('train',[1 89;1 208;1 176],'ivoxvar',0,[70 320]);
abv=Feature_select('train',[1 89;1 208;1 176],'ivoxvar',0,[920 1420]);

X=[hv hdm hbm hdv hbv iv idm ibm idv ibv av adm abm adv abv];
X=preprocess(X,y,'train','norm','id');
X_train=X;
if submission==1
        
        %hypocampus
        thv=Feature_select('test',[49 71;100 112;53 65],'voxvar',0,0);
        thdm=Feature_select('test',[49 71;100 112;53 65],'ivox',0,[20 140]);
        thbm=Feature_select('test',[49 71;100 112;53 65],'ivox',0,[420 1420]);
        thdv=Feature_select('test',[49 71;100 112;53 65],'ivoxvar',0,[20 140]);
        thbv=Feature_select('test',[49 71;100 112;53 65],'ivoxvar',0,[420 1420]);
        
        %inner ventricle
        tiv=Feature_select('test',[67 108;116 144;77 90],'voxvar',0,0);
        tidm=Feature_select('test',[67 108;116 144;77 90],'ivox',0,[20 120]);
        tibm=Feature_select('test',[67 108;116 144;77 90],'ivox',0,[920 1721]);
        tidv=Feature_select('test',[67 108;116 144;77 90],'ivoxvar',0,[20 120]);
        tibv=Feature_select('test',[67 108;116 144;77 90],'ivoxvar',0,[920 1721]);

        %over all
        tav=Feature_select('test',[1 89;1 208;1 176],'voxvar',0,0);
        tadm=Feature_select('test',[1 89;1 208;1 176],'ivox',0,[70 320]);
        tabm=Feature_select('test',[1 89;1 208;1 176],'ivox',0,[920 1420]);
        tadv=Feature_select('test',[1 89;1 208;1 176],'ivoxvar',0,[70 320]);
        tabv=Feature_select('test',[1 89;1 208;1 176],'ivoxvar',0,[920 1420]);

        
        Y=[thv thdm thbm thdv thbv tiv tidm tibm tidv tibv tav tadm tabm tadv tabv];
        Y=preprocess(Y,'void','test','norm','id');
        X_test=Y;
end
    
[ClObj void perf]=classifier(X_train,y_train,submission,method,cv,'train','void');
CV_mean_med_std=perf'
if submission==1
        [void yhat void]=classifier(X_test,'void',submission,method,cv,'test',ClObj);
        header={'"ID"','"Prediction"'};
        submission_data=[[1:length(yhat)]',yhat];
        csvwrite_with_headers('submission.csv',submission_data,header)
end


if size(X_train,2)==2
    CLplotter(ClObj,X_train,y_train,'train',method);
end

% 
% figure(4)
%         f=[1 6];
%         plot(X(y==0,f(1)),X(y==0,f(2)),'*r')
%         hold on
%         plot(X(y==1,f(1)),X(y==1,f(2)),'*b')
%         hold off
%         plot(Y(:,f(1)),Y(:,f(2)),'*g')
%         hold off