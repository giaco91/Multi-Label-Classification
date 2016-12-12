clear all
close all
 
% Hyperparameter
submission=0;
method='LR';%NN,SVM,LR,LRlasso
class=[1 2 3];%1=sex,2=age,3=health, void: 110 010
extraclass=1;%the possibility to define extra classes at around line 20
cv=10;%0 for no cv

if submission==1;
    class=[1 2 3];
    extraclass=0;
end

%read in target-vector
y=csvread('targets.csv');
y_train=y(:,class);
if extraclass==1
    %i.e % [2 0 0],[2 0 1] means 0=old-sick, 1=old-healthy
    [z_train idx]=get_extratarget([2 0 0],[2 0 1]);%(2 for arbitrary)   
end
    

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
X=[X(:,5) X(:,3)];
X=preprocess(X,y,'train','norm','id');
X_train=X;
if extraclass==1
    X_train_extra=X(idx==1,:);
end

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

CV_mean_med_std=zeros(3,1);
for i=1:length(class)
    [ClObj void perf]=classifier(X_train,y_train(:,i),submission,method,cv,'train','void');
    save(strcat('ClObj_',num2str(i),'.mat'),'ClObj');
    CV_mean_med_std=CV_mean_med_std+perf';
end
CV_mean_med_std=CV_mean_med_std/length(class)

 
 if extraclass==1
        [ClObj void perf]=classifier(X_train_extra,z_train,submission,method,cv,'train','void');
        save(strcat('ClObj_extra',num2str(1),'.mat'),'ClObj');
        CV_mean_med_std_extra=perf'
 end

if submission==1
    yhat=[];
    for i=1:length(class)
        CO=load(strcat('ClObj_',num2str(i),'.mat'));
        [void yhat(:,i) void]=classifier(X_test,'void',submission,method,cv,...
            'test',CO.ClObj);
    end
    
    figure(1)
    hist(yhat(:,1))
%     yhat = round(process(yhat));
 
    %check if there are young-sick classes
    a=0;
    for i=1:size(yhat,1)
        oli=yhat(i,2:3)==[1 0];
        if sum(oli)==2
            a=a+1;
        end
    end
    if a>0
        warning(strcat('there_are_',num2str(a),'_young-sick_classifications'));
    else
        'there are no young-sick classes'
    end
    
    %write file
    header={'ID','Sample','Label','Predicted'};
    submission_data=cell(413,4);
    Class={'gender','age','health'};
    bool={'False','True'};
    fid = fopen('submission.csv','w');
    fprintf(fid,'%s,%s,%s,%s\n',header{1:4});
    for i=1:3:3*length(yhat)
        for j=1:3
            fprintf(fid,'%d,%d,%s,%s\n',i+j-2,(i-1)/3,Class{j},bool{(yhat((i-1)/3+1,j)==1)+1});
        end
    end
    fclose(fid);
end

if size(X_train,2)==2 && strcmp(method,'NN')==0
    'plotting the featurespace and boundary for a class'
    class=2;
    CO=load(strcat('ClObj_',num2str(class),'.mat'));
    CLplotter(CO.ClObj,X_train,y_train(:,class),'train',method);
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