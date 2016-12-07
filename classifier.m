function [ClObj yh_test perf] = classifier(X_tr,y,submission,method,cv,train_or_test,CLOBJ)
    %make sure, that the classpartitions are the same in crossvalidation-
    %and test-sets
    [sorted_y I]=sort(y);
    y_0=sorted_y(1:length(y)-sum(y));
    y_1=sorted_y(length(y)-sum(y):end);
    X_tr_sorted=X_tr(I,:);
    X_tr_0=X_tr_sorted(1:length(y)-sum(y),:);
    X_tr_1=X_tr_sorted(length(y)-sum(y):end,:);
    
    %amount of crossvalidation
    if strcmp(train_or_test,'test')==1
        cv=0;
    else
        yh_test=0;
    end
    if cv<10
        valblock_0=round(size(X_tr_0,1)/10);
        valblock_1=round(size(X_tr_1,1)/10);
    elseif cv>=10
        valblock_0=round(size(X_tr_0,1)/cv);
        valblock_1=round(size(X_tr_1,1)/cv);
    end
    if cv<=0 || submission==1
        CV=1;
    else
        CV=cv;
    end
    perf=[];
    for i=1:CV %the validation loop
        if submission==0 && cv>0 
            valbegin_0=round(((size(X_tr_0,1)-valblock_0)/cv)*i);
            valbegin_1=round(((size(X_tr_1,1)-valblock_1)/cv)*i);
            val_set_0=[valbegin_0 valbegin_0+valblock_0];
            val_set_1=[valbegin_1 valbegin_1+valblock_1];
            train_set_0=[1 val_set_0(1)-1 val_set_0(2)+1 size(X_tr_0,1)];
            train_set_1=[1 val_set_1(1)-1 val_set_1(2)+1 size(X_tr_1,1)];
            y_val_0=y_0(val_set_0(1):val_set_0(2),:);
            y_val_1=y_1(val_set_1(1):val_set_1(2),:);
            y_val=[y_val_0;y_val_1];
            X_val_0=X_tr_0(val_set_0(1):val_set_0(2),:);
            X_val_1=X_tr_1(val_set_1(1):val_set_1(2),:);
            X_val=[X_val_0;X_val_1];
            y_train_0=[y_0(train_set_0(1):train_set_0(2),:);y_0(train_set_0(3):train_set_0(4),:)];
            y_train_1=[y_1(train_set_1(1):train_set_1(2),:);y_1(train_set_1(3):train_set_1(4),:)];
            y_train=[y_train_0;y_train_1];
            X_train_0=[X_tr_0(train_set_0(1):train_set_0(2),:);X_tr_0(train_set_0(3):train_set_0(4),:)];
            X_train_1=[X_tr_1(train_set_1(1):train_set_1(2),:);X_tr_1(train_set_1(3):train_set_1(4),:)];
            X_train=[X_train_0;X_train_1];
            
        else 
            train_set=[1 2 3 size(X_tr,1)];
            if submission==1 && strcmp(train_or_test,'test')==1
                y_train=0;
            else
                y_train=[y(train_set(1):train_set(2),:);y(train_set(3):train_set(4),:)];
            end
            X_train=[X_tr(train_set(1):train_set(2),:);X_tr(train_set(3):train_set(4),:)];
        end
        
        %LR------
        if strcmp(method,'LR')==1
            if strcmp(train_or_test,'test')==1
               yh_test=round(glmval(CLOBJ,X_train,'logit'));
               ClObj=CLOBJ;
               if submission==0
                   score=HammingLoss(y_train,yh_test)
               end
            else
                [b,dev,stats] = glmfit(X_train,y_train,'binomial','link','logit');
                ClObj=b;
                if submission==0 && cv>0 
                    yhat = round(glmval(b,X_val,'logit'));
                    perf=[perf HammingLoss(y_val,yhat)];
                end
            end
        

        %LRlasso------
        elseif strcmp(method,'LRlasso')==1
            if strcmp(train_or_test,'test')==1
               yh_test=round(glmval(CLOBJ,X_train,'logit'));
               ClObj=CLOBJ;
               if submission==0
                   score=HammingLoss(y_train,yh_test)
               end
            else
%               find right lambda:
%                 [B,FitInfo] = lassoglm(X_train,y_train,'binomial','NumLambda',25,'CV',10);
%                 lambda=FitInfo.Lambda1SE
                lambda=0.02;
                [B,FitInfo] = lassoglm(X_train,y_train,'binomial','Lambda',lambda);
                b=[FitInfo.Intercept;B];
                ClObj=b;
                if submission==0 && cv>0 
                    yhat = round(glmval(b,X_val,'logit'));
                    perf=[perf HammingLoss(y_val,yhat)];
                end
            end


        %  SVM----------
        elseif strcmp(method,'SVM')==1
            rng(4)
            if strcmp(train_or_test,'test')==1
               [Decision,Posterior] = predict(CLOBJ,X_train);
               yh_test=Posterior(:,2);
               ClObj=CLOBJ;
               if submission==0
                   score=Crossentropy(y_train,yh_test)
               end
            else 
                %kernels: linear,quadratic,polynomial,rbf,mlp --- ,'KernelScale','auto'
                SVMModel = fitcsvm(X_train,y_train,'KernelFunction','linear');
                ClObj=SVMModel;
                if submission==0 && cv>0 
                    [Decision,Posterior] = predict(SVMModel,X_val);
                    yhat=Decision;
                    perf=[perf HammingLoss(y_val,yhat)];
                    strcat('Crossvalidation epoch: ',num2str(i),'/',num2str(cv))
                    loss=[mean(perf);median(perf);var(perf)^(1/2)]
                end
            end
          

    %         % Neural Netowrk---------------
        elseif strcmp(method,'NN')==1
            if strcmp(train_or_test,'test')==1
               yh_test=roundCLOBJ(X_train')';
               ClObj=CLOBJ;
               if submission==0
                   score=HammingLoss(y_train,round(yh_test(:,2)))
               end
            else
                rng(2)
                layers=[4];
                net=patternnet(layers);
                net.divideParam.trainRatio = 100/100;
                net.divideParam.valRatio = 0/100;
                net.divideParam.testRatio = 0/100;
                net.trainParam.epochs = 15;
                net.performParam.regularization = 0.29;
                y_train_net=[y_train==0 y_train];
                [net tr] = train(net,X_train',y_train_net');%train
                ClObj=net;
                if submission==0 && cv>0 
                    yhat = net(X_val')';%estimate
                    perf=[perf HammingLoss(y_val,round(yhat(:,2)))];
                end
            end
        else
            method
            error('choose a valid method')
        end
    end
    if submission==0 && strcmp(train_or_test,'test')==0 && cv>0
        perf=[mean(perf) median(perf) var(perf)^(1/2)];
        if cv>0
            ClObj=classifier(X_tr,y,submission,method,0,'train','void');
        end
    else
        perf=[0 0 0];
    end
end

