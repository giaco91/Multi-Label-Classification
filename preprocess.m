function X_prepro= preprocess(X,y,train_or_test,method,id)
%normalization of the features
if strcmp(train_or_test,'train')==1
    s2=size(X,2);
    X_prepro=X;

    %shrink to maximum featuresize = 1
    if strcmp(method,'shrink')==1
        shrink=ones(s2,1);
        for i=1:s2
            m=max(X(:,i));
            if m>1e-6
                shrink(i)=1/m;
                X_prepro(:,i)=X(:,i)*shrink(i);
            else
                shrink(i)=0;
                X_prepro(:,i)=X(:,i)*shrink(i);
            end
        end
        csvwrite(strcat('shrink_',id,'.csv'),shrink);

    %normalize zero mean and one variance
    elseif strcmp(method,'norm')==1
        [X_prepro mu sigma]=zscore(X,0,1);
        csvwrite(strcat('norm_',id,'.csv'),[mu;sigma]);
    
    %center the data (zero mean)
    elseif strcmp(method,'center')==1
        m=mean(X,1);
        X_prepro=X-repmat(m,size(X,1),1);
        csvwrite(strcat('center_',id,'.csv'),m);
    else
        error('choose a valid preprocess method');
    end
else
    if strcmp(method,'shrink')==1
        shrink=csvread(strcat('shrink_',id,'.csv'));
        X_prepro=X*diag(shrink);
    elseif strcmp(method,'norm')==1
        norm=csvread(strcat('norm_',id,'.csv'));
        norm(2,norm(2,:)==0)=1;
        X_prepro=(X-repmat(norm(1,:),size(X,1),1))*diag(norm(2,:).^-1);
    elseif strcmp(method,'center')==1
        m=csvread(strcat('center_',id,'.csv'));
        X_prepro=X-repmat(m,size(X,1),1);
    else
        error('choose a valid preprocess method');
    end
end



