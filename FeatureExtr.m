function Xextr = FeatureExtr(X,y,N,method,train_or_test,id)
%methods:PCA,MI,MaxReg,PCA_MaxReg,FDA
 s=size(X);
if strcmp(train_or_test,'train')==1
    if N>s(2)
        N=s(2);
        Xextr=X;
    else
    if strcmp(method,'MI')==1
        Xextr=zeros(s(1),N);
        mutInf=zeros(s(2),1);
        for i=1:s(2)
            mutInf(i)=mutInfo(round(X(:,i)),round(y));
        end
        [sorted_mutInf I]=sort(mutInf);%sorted_mutInf(1)=smallest
        I=I(end-N+1:end);
        csvwrite('Mutextr_idx.csv',I);
        for i=1:N
            Xextr(:,i)=X(:,I(i));
        end
    elseif strcmp(method,'PCA')==1
        X=X';
        [U, S, V] = svd(X);
        B=U(:,1:N);
        Y=B'*X;
        Xextr=Y';
        csvwrite(strcat('PCAextr_proj_',id,'.csv'),B);
    elseif strcmp(method,'MaxReg')==1
        Xextr=zeros(s(1),N);
        MaxReg=zeros(s(2),1);
        for i=1:s(2)
            MaxReg(i)=score_logreg(X(:,i),y);
        end
        [sorted_MaxReg I]=sort(MaxReg);%sorted_MaxReg(1)=smallest
        strcat('worst score: ',num2str(sorted_MaxReg(N)))
        strcat('best score: ',num2str(sorted_MaxReg(1)))
        hist(sorted_MaxReg)
        sorted_MaxReg=sorted_MaxReg(1:N);
        IN=I(1:N);
        length(IN);
        csvwrite(strcat('MaxRegextr_idx_',id,'.csv'),IN);
        Xextr=X(:,IN);
%       Xextr=X(:,I);
              
    elseif strcmp(method,'PCA_MaxReg')==1
        alpha=0;
        scale=zeros(s(2),1);
        X=preprocess(X,y,train_or_test,'norm',id);
        for i=1:s(2)
            scale(i)=score_logreg(X(:,i),y)^-alpha;
            X(:,i)=X(:,i)*scale(i);
        end
        csvwrite(strcat('scale_',id,'.csv'),scale);
        Xextr=FeatureExtr(X,y,N,'PCA','train',id);
    elseif strcmp(method,'FDA')==1
        size(y)
        [X_FDA,W] = FDA(X',y,N);
        Xextr=real(X_FDA');
        csvwrite(strcat('FDA_proj_',id,'.csv'),W);
    else
        error('choose a valid feature extraction method')
    end
    end
else
    if strcmp(method,'MI')==1
        I=csvread(strcat('Mutextr_idx_',id,'.csv'));
        Xextr=X(:,I);
    elseif strcmp(method,'PCA')==1
        X=X';
        B=csvread(strcat('PCAextr_proj_',id,'.csv'));
        Y=B'*X;
        Xextr=Y';
    elseif strcmp(method,'MaxReg')==1
        I=csvread(strcat('MaxRegextr_idx_',id,'.csv'));
        for i=1:length(I)
            Xextr(:,i)=X(:,I(i));
        end
    elseif strcmp(method,'PCA_MaxReg')==1
        X=preprocess(X,y,train_or_test,'norm',id);
        scale=csvread(strcat('scale_',id,'.csv'));
        B=csvread(strcat('PCAextr_proj_',id,'.csv'));
        X=X*diag(scale);
        Y=X*B;
        Xextr=Y;
    elseif strcmp(method,'FDA')==1
        W=csvread(strcat('FDA_proj_',id,'.csv'));
        Xextr=real(X*W);
    else
        error('choose a valid feature extraction method')
    end
end

