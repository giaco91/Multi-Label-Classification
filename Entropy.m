function E = Entropy(X)
%takes an nxm Matrix X and returns a nx1 vector E of the row-wise entropy
%the rows of X are intepreted as (unnrmalized) discrete probability distributions
n=size(X,1);
m=size(X,2);
M=sum(X,2);
for i=1:n
    X(i,:)=X(i,:)/(M(i)+1e-30);
end
E=zeros(n,1);
for i=1:m
    E=E-X(:,i).*log2(X(:,i)+1e-30);
end

