function [z idx] = get_extratarget(c0,c1)
if length(c0)~=3||length(c1)~=3
    error('the extraclass-vector must be 3 dimensional')
end
y=csvread('targets.csv');
y0=y(:,c0~=2);
y1=y(:,c1~=2);
c0=c0(c0~=2);
c1=c1(c1~=2);
n=size(y,1);
count=1;
idx=zeros(n,1);
for i=1:n
    if sum(y0(i,:)==c0)==length(c0)
        z(count)=0;
        count=count+1;
        idx(i)=1;
    elseif sum(y1(i,:)==c1)==length(c1)
        z(count)=1;
        count=count+1;
        idx(i)=1;
    end
end
z=reshape(z,length(z),1);

