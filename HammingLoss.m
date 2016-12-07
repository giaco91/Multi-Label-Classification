function e = HammingLoss(y_hat,y)
    D=size(y,1);
    L=size(y,2);
    e=0;
    for i=1:D
        for j=1:L
            e=e+(y_hat(i,j)~=y(i,j));
        end
    end
    e=e/(D*L);
end

