function C = Crossentropy(y,yhat)
for i=1:length(yhat)
    if yhat(i)<1e-12
        yhat(i)=1e-12;
    elseif yhat(i)>1-1e-12
        yhat(i)=1-1e-12;
    end
        
    C = sum(-y .* log(yhat) - (1-y) .* log(1-yhat))/length(y);
end

