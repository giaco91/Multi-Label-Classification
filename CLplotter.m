function dummy = CLplotter(ClObj,X,y,train_or_test,method)
rng(1);
    d1 = (max(X(:,1))-min(X(:,1)))/100;
    d2 = (max(X(:,2))-min(X(:,2)))/100;
    [x1Grid,x2Grid] = meshgrid(min(X(:,1)):d1:max(X(:,1)),...
    min(X(:,2)):d2:max(X(:,2)));
    xGrid = [x1Grid(:),x2Grid(:)];        % The grid
    if strcmp(method,'SVM')==1
        [~,scores] = predict(ClObj,xGrid); % The scores
        scores=scores(:,2);
    elseif strcmp(method,'LR')==1 || strcmp(method,'LRlasso')==1
        scores=glmval(ClObj,xGrid,'logit');
    end
    
    % Plot the data and the decision boundary
    figure(strcmp(train_or_test,'test')+1);
    h(1:2) = gscatter(X(:,1),X(:,2),y,'rb','.');
    hold on
    contour(x1Grid,x2Grid,reshape(scores,size(x1Grid)),[0.5 0.5],'k');
    title(train_or_test)
    legend(h,{'0','1'});

    hold off

end

