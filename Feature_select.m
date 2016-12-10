function X = Feature_select(train_or_test,B,spec,sb,interval)
    id=[num2str(sum(sum(B,1),2))];
    X=[];
    if strcmp(train_or_test,'train')==1
        n=278;
    else
        n=138;
    end
    if exist(strcat('X',spec,id,'_',train_or_test,'_'...
            ,num2str(sb),num2str(sum(interval)),'.csv'))==0 
        strcat('calculate_',spec,'_features_for_the_',train_or_test,'_set')
        for i=1:n  
            j=num2str(i);     
            Data=load_nii(strcat(train_or_test,'_',j,'.nii'));
            Xi=Data.img;
            Xi=Xi(B(1,1):B(1,2),B(2,1):B(2,2),B(3,1):B(3,2));
            if strcmp(spec,'mean')==1  
                BTV=BoxToVoxel(Xi,1);
                X=[X;mean((1:length(BTV)).*BTV)];
             elseif strcmp(spec,'smean')==1
                BTV=BoxToVoxel(Xi,1);
                bin=floor(length(BTV)/sb);
                l=1:bin;
                x=[];
                for j=1:sb
                    x=[x mean(l.*BTV(bin*(j-1)+1:bin*j))];
                end
                X=[X;x];
            elseif strcmp(spec,'voxvar')==1  
                X=[X;var(BoxToVoxel(Xi,1))];
            elseif strcmp(spec,'var')==1
                BTV=BoxToVoxel(Xi,1);
                x=1:length(BTV);
                X=[X;var((x-mean(x.*BTV)).^2.*BTV)];
            elseif strcmp(spec,'vox')==1
                X=[X;BoxToVoxel(Xi,sb)];
            elseif strcmp(spec,'svar')==1
                BTV=BoxToVoxel(Xi,1);
                bin=floor(length(BTV)/sb);
                l=1:bin;
                x=[];
                for j=1:sb
                   BTVs=BTV(bin*(j-1)+1:bin*j);
                   x=[x var((l-mean(l.*BTVs)).^2.*BTVs)];
                end
                X=[X;x];
                elseif strcmp(spec,'svoxvar')==1
                    BTV=BoxToVoxel(Xi,1);
                    bin=floor(length(BTV)/sb);
                    x=[];
                    for j=1:sb
                        x=[x var(BTV(bin*(j-1)+1:bin*j))];
                    end
                    X=[X;x];
                elseif strcmp(spec,'ivoxvar')==1
                    BTV=BoxToVoxel(Xi,1);
                    X=[X;var(BTV(interval(1):interval(2)))];
                elseif strcmp(spec,'ivox')==1
                    BTV=BoxToVoxel(Xi,1);
                    X=[X;mean(BTV(interval(1):interval(2)))];
            end
            
        end
        csvwrite(strcat('X',spec,id,'_',train_or_test,'_',...
            num2str(sb),num2str(sum(interval)),'.csv'),X);
    else
        X=csvread(strcat('X',spec,id,'_',train_or_test,'_',...
            num2str(sb),num2str(sum(interval)),'.csv'));
    end
            

end

