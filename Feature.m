function X = Feature(train_or_test,N,spec,y)
     Nstring=[num2str(N(1)) num2str(N(2)) num2str(N(3))];
     if N(1)<0
         dl1=2^(N(1)+4);
     else
        dl1=11*2^N(1);%factors:2^4*11
     end
     if N(2)<0
         dl2=2^(N(2)+4);
     else
        dl2=13*2^N(2);%factors:2^4*13
     end
     if N(3)<0
         dl3=2^(N(3)+4);
     else
        dl3=11*2^N(3);%factors:2^4*11
     end
     X=[];
     if strcmp(train_or_test,'train')==1
         n=278;
     else
         n=138;
     end
     if exist(strcat('X',spec,Nstring,'_',train_or_test,'.csv'))==0
         'calculate raw features'
         for i=1:n  
            j=num2str(i);     
            Data=load_nii(strcat(train_or_test,'_',j,'.nii'));
            Xi=Data.img;
            x=[];
            if strcmp(spec,'mean')==1  
                for k3=1:dl1-1:176-dl1+1
                    for k2=1:dl2-1:208-dl2+1
                      for k1=1:dl3-1:176-dl3+1
                          x=[x mean(mean(mean(Xi(k1:k1+dl1-1,k2:k2+dl2-1,k3:k3+dl3-1),1),2),3)];
                      end
                    end
                end
                elseif strcmp(spec,'var')==1  
                    for k3=1:dl1-1:176-dl1+1
                        for k2=1:dl2-1:208-dl2+1
                          for k1=1:dl3-1:176-dl3+1
                              x=[x mean(mean(var(double(Xi(k1:k1+dl1-1,k2:k2+dl2-1,k3:k3+dl3-1))),2),3)];
                          end
                        end
                    end
            elseif strcmp(spec,'voxel')+strcmp(spec,'MI')+strcmp(spec,'voxvar')+strcmp(spec,'entropy')==1
                if exist(strcat('X','voxel',Nstring,'_',train_or_test,'.csv'))==0
                    for k3=1:dl3-1:176-dl3+1
                        for k2=1:dl2-1:208-dl2+1
                          for k1=1:dl1-1:176-dl1+1
                              x=[x BoxToVoxel(Xi(k1:k1+dl1-1,k2:k2+dl2-1,k3:k3+dl3-1),10)];
                          end
                        end
                    end
                end
                    
            else
                error('choose a valid feature specification');
            end
            X=[X;x];  
         end
         if strcmp(spec,'MI')+strcmp(spec,'voxvar')+strcmp(spec,'entropy')==1
            if exist(strcat('X','voxel',Nstring,'_',train_or_test,'.csv'))==1
                X=csvread(strcat('X','voxel',Nstring,'_',train_or_test,'.csv'));
            else
                csvwrite(strcat('Xvoxel',Nstring,'_',train_or_test,'.csv'),X);
            end
            
            if strcmp(spec,'voxvar')==1
                Vartot=[];
                t=0;
                l=length(BoxToVoxel(zeros(1,1),10));
                     for k3=1:dl1-1:176-dl1+1
                        for k2=1:dl2-1:208-dl2+1
                          for k1=1:dl3-1:176-dl3+1
                              Y=X(:,t*l+1:(t+1)*l);
                              Va=[];
                                  for i=1:size(Y,1)
                                     Va=[Va;var(Y(i,:))];
                                  end
                              Vartot=[Vartot Va];
                              t=t+1;
                          end
                      end
                 end
                X=Vartot;
                csvwrite(strcat('X',spec,Nstring,'_',train_or_test,'.csv'),X);
            elseif strcmp(spec,'entropy')==1
                Entot=[];
                t=0;
                l=length(BoxToVoxel(zeros(1,1),10));
                     for k3=1:dl1-1:176-dl1+1
                        for k2=1:dl2-1:208-dl2+1
                          for k1=1:dl3-1:176-dl3+1
                              Y=X(:,t*l+1:(t+1)*l);
                              En=[];
                                  for i=1:size(Y,1)
                                     En=[En;Entropy(Y(i,:))];
                                  end
                              Entot=[Entot En];
                              t=t+1;
                          end
                      end
                 end
                X=Entot;
                csvwrite(strcat('X',spec,Nstring,'_',train_or_test,'.csv'),X);
            else
            
             if strcmp(train_or_test,'train')==1
                 MItot=[];
                 M1=[];
                 t=0;
                 l=length(BoxToVoxel(zeros(1,1),10));
                 
                 for k3=1:dl1-1:176-dl1+1
                     for k2=1:dl2-1:208-dl2+1
                          for k1=1:dl3-1:176-dl3+1
                              Y=X(:,t*l+1:(t+1)*l);
                              m1=round(mean(Y(y==1,:),1));
                              MI=[];
                                  for i=1:size(Y,1)
                                     MI=[MI;mutInfo(Y(i,:),m1)];
                                  end
                              M1=[M1;m1];
                              MItot=[MItot MI];
                              t=t+1;
                          end
                      end
                 end
                X=MItot;
                csvwrite(strcat('X',spec,Nstring,'_',train_or_test,'.csv'),X);
                csvwrite('M1.csv',m1);
                 
            else 
            MItot=[];
            M1=csvread('M1.csv');
            l=size(M1,2);
            t=0;
            for k3=1:dl1-1:176-dl1+1
                for k2=1:dl2-1:208-dl2+1
                    for k1=1:dl3-1:176-dl3+1
                        Y=X(:,t*l+1:(t+1)*l);
                        MI=[];
                        for i=1:size(Y,1)
                            MI=[MI;mutInfo(Y(i,:),M1(t+1,:))];
                        end
                        M1=[M1;m1];
                        MItot=[MItot MI];
                        t=t+1;
                     end
                 end
            end
            X=MItot;
            csvwrite(strcat('X',spec,Nstring,'_',train_or_test,'.csv'),X);
             end
            end
         else
                csvwrite(strcat('X',spec,Nstring,'_',train_or_test,'.csv'),X);
         end
     
     else
         X=csvread(strcat('X',spec,Nstring,'_',train_or_test,'.csv'));
     end
     

end

