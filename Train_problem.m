function [labels,U,Z,M,converge_Z,converge_Z_G] = Train_problem(N_m,G,X, cls_num, anchor,alpha,gamma,delta)

nV = length(X);
N = size(X{1},2);
t=anchor;%锚点数目
nC=cls_num;
%% ============================ Initialization ============================
for k=1:nV
    X{k}=X{k}';
    M{k}= zeros(N_m,size(X{k},2));
    Z{k} = zeros(N,t); 
    W{k} = zeros(N,t);
    J{k} = zeros(N,t);
    A{k} = zeros(t,size(X{k},2));
    B{k} = zeros(t,t);
    Lb{k} = zeros(t,t);
    E{k} = zeros(N,size(X{k},2)); %E{2} = zeros(size(X{k},1),N);
    Y{k} = zeros(N,size(X{k},2)); %Y{2} = zeros(size(X{k},1),N);
end

w = zeros(N*t*nV,1);
j = zeros(N*t*nV,1);
sX = [N, t, nV];

Isconverg = 0;epson = 1e-7;
iter = 0;
mu = 10e-5; max_mu = 10e10; pho_mu = 2;
rho = 0.0001; max_rho = 10e12; pho_rho = 2;

converge_Z=[];
converge_Z_G=[];


%% ================================ Upadate ===============================
while(Isconverg == 0)
%% ============================== Upadate M^k =============================
    for k = 1:nV
    M{k} = inv(G{k}'*G{k})*G{k}'*(Z{k}*A{k}+E{k}-(1/mu)*Y{k}-X{k});
     end
    
%% ============================== Upadate Z^k =============================
     clear i l
       temp_E =[];
      for k =1:nV;
          tmp = Y{k}*A{k}' + mu*(X{k}+G{k}*M{k})*A{k}' +  rho*J{k}- mu*E{k}*A{k}' - W{k};
          Z{k}=tmp*inv(2*gamma*Lb{k}+rho*eye(t,t)+ mu*eye(t,t));
          temp_E=[temp_E,X{k}+G{k}*M{k}-Z{k}*A{k}+Y{k}/mu];
      end
      clear k 

%% =========================== Upadate E^k, Y^k ===========================
        temp_E=temp_E';
       [Econcat] = solve_l1l2(temp_E,alpha/mu);
       ro_b =0;
       E{1} =  Econcat(1:size(X{1},2),:)';
       Y{1} = Y{1} + mu*(X{1}+G{1}*M{1}-Z{1}*A{1}-E{1});
       ro_end = size(X{1},2);
       for i=2:nV
           ro_b = ro_b + size(X{i-1},2);
           ro_end = ro_end + size(X{i},2);
           E{i} =  Econcat(ro_b+1:ro_end,:)';
           Y{i} = Y{i} + mu*(X{i}+G{i}*M{i}-Z{i}*A{i}-E{i});
       end

%% ============================= Upadate J^k ==============================

                Z_tensor = cat(3, Z{:,:});
                W_tensor = cat(3, W{:,:});
                z = Z_tensor(:);
                w = W_tensor(:);
                J_tensor = solve_G(Z_tensor + 1/rho*W_tensor,rho,sX,delta);
                j = J_tensor(:);
                %nuclear
%                  [j,objV] = wshrinkObj(Z_tensor + 1/rho*W_tensor,1/rho,sX,0,3);
%                  J_tensor=reshape(j, sX);
%% ============================== Upadate W ===============================
        w = w + rho*(z - j);
        W_tensor = reshape(w, sX);
    for k=1:nV
        W{k} = W_tensor(:,:,k);
    end
     %%AAAA
   GA={};
for i = 1 :nV
    GA{i}=Z{i}'*(Y{i}+mu*(X{i}+G{i}*M{i})-mu*E{i});
    [Au,ss,Av] = svd(GA{i},'econ');
    A{i}=Au*Av';
end
for i = 1 :nV
    B{i} = constructW_PKN(A{i}', nC);
    Db = diag(sum(B{i},1)+eps);
    Lb{i} = eye(t,t)-Db^-0.5*B{i}*Db^-0.5;
end
%% ====================== Checking Coverge Condition ======================
    max_Z=0;
    max_Z_G=0;
    Isconverg = 1;
    for k=1:nV
        if (norm(X{k}+G{k}*M{k}-Z{k}*A{k}-E{k},inf)>epson)
            history.norm_Z = norm(X{k}+G{k}*M{k}-Z{k}*A{k}-E{k},inf);
            Isconverg = 0;
            max_Z=max(max_Z,history.norm_Z );
        end
        
        J{k} = J_tensor(:,:,k);
        W_tensor = reshape(w, sX);
        W{k} = W_tensor(:,:,k);
        if (norm(Z{k}-J{k},inf)>epson)
            history.norm_Z_G = norm(Z{k}-J{k},inf);
            Isconverg = 0;
            max_Z_G=max(max_Z_G, history.norm_Z_G);
        end
    end
    converge_Z=[converge_Z max_Z];
    converge_Z_G=[converge_Z_G max_Z_G];
   
    
    if (iter>50)
        Isconverg  = 1;
    end
    iter = iter + 1;
    mu = min(mu*pho_mu, max_mu);
    rho = min(rho*pho_rho, max_rho);
end

Sbar=[];
for i = 1:nV
    Sbar=cat(1,Sbar,1/sqrt(nV)*Z{i}');
end
[U,Sig,V] = mySVD(Sbar',nC); 

rand('twister',5489)
labels=litekmeans(U, nC, 'MaxIter', 100,'Replicates',10);
end
