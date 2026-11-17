function res = PCAFM(rt,gt,pmax,q, factor_names)
%% INPUT
% rt          is n by T matrix
% gt          is d by T factor proxies
% pmax        is the upperbound of the number of latent factors
% q           is # of lags used in Newy-West standard errors

%% OUTPUT
% GammaFM     is Fama and MacBeth estimator
% GammaFM_Crt is Bai and Zhou bias-corrected estimates
% Gammahat    is (d+1) by pmax matrix of risk premia estimates
% avarhatFM   is (d+1) by 1 vector of the avar of zero-beta and risk premia using FM
% avarhatBZ   is (d+1) by 1 vector of the avar of zero-beta and risk premia using BZ
% avarhat     is (d+1) by pmax matrix of the avar of zero-beta and risk premia estimates
% gthat       is d by T by pmax matrix of cleaned factor proxies
% alphahat    is n by pmax matrix of pricing error estimates
% R2G         is d by pmax matrix of time series Rs quare
% R2F         is 1 by pmax matrix of cross-sectional Rsquare
% avarhat_nozero   is d by pmax matrix of the avar of risk premia estimates
% avarhatFM_nozero is d by 1 vector of the avar of risk premia using FM
% gtMP             is d by 1 vector of the risk premia using MP
% avarMP           is d by 1 vector of avar of the risk premia using MP
%% INITIALIZATION

    T  =  size(rt,2);
    n  =  size(rt,1);
    d  =  size(gt,1);
    
    alphahat              =      nan(n,pmax); 
    R2F                   =      nan(1,pmax); %
    R2G                   =      nan(d,pmax); %
    Gammahat              =      nan(d+1,pmax); 
    Gammahat_nozero       =      nan(d+1,pmax); 
    avarhat               =      nan(d+1,pmax);
    avarhat_nozero        =      nan(d,pmax);
    gthat                 =      nan(d,T,pmax); 
    chi2test              =      nan(d,pmax); % 
    SR2hat                =      nan(1,pmax);
%% ESTIMATION

    rtbar                 =      rt - repmat(mean(rt,2),1,T);% 
    rbar                  =      mean(rt,2);
    gtbar                 =      gt - repmat(mean(gt,2),1,T);% 
    
    % Fama MacBeth Method
    
    betaFM    =  rtbar * gtbar'*inv(gtbar*gtbar');
    GammaFM   =  inv([ones(n,1),betaFM]'*[ones(n,1),betaFM]) * [ones(n,1),betaFM]'* mean(rt,2);
   
    % FM Standard Errors
        
    Gammas    = inv([ones(n,1),betaFM]'*[ones(n,1),betaFM]) * [ones(n,1),betaFM]'*rt;    
    avarhatFM = var(Gammas')'/(T);
    
    Gammas_nozero    = inv(betaFM'*betaFM) * betaFM'*rt;  
    GammaFM_nozero = mean(Gammas_nozero,2);
    avarhatFM_nozero = var(Gammas_nozero')'/(T);
    
    
    % Mimicking Portfolio approach

    if n<T

        erhat    =  gtbar * rtbar'*inv(rtbar*rtbar');
        zrbar    =  gtbar - erhat*rtbar;
        varr     =  T^(-1)*rtbar*rtbar';
        avarMP   =  zeros(d,1);
        for ll=1:d

            vareta   =  inv(varr)*(rtbar*diag(zrbar(ll,:)')*diag(zrbar(ll,:)')* rtbar')*inv(varr)/T;
            avarMP(ll)=  T^(-1)*(rbar'*vareta*rbar+erhat(ll,:)*varr*erhat(ll,:)');
        end
        
    else
        erhat=NaN(d,n);
        avarMP=NaN(d,1);
    end
       
    % Bai and Zhou Standard Errors: 
   
    uhat          =   rtbar -  betaFM*gtbar; 
    Sigmahat      =   diag(diag((uhat-repmat(mean(uhat,2),1,T))* (uhat-repmat(mean(uhat,2),1,T))'/T));    
    sigma2bar     =   mean(diag(Sigmahat));        
    Omegahat      =   1/n*[ones(n,1)'*ones(n,1),ones(n,1)'*betaFM;betaFM'*ones(n,1),betaFM'*betaFM] + 1/T*[0,zeros(1,d);zeros(d,1),sigma2bar*inv(1/T*gtbar*gtbar')];
    GammaFM_Crt   =   GammaFM  +  inv(Omegahat)*[0;1/T*sigma2bar*inv(gtbar*gtbar'/T)*GammaFM(2:end)];
    H             =   eye(d+1) - inv(Omegahat)*[0,zeros(1,d);zeros(d,1),1/T*sigma2bar*inv(gtbar*gtbar'/T)];
    Pi1hat        =   H*[0, zeros(1,d);zeros(d,1),(gtbar*gtbar'/T)]*H';     
    Pi2hat        =   inv(Omegahat)*[(1+GammaFM(2:end)'*inv((gtbar*gtbar'/T))*GammaFM(2:end))*ones(1,n)*Sigmahat*ones(n,1)/n,...
                      (1+GammaFM(2:end)'*inv((gtbar*gtbar'/T))*GammaFM(2:end))*ones(1,n)*Sigmahat*betaFM/n; ...
                      (1+GammaFM(2:end)'*inv((gtbar*gtbar'/T))*GammaFM(2:end))*betaFM'*Sigmahat*ones(n,1)/n,...
                      (1+GammaFM(2:end)'*inv((gtbar*gtbar'/T))*GammaFM(2:end))*betaFM'*Sigmahat*betaFM/n]*inv(Omegahat);     
    avarhatBZ     =   diag(1/T*Pi1hat+1/n/T*Pi2hat);

%%

     Pihat        =        zeros(d,d);  % avarhat for avg returns
        
        
     for t = 1:T
                     
            Pihat             =        Pihat    +  gtbar(:,t)     * gtbar(:,t)'/T;
            
            for s = 1:min(t-1,q) 
                
                   Pihat         =        Pihat    +  1/T*(1-s/(q+1))* (gtbar(:,t)     * gtbar(:,t-s)' + gtbar(:,t-s) * gtbar(:,t)' );
                        
            end        
     end
    
     avarghat                 = diag(Pihat/T);
     
     
    % Giglio Xiu Method
    
    % step 1: SVD
    
    [U,S,V] = svd(rtbar/sqrt(n)/sqrt(T),'econ');    
    tempeig = diag(S^2);
    
    for phat  = 1 : pmax   % at least one factor up to pmax
        
            vhat              =       T^0.5  * V(:,1:phat)';
            betahat           =       T^(-1) * rtbar * vhat';
         
        
        Sigmavhat             =       vhat*vhat'/T;
       
    % step 2: FM with PCs
    
        Gammatilde            =       inv([ones(n,1),betahat]'*[ones(n,1),betahat]) * [ones(n,1),betahat]'* mean(rt,2);
      % without estimating zero-beta-rate.
        Gammatilde_nozero     =      [0; inv(betahat'*betahat) * betahat'* mean(rt,2)];

    % step 3: TS Regression
    
        etahat                =       gtbar   *  vhat' * inv(vhat*vhat');   
        what                  =       gtbar   -  etahat  *  vhat;
        
    % step 4: Combine TS and CS estimates
        
        Gammahat(:,phat)      =       [Gammatilde(1);etahat*Gammatilde(2:end)];   
        Gammahat_nozero(:,phat)=      [Gammatilde_nozero(1);etahat*Gammatilde_nozero(2:end)];   
    
    % Calculating the Squared Sharpe Ratio 
    
        vres(:,:,phat)        =       1/T*rtbar*rtbar'  -  betahat*Sigmavhat*betahat';
        vres_nofactor         =       1/T*rtbar*rtbar';
        Sigmauhat             =       diag(1/T*rtbar*rtbar'  -  betahat*Sigmavhat*betahat');
        SR2hat(1,phat)        =       Gammatilde(2:end)'*betahat'*inv(betahat*Sigmavhat*betahat' + diag(Sigmauhat))*betahat*Gammatilde(2:end);   
        
    % step 5: Goodness of Fit
    
        Miota                 =        eye(n) - ones(n,1)*ones(1,n)/n;              
        R2F(1,phat)           =        rbar'*Miota*betahat*inv(betahat'*Miota*betahat)*betahat'*Miota*rbar/(rbar'*Miota*rbar);
        R2G(:,phat)           =        1./diag(gtbar*gtbar').*diag(etahat*(vhat*vhat')*etahat');
        
    % step 6: Newy-West Estimation of Avar
        
        Pi11hat               =        zeros(d*phat,d*phat);
        Pi12hat               =        zeros(d*phat,phat);
        Pi22hat               =        zeros(phat,phat);
        
             
        for t = 1:T
                    
            Pi11hat           =        Pi11hat  +  vec(what(:,t) * vhat(:,t)')*vec(what(:,t) * vhat(:,t)')'/T;
            Pi12hat           =        Pi12hat  +  vec(what(:,t) * vhat(:,t)')*vhat(:,t)'/T;
            Pi22hat           =        Pi22hat  +  vhat(:,t)     * vhat(:,t)'/T;
            
               
            for s = 1:min(t-1,q) 
                
                Pi11hat       =        Pi11hat  + 1/T*(1-s/(q+1))* (vec(what(:,t) * vhat(:,t)')*vec(what(:,t-s) * vhat(:,t-s)')'+vec(what(:,t-s) * vhat(:,t-s)')*vec(what(:,t) * vhat(:,t)')');
                Pi12hat       =        Pi12hat  + 1/T*(1-s/(q+1))* (vec(what(:,t) * vhat(:,t)')*vhat(:,t-s)' + vec(what(:,t-s) * vhat(:,t-s)')*vhat(:,t)');
                Pi22hat       =        Pi22hat  + 1/T*(1-s/(q+1))* (vhat(:,t)     * vhat(:,t-s)' + vhat(:,t-s) * vhat(:,t)' );
            
                                     
            end        
        end
 
        alphahat(:,phat)   =  rbar  - [ones(n,1),betahat]*Gammatilde;
       

        
         
        avarhat(:,phat)   = [(1 - mean(betahat,1)*inv(betahat'*betahat/n)*mean(betahat,1)')^(-1) *var(alphahat(:,phat))/n;...
            diag(kron(Gammatilde(2:end)'*inv(Sigmavhat),eye(d))*Pi11hat*kron(inv(Sigmavhat)*Gammatilde(2:end),eye(d))/T + ...
             kron(Gammatilde(2:end)'*inv(Sigmavhat),eye(d))*Pi12hat*etahat'/T + (kron(Gammatilde(2:end)'*inv(Sigmavhat),eye(d))*Pi12hat*etahat')'/T + ...
             etahat*Pi22hat*etahat'/T) + ...
             diag(var(alphahat(:,phat))*etahat*inv(betahat'*betahat/n - mean(betahat,1)'*mean(betahat,1))*etahat')/n];


        avarhat_nozero(:,phat)   = [diag(kron(Gammatilde(2:end)'*inv(Sigmavhat),eye(d))*Pi11hat*kron(inv(Sigmavhat)*Gammatilde(2:end),eye(d))/T + ...
             kron(Gammatilde(2:end)'*inv(Sigmavhat),eye(d))*Pi12hat*etahat'/T + (kron(Gammatilde(2:end)'*inv(Sigmavhat),eye(d))*Pi12hat*etahat')'/T + ...
             etahat*Pi22hat*etahat'/T)];

         
       
        
     % step 7: recovery of gt
         
        gthat(:,:,phat) = etahat*vhat;

     % step 8: testing
        for ll=1:d
            chi2test(ll,phat)    = 1-chi2cdf((T*etahat(ll,:)/((vhat*vhat'/T)\Pi11hat((0:(phat-1))*d+ll,(0:(phat-1))*d+ll)/(vhat*vhat'/T))*etahat(ll,:)'),phat);
        end
    end
    
     % step 9: choose number of latent factors
    
    phat        = zeros(3,1);
    ppmax       = 10;
    obj         = tempeig(1:ppmax) + 0.25*(1:ppmax)'* (log(n)+log(T))*(n^(-0.5)+T^(-0.5))*median(tempeig(1:ppmax));
    phat(1)     = find(obj==min(obj))-1; 
    obj2        = tempeig(1:ppmax) + 0.5*(1:ppmax)'* (log(n)+log(T))*(n^(-0.5)+T^(-0.5))*median(tempeig(1:ppmax));
    phat(2)     = find(obj2==min(obj2))-1; 
    obj3        = tempeig(1:ppmax) + 0.75*(1:ppmax)'* (log(n)+log(T))*(n^(-0.5)+T^(-0.5))*median(tempeig(1:ppmax));
    phat(3)     = find(obj3==min(obj3))-1; 
    
    
   % ridge 
   
   lam  = S(pmax*2,pmax*2)^2*n*T;
   beta_ridge          =    (gtbar*rtbar'*inv(rtbar*rtbar'+lam*eye(n)))';  %U*((S*S'*n+lam*eye(n))\(S*(sqrt(n*T))*V'*gtbar'))/T;      
   Gammahat_nozero_ridge     =      beta_ridge'*mean(rt,2);
 
%% OUTPUT
    %res.GammaFM        = GammaFM;
    %res.GammaFM_nozero = GammaFM_nozero;
    %res.GammaFM_Crt    = GammaFM_Crt;
    %res.avarhatFM      = avarhatFM;    
    %res.avarhatFM_nozero=avarhatFM_nozero;    
    %res.avarhatBZ      = avarhatBZ;
    %res.avarhat_nozero = avarhat_nozero;
    %res.avarhat        = avarhat;
    %res.alphahat       = alphahat;    
    %res.gthat          = gthat;
    %res.Gammahat       = Gammahat;
    %res.Gammahat_nozero= Gammahat_nozero;
    %res.Gammahat_nozero_ridge= Gammahat_nozero_ridge;
    %res.R2F            = R2F;
    %res.R2G            = R2G;
    %res.phat           = phat;
    %res.chi2test       = chi2test;
    %res.SR2hat         = SR2hat;
    %res.gtMP           =  mean(erhat*rt,2);
    %res.avarMP         = avarMP; 
    %res.vt = vhat;     % Corresponds to the largest phat
    %res.avarghat       = avarghat;
    %res.vres           = vres;
    %res.vres_nofactor           = vres_nofactor;

% Assign relevant row names to the outputs
    % Row labels:
    % At the beginning of the PCAFM function:
    if ~iscell(factor_names)
        factor_names = cellstr(factor_names);
    end
    gamma_names = vertcat({'Zero-beta'}, factor_names(:));
    pmax_cols = cellstr(num2str((1:pmax)'));
    %gamma_names = ['Zero-beta'; factor_names];
    
    % Outputs kept as tables
    %% 1. Risk Premium Estimates
    % Core Fama-MacBeth estimates
    res.GammaFM = array2table(GammaFM, 'RowNames', gamma_names);
    res.GammaFM_nozero = array2table(GammaFM_nozero, 'RowNames', factor_names);
    res.GammaFM_Crt = array2table(GammaFM_Crt, 'RowNames', gamma_names);
    
    % PCA-based estimates
    res.Gammahat = array2table(Gammahat, 'RowNames', gamma_names, 'VariableNames', pmax_cols);
    res.Gammahat_nozero = array2table(Gammahat_nozero, 'RowNames', gamma_names, 'VariableNames', pmax_cols);
    res.Gammahat_nozero_ridge = array2table(Gammahat_nozero_ridge, 'RowNames', factor_names);
    
    % Mimicking portfolio estimates
    res.gtMP = array2table(mean(erhat*rt,2), 'RowNames', factor_names);
    
    %% 2. Statistical Inference
    % Standard errors for Fama-MacBeth
    res.avarhatFM = array2table(avarhatFM, 'RowNames', gamma_names);
    res.avarhatFM_nozero = array2table(avarhatFM_nozero, 'RowNames', factor_names);
    res.avarhatBZ = array2table(avarhatBZ, 'RowNames', gamma_names);
    
    % Standard errors for PCA estimates
    res.avarhat = array2table(avarhat, 'RowNames', gamma_names, 'VariableNames', pmax_cols);
    res.avarhat_nozero = array2table(avarhat_nozero, 'RowNames', factor_names, 'VariableNames', pmax_cols);
    
    % Standard errors for mimicking portfolios and factors
    res.avarMP = array2table(avarMP, 'RowNames', factor_names);
    res.avarghat = array2table(avarghat, 'RowNames', factor_names);
    
    % Statistical tests
    res.chi2test = array2table(chi2test, 'RowNames', factor_names, 'VariableNames', pmax_cols);
    
    %% 3. Model Diagnostics
    % R-squared measures
    res.R2F = R2F;  % Cross-sectional R-squared
    res.R2G = array2table(R2G, 'RowNames', factor_names, 'VariableNames', pmax_cols);  % Time-series R-squared
    
    % Model selection
    res.phat = phat;  % Selected number of factors
    res.SR2hat = SR2hat;  % Squared Sharpe ratios
    
    %% 4. Factor and Residual Analysis
    % Estimated factors and residuals
    res.gthat = gthat;  % Cleaned factor proxies
    res.vt = vhat;  % Principal components
    res.alphahat = alphahat;  % Pricing errors
    res.vres = vres;  % Residual variance with factors
    res.vres_nofactor = vres_nofactor;  % Residual variance without factors