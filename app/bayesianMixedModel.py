"""
Bayesian Mixed model
"""

import pymc
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, f1_score, precision_score, balanced_accuracy_score
from scipy.stats import multivariate_normal

class MAP_LR_model:
    """
    A Bayesian mixed model 
    
    
    
    Parameters:
    -----------
    beta_prior:  
    alpha_prior: 
    sigma_prior: 
    eta_prior: 
    eta_beta_prior: 
    n_params: 
    idx_random: , default=[]
    random_seed: , default=None
    
    Returns:
    --------
    """

    def __init__(self, beta_prior, alpha_prior, sigma_prior, eta_prior, eta_beta_prior, n_params, idx_random=[], random_seed=None):
        self.beta_prior=beta_prior
        self.alpha_prior=alpha_prior
        self.sigma_prior=sigma_prior
        self.eta_prior=eta_prior
        self.eta_beta_prior=eta_beta_prior
        self.random_seed=random_seed
        self.n_params = n_params
        self.idx_random = idx_random
    
    def log_reg_train(self, X, idx_grp):
        """
        Prediction on train test
        Parameters:
        -----------
        X:
        idx_grp:
        Returns:
        --------
        
        """
        theta = self.alpha_prior + sum([X[:,j]*self.beta_prior[j] + X[:,j]*self.eta_beta_prior[j][idx_grp] for j in range(self.n_params)]) +self.eta_prior[idx_grp]
        return 1/(1+np.exp(-theta))

    def log_reg_test(self, X):
        """
        Prediction on test set. DIFFERENCE WITH TRAIN --> 
        Parameters:
        -----------
        X:
        Returns:
        --------
        """
        theta = self.intercept + sum([np.array(X)[:,i]*self.coefs[i] + np.array(X)[:,i]*self.random_effect_beta[i] for i in range(self.n_params)]) + self.random_effect
        return 1/(1+np.exp(-theta))

    def fit(self, X, y, train_group, y_obs="y_obs"):
        """
        Fit the model
        Parameters:
        -----------
        Returns:
        --------
        """
        self.grp_unique = train_group.unique()
        self.n_grp = len(self.grp_unique)
        idx_grp = [int(np.where(self.grp_unique == grp)[0]) for grp in train_group]
        
        y_obs = pymc.Normal(y_obs, mu=self.log_reg_train(X, idx_grp), sigma=self.sigma_prior, observed=y)
        self.samplePosterior = pymc.sample(draws=10000, tune=1000, random_seed=self.random_seed, idata_kwargs={"log_likelihood": True})
        sample_alpha = self.samplePosterior.posterior["alpha"].values
        sample_beta = self.samplePosterior.posterior["beta"].values
        sample_sigma = self.samplePosterior.posterior["sigma"].values
        self.intercept = sample_alpha[0].mean()
        self.coefs = sample_beta[0].mean(axis=0)
        self.sigma = sample_sigma[0].mean()

        self.sample_eta = None
        sample_eta_beta = []
        for i in range(self.n_params+1):
            if i in set(self.idx_random):
                if i==0:
                    self.sample_eta = self.samplePosterior.posterior["eta"].values
                else:
                    sample_eta_beta.append(self.samplePosterior.posterior["eta_beta"+str(i)].values)
            else:
                if i !=0 :
                    sample_eta_beta.append(0)

        self.sample_eta_beta = sample_eta_beta
        
    def cv_eval(self, X, y, group, score=roc_auc_score, n_split=3, random_state=42):
        """
        Cross validation to evaluate model efficiency
        Parameters:
        -----------
        X: 
        y: 
        group:
        score:
        n_split: , default=3
        random_state: , default=None
        Returns:
        --------
        """
        n = X.shape[0]
        fold_size = np.array([int(n / n_split)]*n_split) + np.array([1]*(n%n_split) + [0]*(n_split-n%n_split))
        idx = np.arange(X.shape[0])
        np.random.seed(random_state)
        np.random.shuffle(idx)
        cv_scores = []
        for i, fold in enumerate(fold_size):
            idx_val = idx[sum(fold_size[:i]):sum(fold_size[:i])+fold]
            idx_cal = np.array(list(set(idx) - set(idx_val)))
            val_grp = group.iloc[idx_val]
            X_val = X[idx_val]
            y_val = np.array(y)[idx_val]
            cal_grp = group.iloc[idx_cal]
            X_cal = X[idx_cal]
            y_cal = np.array(y)[idx_cal]
            self.fit(X=X_cal, y=y_cal, train_group=cal_grp, y_obs="y_obs"+str(i))
            pred = self.predict(X=X_val, test_group=val_grp)
            cv_scores.append(roc_auc_score(y_val, pred))
        self.cv_scores = cv_scores


    def predict(self, X, test_group):
        """
        Predict using the fitted model 
        
        Parameters:
        -----------
        X: the data to predict
        test_group: a list of size X.shape[0]. The group to which each observation is associated.  
        Returns:
        --------
        pred: the prediction of the bayesian mixed model 
        """

        idx_view_in_train = [test in set(self.grp_unique) for test in test_group]
        idx_grp_test = [int(np.where(self.grp_unique == grp)[0]) for grp in test_group[idx_view_in_train]]
        n_grp = len(test_group)

        random_effect = np.zeros(n_grp)

        random_effect_beta = []
        for i in range(self.n_params+1):
            if i in set(self.idx_random):
                if i==0:
                    random_effect[idx_view_in_train] = self.sample_eta[0].mean(axis=0)[idx_grp_test]
                else:
                    random_effect_tmp = np.zeros(n_grp)
                    random_effect_tmp[idx_view_in_train] = self.sample_eta_beta[i-1][0].mean(axis=0)[idx_grp_test]
                    random_effect_beta.append(random_effect_tmp)
            elif i!=0:
                random_effect_beta.append(np.zeros(n_grp))
        self.random_effect = random_effect
        self.random_effect_beta = random_effect_beta

        pred = self.log_reg_test(X)
        return pred
    
    def log_likelihood(self, X, y, group):
        """
        Compute the log likelihood
        Parameters:
        -----------
        Returns:
        --------
        """
        pred = self.predict(X, group)
        return multivariate_normal.logpdf(y, pred, np.diag([self.sigma]*len(y)))

    def AIC(self, X, y, group):
        """
        Compute the AIC
        Parameters:
        -----------
        Returns:
        --------
        """
        k = self.n_params + len(self.idx_random) + 1
        return 2*k - 2*self.log_likelihood(X,y,group)