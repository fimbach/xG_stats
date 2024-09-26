import streamlit as st
import pandas as pd
import numpy as np
from tqdm import tqdm
import pymc
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, balanced_accuracy_score, f1_score, roc_curve
import arviz as az

from bayesianMixedModel import MAP_LR_model


@st.cache_resource
# BEST RANDOM EFFECTS
def best_re_selection(Xy, comp_to_model, all_comp, train_group, test_group, X_trainNorm, X_testNorm, X_train, y_train, X_test, y_test, params_used):
    
    best_random_effects = {}
    AUC_comp = pd.DataFrame(columns = ["comp", "AUC_bayes"]) #, "AUC_statsbomb"])
    models_tmp = {}
    shots_from_comp = Xy[np.sum(all_comp == comp_to_model, axis = 1) == 2]

    X_train_NotComp = X_trainNorm.loc[list(set(X_train.index) - set(shots_from_comp.index))]
    X_train_comp = X_trainNorm.loc[list(set(X_train.index) & set(shots_from_comp.index))]
    y_train_comp = y_train.loc[list(set(X_train.index) & set(shots_from_comp.index))]
    X_test_comp = X_testNorm.loc[list(set(X_test.index) & set(shots_from_comp.index))]
    y_test_comp = y_test.loc[list(set(X_test.index) & set(shots_from_comp.index))]

    # Build prior using other comp
    X = np.array(X_train_NotComp)
    
    BayesModel = pymc.Model()

    with BayesModel:
        alpha = pymc.Uniform("alpha", lower = -10, upper = 10)
        beta = pymc.Uniform("beta", lower = -10, upper = 10, shape = len(params_used))
        sigma = pymc.Gamma("sigma", mu = 1, sigma = 1/2)
        traceFixedEffectOnly = pymc.sample(draws = 10000, tune = 1000, random_seed = 42)

    sample_alpha = traceFixedEffectOnly.posterior["alpha"].values
    sample_beta = traceFixedEffectOnly.posterior["beta"].values
    cov_beta = np.cov(sample_beta[0].T)
    std_alpha = np.std(sample_alpha[0])
    g = np.mean(np.diag(cov_beta))
    cov_beta = cov_beta / g
    std_alpha = std_alpha / np.sqrt(g)
    mean_beta = sample_beta[0].mean(axis = 0) 
    mean_alpha = sample_alpha[0].mean()

    # Build model on studied comp
    X = np.array(X_train_comp)
    grp_unique = train_group[X_train_comp.index].unique()
    n_grp = len(grp_unique)
    
    # Sans effet aléatoire
    idx_random = []

    BayesModel = pymc.Model()
    with BayesModel:

        alpha = pymc.Normal("alpha", mu = mean_alpha, sigma = std_alpha)
        beta = pymc.MvNormal("beta", mu = mean_beta, cov = cov_beta)
        sigma = pymc.Gamma("sigma", mu = 1, sigma = 1/2)
        
        # Pas d'effet aléatoire sur l'intercept
        omega = 0
        eta = np.zeros(n_grp)
        omega_beta = []
        eta_beta = []
        for i in range(len(params_used)): 
            omega_beta.append(0)
            eta_beta.append(np.zeros(n_grp))
        
        model = MAP_LR_model(beta_prior = beta,
                            alpha_prior = alpha,
                            sigma_prior = sigma,
                            eta_prior = eta, 
                            eta_beta_prior = eta_beta,
                            n_params = len(params_used), 
                            idx_random = idx_random,
                            random_seed = 42)

        model.fit(X, y_train_comp, train_group = train_group[X_train_comp.index])
        pred = model.predict(X_test_comp, test_group = test_group[X_test_comp.index])
        models_tmp["None"] = model
    
    AUC = pd.Series()
    AUC[0] = roc_auc_score(y_test_comp, pred)
    to_compare = {0: model.samplePosterior}

    # Which params ?
    n_re = 1
    best_loo = -np.infty
    best_reach = False
    while (~best_reach) & (n_re <= 7):
        if n_re != 1:
            to_compare = {}
            AUC = pd.Series()
        for j in set(range(len(params_used) + 1)) - set(idx_random):
            idx_random_tmp = idx_random + [j]

            BayesModel = pymc.Model()
            with BayesModel:

                alpha = pymc.Normal("alpha", mu = mean_alpha, sigma = std_alpha)
                beta = pymc.MvNormal("beta", mu = mean_beta, cov = cov_beta)
                sigma = pymc.Gamma("sigma", mu = 1, sigma = 1/2)

                # Initialisation si pas d'effet aléatoire sur l'intercept
                omega = 0
                eta = np.zeros(n_grp)
                omega_beta = []
                eta_beta = []
                for i in range(len(params_used) + 1):
                    if i in(idx_random_tmp):
                        if i == 0: 
                            omega = pymc.Gamma("omega", mu = 0.3, sigma = 1/10)
                            eta = pymc.Normal("eta", mu = np.zeros(n_grp), sigma = np.repeat(omega,n_grp))
                        else:
                            omega_beta.append(pymc.Gamma("omega_beta" + str(i), mu = 0.3, sigma = 1/10))
                            eta_beta.append(pymc.Normal("eta_beta" + str(i), mu = np.zeros(n_grp), sigma = np.repeat(omega_beta[i-1], n_grp)))
                    else:
                        if i != 0:
                            omega_beta.append(0)
                            eta_beta.append(np.zeros(n_grp))

                model = MAP_LR_model(beta_prior = beta,
                                    alpha_prior = alpha,
                                    sigma_prior = sigma, 
                                    eta_prior = eta,
                                    eta_beta_prior = eta_beta,
                                    n_params = len(params_used), 
                                    idx_random = idx_random_tmp,
                                    random_seed = 42)

                model.fit(X, y_train_comp, train_group = train_group[X_train_comp.index])
                pred = model.predict(X_test_comp, test_group = test_group[X_test_comp.index])

                AUC[str(n_re) + '-' + str(j)] = roc_auc_score(y_test_comp, pred)

                to_compare[str(n_re) + '-' + str(j)] = model.samplePosterior
                models_tmp[str(n_re) + '-' + str(j)] = model

        df_compare = az.compare(to_compare, ic = "loo")
        df_compare["AUC_test"] = AUC

        if len(df_compare[~df_compare.warning]) == 0:
            best_reach = True
            if n_re == 1:
                best_re = "None"
        else:
            new_loo = df_compare[~df_compare.warning].iloc[0].elpd_loo   
            if (n_re == 1) & (df_compare[~df_compare.warning].index[0] == 0):
                best_loo = df_compare[~df_compare.warning].iloc[0].elpd_loo
                best_re = "None"
                best_reach = True     
            elif new_loo <= best_loo: 
                best_reach = True       
            else: 
                best_loo = df_compare[~df_compare.warning].iloc[0].elpd_loo
                best_re = df_compare[~df_compare.warning].index[0]
                idx_random += [int(best_re[-1])]
                n_re += 1
        
    # Predictions Statsbombs
    #pred_statsbomb = Xy.loc[X_test_comp.index].shot_statsbomb_xg

    # Predictions Bayes
    pred_bayes = models_tmp[best_re].predict(X_test_comp, test_group = test_group[X_test_comp.index])
    
    AUC_comp.loc[len(AUC_comp)] = [comp_to_model, roc_auc_score(y_test_comp, pred_bayes)] #, roc_auc_score(y_test_comp, pred_statsbomb)]
    best_random_effects[str(comp_to_model[0]) + '-' + str(comp_to_model[1])] = [best_re, idx_random]
    
    return best_random_effects, AUC_comp
        
@st.cache_resource
def model_re_train(Xy, comp_to_model, all_comp, train_group, test_group, X_trainNorm, X_testNorm, X_train, y_train, X_test, y_test, params_used, best_random_effects):

    pred_test = pd.Series(index = y_test.index)
    shots_from_comp = Xy[np.sum(all_comp == comp_to_model, axis = 1) == 2]

    X_train_NotComp = X_trainNorm.loc[list(set(X_train.index) - set(shots_from_comp.index))]
    X_train_comp = X_trainNorm.loc[list(set(X_train.index) & set(shots_from_comp.index))]
    y_train_comp = y_train.loc[list(set(X_train.index) & set(shots_from_comp.index))]
    X_test_comp = X_testNorm.loc[list(set(X_test.index) & set(shots_from_comp.index))]
    y_test_comp = y_test.loc[list(set(X_test.index) & set(shots_from_comp.index))]

    # Build prior using other comp
    X = np.array(X_train_NotComp)

    BayesModel = pymc.Model()

    with BayesModel:
        alpha = pymc.Uniform("alpha", lower = -10, upper = 10)
        beta = pymc.Uniform("beta", lower = -10, upper = 10, shape = len(params_used))
        sigma = pymc.Gamma("sigma", mu = 1, sigma = 1/2)
        traceFixedEffectOnly = pymc.sample(draws = 10000, tune = 1000, random_seed = 42)

    sample_alpha = traceFixedEffectOnly.posterior["alpha"].values
    sample_beta = traceFixedEffectOnly.posterior["beta"].values
    cov_beta = np.cov(sample_beta[0].T)
    std_alpha = np.std(sample_alpha[0])
    g = 10 * np.mean(np.diag(cov_beta))
    cov_beta = cov_beta/g
    std_alpha = std_alpha/np.sqrt(g)
    mean_beta = sample_beta[0].mean(axis = 0) 
    mean_alpha = sample_alpha[0].mean()

    # Build model on studied competition
    X = np.array(X_train_comp)
    grp_unique = train_group[X_train_comp.index].unique()
    n_grp = len(grp_unique)
    
    idx_random = best_random_effects[str(comp_to_model[0]) + '-' + str(comp_to_model[1])][1]

    BayesModel = pymc.Model()
    with BayesModel:

        alpha = pymc.Normal("alpha", mu = mean_alpha, sigma = std_alpha)
        beta = pymc.MvNormal("beta", mu = mean_beta, cov = cov_beta)
        sigma = pymc.Gamma("sigma", mu = 1, sigma = 1/2)

        # Initialisation si pas d'effet aléatoire sur l'intercept
        omega = 0
        eta = np.zeros(n_grp)
        omega_beta = []
        eta_beta = []

        for i in range(len(params_used) + 1):
            if i in(idx_random):
                if i == 0: 
                    omega = pymc.Gamma("omega", mu = 0.3, sigma = 1/10)
                    eta = pymc.Normal("eta", mu = np.zeros(n_grp), sigma = np.repeat(omega, n_grp))
                else:
                    omega_beta.append(pymc.Gamma("omega_beta" + str(i), mu = 0.3, sigma = 1/10))
                    eta_beta.append(pymc.Normal("eta_beta" + str(i), mu = np.zeros(n_grp), sigma = np.repeat(omega_beta[i-1], n_grp)))
            else:
                if i != 0:
                    omega_beta.append(0)
                    eta_beta.append(np.zeros(n_grp))

        model = MAP_LR_model(beta_prior = beta,
                            alpha_prior = alpha,
                            sigma_prior = sigma, 
                            eta_prior = eta,
                            eta_beta_prior = eta_beta,
                            n_params = len(params_used), 
                            idx_random = idx_random,
                            random_seed = 42)

        model.fit(X, y_train_comp, train_group = train_group[X_train_comp.index])

        pred = model.predict(X_test_comp, test_group = test_group[X_test_comp.index])
        pred_test[X_test_comp.index] = pred
        #pred_statsbomb = Xy.loc[X_test_comp.index].shot_statsbomb_xg
        
    index_test = []
    index_test += list(set(Xy[np.sum(all_comp == comp_to_model, axis = 1) == 2].index) & set(y_test.index))
    #pred_statsbomb = Xy.loc[index_test].shot_statsbomb_xg
    y_test_comp = y_test[index_test]

    # AUC
    AUC = roc_auc_score(y_test_comp, pred_test.loc[index_test])
    #AUC_statsbomb = roc_auc_score(y_test_comp, pred_statsbomb)

    for seuil in [0.5, 0.3, 0.2, 0.1]:
        # Predictions
        y_pred = pred_test.loc[index_test] >= seuil
        #y_pred_sb = pred_statsbomb >= seuil

        # Confusion matrices
        confusionMatrix = confusion_matrix(y_test_comp, y_pred)
        #confusionMatrix_sb = confusion_matrix(y_test_comp, y_pred_sb)

        # Precision
        precision = precision_score(y_test_comp, y_pred)
        #precision_sb = precision_score(y_test_comp, y_pred_sb)

        # Recall
        recall = recall_score(y_test_comp, y_pred)
        #recall_sb = recall_score(y_test_comp, y_pred_sb)

        # Balanced accuracy
        b_acc = balanced_accuracy_score(y_test_comp, y_pred)
        #b_acc_sb = balanced_accuracy_score(y_test_comp, y_pred_sb)

        # Sensitivity
        sensitivity = confusionMatrix[1, 1] / (confusionMatrix[1, 1] + confusionMatrix[1, 0])
        #sensitivity_sb = confusionMatrix_sb[1, 1] / (confusionMatrix_sb[1, 1] + confusionMatrix_sb[1, 0])

        # Specificity
        specificity = confusionMatrix[0, 0] / (confusionMatrix[0, 0] + confusionMatrix[0, 1])
        #specificity_sb = confusionMatrix_sb[0, 0] / (confusionMatrix_sb[0, 0] + confusionMatrix_sb[0, 1])

        # F1
        F1 = f1_score(y_test_comp, y_pred, average = 'binary')
        #F1_sb = f1_score(y_test_comp, y_pred_sb, average = 'binary')

        #st.write('Seuil = ', seuil)
        #st.write("Confusion matrices(bayes, statsBomb):\n", confusionMatrix, "\n", confusionMatrix_sb)
        #dataframe = pd.DataFrame({"score": ['AUC', 'b_acc', 'precision', 'recall', 'sensitivity', 'specificity', 'F1_score'],
                            #"bayes": [AUC, b_acc, precision, recall, sensitivity, specificity, F1], 
                            #"statsBomb": [AUC_statsbomb, b_acc_sb, precision_sb, recall_sb, sensitivity_sb, specificity_sb, F1_sb]})
        #st.dataframe(dataframe)

    ROC = roc_curve(y_test_comp, pred_test.loc[index_test])
    #ROC_statsbomb = roc_curve(y_test_comp, pred_statsbomb)

    return ROC, confusionMatrix

@st.cache_resource
def features_selection(X_train_all, y_train):

    LR = LogisticRegression(penalty = None, max_iter = 1000)
    list_params = X_train_all.columns
    params_used_auc = []
    model_scores_auc = []

    for i in range(15):
        scores = []
        params_not_used = list(set(list_params) - set(params_used_auc))
        for par in tqdm(params_not_used):
            params_tmp = params_used_auc + [par]
            lr = cross_val_score(LR, X = X_train_all[params_tmp], y = y_train, cv = 5, scoring = 'roc_auc')
            scores.append(lr.mean())
        params_used_auc.append(params_not_used[np.array(scores).argmax()])
        model_scores_auc.append(max(scores))

    return params_used_auc, model_scores_auc

def click_button_fs():

    st.session_state.fs_clicked = True

def change_predict_variable():
    
    st.session_state.fs_clicked = False

def change_competition():

    st.session_state.best_re_clicked = False
    st.session_state.model_re_clicked = False

def click_estimation():

    st.session_state.estimation_clicked = True

def click_estimation_2():

    st.session_state.estimation_clicked_2 = True

def change_re_choice():

    st.session_state.estimation_clicked_2 = False

def change_features():

    st.session_state.estimation_clicked = False
    st.session_state.estimation_clicked_2 = False