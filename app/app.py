import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


import utils


st.set_page_config(page_title = "xG Bayesian Modeling",
                   layout = 'wide',
                   page_icon = "⚽")

st.markdown("""
        <style>
               .block-container {
                    padding-top: 2rem;
                    padding-bottom: 2rem;
                    padding-left: 4rem;
                    padding-right: 3rem;
                }
        </style>
        """, unsafe_allow_html = True)

st.subheader("🎯Bayesian Mixed Modeling of Expected Goals")
add_vertical_space(2)
st.markdown("""
            This study uses a Bayesian generalized linear mixed-effects
            model to introduce a simple and interpretable xG modeling approach. <br>
            The model provided similar performances when compared to the StatsBomb model,
            property of StatsBomb company, with only seven variables about the *shot type*, *position*,
            and *surrounding opponents* ($AUC = 0.781$ and $AUC = 0.801$, respectively). <br>
            Pre-trained models through transfer learning are suitable for identifying teams’ strengths
            and weaknesses in small sample sizes and enabling interpretation of the model’s predictions.
            <br>
            <br>
            [GitHub repository]() </br>
            [DOI]()
            """, unsafe_allow_html = True)

# DATASET
######################################################################################################################################################
st.markdown("---")
st.markdown("##### 🔎 Data processing :")
col1, col2 = st.columns([1, 1.7])
Xy = col1.file_uploader("Select your dataset :",
                                accept_multiple_files = False,
                                type = 'csv')

if Xy:

    with col2:
            
        Xy = pd.read_csv(Xy, sep = ';')
        if 'Unnamed: 0' in Xy.columns:
                Xy = Xy.drop(columns = {'Unnamed: 0'})

        st.dataframe(Xy.head(3))

    st.info(
                """
                Each row in your dataset should represent **one event** for **one player** during a match.  
                """,
                icon = "ℹ️"
            )
    col1, col2, col3 = st.columns([1, 1.3, 1])
    variable_y = col1.selectbox("Select your variable of interest :", Xy.columns, on_change = utils.change_predict_variable, index = len(Xy.columns) - 1)

    if 'multiselect' not in st.session_state:
            st.session_state.multiselect = False
    quali_features = col2.multiselect("Select your qualitative features :", Xy.columns[Xy.columns != variable_y].tolist(), on_change = utils.change_predict_variable)

    variable_competition = col1.selectbox("Select your variable indicating competitions :", Xy.columns, on_change = utils.change_predict_variable, index = len(Xy.columns) - 3)
    variable_seasons = col2.selectbox("Select your variable indicating seasons :", Xy.columns, on_change = utils.change_predict_variable, index = len(Xy.columns) - 2)
    variable_teams = col3.selectbox("Select your variable indicating teams :", Xy.columns, on_change = utils.change_predict_variable, index = len(Xy.columns) - 4)
    
    exclude_columns = [variable_y, variable_competition, variable_teams, variable_seasons]
    # Performing one-hot encoding on the quali_features
    X = pd.get_dummies(data = Xy.loc[:, ~Xy.columns.isin(exclude_columns)], columns = quali_features)
    #st.dataframe(X)

    y = Xy[variable_y]
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.3,
                                                    random_state = 42)

    X_train_all = X_train.copy()
    X_test_all = X_test.copy()

    comp_seas = []
    for comps in Xy[variable_competition].unique():
        for seas in Xy[Xy[variable_competition] == comps][variable_seasons].unique():
            comp_seas.append([comps, seas])

    all_comp = np.concatenate([np.reshape(np.array(Xy[variable_competition]), (-1, 1)),
                                np.reshape(np.array(Xy[variable_seasons]), (-1, 1))], axis = 1)
    
    comp_to_model = col3.selectbox("Select your competition of interest :", comp_seas, on_change = utils.change_competition())

# SELECTION DE VARIABLES
######################################################################################################################################################
    st.markdown("---")
    st.markdown("##### 📋 Features selection :")
    col1, col2, col3 = st.columns([1, 0.5, 1])
    
    if 'fs_clicked' not in st.session_state:
        st.session_state.fs_clicked = False

    fs_button = col2.button('Features Selection',
                            help = 'Forward selection by adding one by one the features maximizing the AUC criterion through cross-validation',
                            on_click = utils.click_button_fs,
                            use_container_width = True)
                                    
    if st.session_state.fs_clicked == True:

        try:
            scaler = StandardScaler()
            X_trainNorm = pd.DataFrame(scaler.fit(X_train).transform(X_train), columns = X_train.columns, index = X_train.index)
            X_testNorm = pd.DataFrame(scaler.transform(X_test), columns = X_test.columns, index = X_test.index)
            
            #params_used_auc = ['best_angle',
                                #'distance_goalkeeper_to_goal',
                                #'body_part_Head',
                                #'distance',
                                #'angle',
                                #'nb_opponent_traj',
                                #'closest_opponent',
                                #'opponent_nearby',
                                #'body_part_Other',
                                #'position_Left Attacking Midfield',
                                #'position_Left Center Back',
                                #'position_Right Midfield',
                                #'position_Left Defensive Midfield',
                                #'competition_id',
                                #'position_Center Attacking Midfield']
            #model_scores_auc = [0.760025569961749,
                                #0.7797377196138336,
                                #0.7864159471752693,
                                #0.7957194782542357,
                                #0.7988373951878763,
                                #0.8022607740475994,
                                #0.8056781510718516,
                                #0.805984769514672,
                                #0.8061360048730621,
                                #0.8062448712649326,
                                #0.8063564954900775,
                                #0.8064259654323995,
                                #0.806496904371453,
                                #0.8065221147450456,
                                #0.8066105942188673]

            params_used_auc, model_scores_auc = utils.features_selection(X_train_all, y_train)

            col1, col2, col3 = st.columns([1.2, 0.3, 1])
            with col1:
                labels = params_used_auc.copy()

                fig = go.Figure()
                fig.add_trace(go.Scatter(x = list(range(1, len(model_scores_auc) + 1)),
                                                y = model_scores_auc,
                                                mode = 'lines+markers',
                                                name = 'AUC Scores',
                                                line = dict(color = 'darkorchid'),
                ))

                fig.update_layout(title = {'text': "AUC Cross Validation Score by Added Feature",
                                            'y': 0.9},
                                        xaxis = dict(tickmode = 'array',
                                                tickvals = list(range(1, len(model_scores_auc) + 1)),
                                                ticktext = labels,
                                                tickangle = -45
                                                ),
                                        yaxis = dict(title = "AUC Cross Validation Score"),
                                        margin = dict(t = 80),
                                        showlegend = False,
                                        plot_bgcolor = 'white'
                )

                fig.update_yaxes(showgrid = False)

                st.plotly_chart(fig, use_container_width = True)

            with col3:
                add_vertical_space(2)
                with st.container(border = False, height = 210):
                    fs_last_variable = st.radio("Select the last feature to keep :", labels, on_change = utils.change_features)
                features_used = labels[:params_used_auc.index(fs_last_variable) + 1]
                add_vertical_space(1)
                st.write("Your explanatory features retained :")
                st.code(features_used, 'python')
        
            #features_used = ['best_angle', 'distance_goalkeeper_to_goal', 'body_part_Head', 'distance', 'angle', 'nb_opponent_traj', 'closest_opponent']
            X_train = X_train_all[features_used]
            X_test = X_test_all[features_used]
            X_trainNorm = pd.DataFrame(scaler.transform(X_train_all), columns = X_train_all.columns, index = X_train.index)[features_used]
            X_testNorm = pd.DataFrame(scaler.transform(X_test_all), columns = X_test_all.columns, index = X_test.index)[features_used]

            # Group by team
            train_group = Xy[variable_teams][X_train.index]
            test_group = Xy[variable_teams][X_test.index]

            # MEILLEURS EFFETS ALEATOIRES
            ######################################################################################################################################################
            st.markdown("---")
            st.markdown("##### 🎲 Attribution of random effects on selected explanatory features :")

            col1, col2 = st.columns([1, 1])
            choix_re = col1.radio("",
                                ["Estimation", "Choice"],
                                captions = ["Best random effects are estimated by the algorithm", "Choose yourself the variables (maximum 3) on which you wish to apply a random effect"])
            
            # CHOIX ESTIMATION PAR L'ALGORITHME
            ######################################################################################################################################################
            if choix_re == "Estimation":
                try:
                    if 'estimation_clicked' not in st.session_state:
                        st.session_state.estimation_clicked = False
                    estimation = col2.button("Estimation algorithm", on_click = utils.click_estimation)

                    if st.session_state.estimation_clicked == True:
                        #best_random_effects = {'9-281': ['None', []]} 
                        #best_random_effects = {'43-106': ['2-5', [0, 5]]}
                        with col2:
                            best_random_effects, AUC_comp = utils.best_re_selection(Xy, comp_to_model, all_comp, train_group, test_group, X_trainNorm, X_testNorm, X_train, y_train, X_test, y_test, features_used)
                            st.markdown("Selected features by the algorithm are :")
                            idx_re = best_random_effects[str(comp_to_model[0]) + '-' + str(comp_to_model[1])][1]
                            features_used_2 = features_used.copy()
                            features_used_2.insert(0, 'intercept')
                            st.code([features_used_2[i] for i in idx_re])

                        # ENTRAINEMENT DU MODELE
                        ######################################################################################################################################################
                        st.markdown("---")
                        st.markdown("##### 👾 Model learning :")
                        col1, col2, col3 = st.columns([1, 0.7, 1])
                        try:
                        
                            model_re = col2.button('Model with estimated Random Effects',
                                            help = '',
                                            use_container_width = True)
                                        
                            if model_re:

                                ROC, confusionMatrix = utils.model_re_train(Xy, comp_to_model, all_comp, train_group, test_group, X_trainNorm, X_testNorm, X_train, y_train, X_test, y_test, features_used, best_random_effects)

                                col1, col2, col3 = st.columns([1, 0.1, 1])
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x = ROC[0], y = ROC[1], mode = 'lines', name = 'Bayesian model', line = dict(color = 'darkorange')))
                                #fig.add_trace(go.Scatter(x = ROC_statsbomb[0], y = ROC_statsbomb[1], mode = 'lines', name = 'StatsBomb model', line = dict(color = 'darkturquoise')))
                                fig.add_trace(go.Scatter(x = [0, 1], y = [0, 1], mode = 'lines', line = dict(color = 'black', dash = 'dot'), name = 'Reference Line'))

                                fig.update_layout(title = {'text': "ROC curve",
                                                        'y': 0.9},
                                                xaxis = dict(title = 'Specificity', range = [0, 1], zeroline = False),
                                                yaxis = dict(title = 'Sensitivity', range = [0, 1]),
                                                legend = dict(x = 0.9, y = 0.1, xanchor = 'center'),
                                                )

                                fig.update_yaxes(showgrid = False)

                                col1.plotly_chart(fig, use_container_width = True)
                                with col3:
                                    add_vertical_space(1)
                                    st.markdown("Confusion Matrix")
                                    st.dataframe(confusionMatrix, use_container_width = True)

                        except Exception as e:
                            st.write(e)
                
                except Exception as e:
                    st.write(e)

            # CHOIX MANUEL
            ######################################################################################################################################################
            else:

                features_used_2 = features_used.copy()
                features_used_2.insert(0, 'intercept')
                choices_re = col2.multiselect("Features :", features_used_2, max_selections = 3, on_change = utils.change_re_choice)
                col2.markdown("Selected features by you are :")
                col2.code(choices_re)

                indices = [features_used_2.index(element) for element in choices_re]

                best_random_effects = {str(comp_to_model[0]) + '-' + str(comp_to_model[1]): ['', [0, 5]]}
                best_random_effects[str(comp_to_model[0]) + '-' + str(comp_to_model[1])][1] = indices
                #st.code(best_random_effects)

                # ENTRAINEMENT DU MODELE
                ######################################################################################################################################################
                st.markdown("---")
                st.markdown("##### 👾 Model learning :")
                col1, col2, col3 = st.columns([1, 0.7, 1])
                try:
                
                    if 'estimation_clicked_2' not in st.session_state:
                        st.session_state.estimation_clicked_2 = False
                    model_re = col2.button("Model with choosen Random Effects", on_click = utils.click_estimation_2)

                    col1, col2, col3 = st.columns([1, 0.1, 1])
                    if st.session_state.estimation_clicked_2 == True:
        
                        ROC, confusionMatrix = utils.model_re_train(Xy, comp_to_model, all_comp, train_group, test_group, X_trainNorm, X_testNorm, X_train, y_train, X_test, y_test, features_used, best_random_effects)

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x = ROC[0], y = ROC[1], mode = 'lines', name = 'Bayesian model', line = dict(color = 'darkorange')))
                        #fig.add_trace(go.Scatter(x = ROC_statsbomb[0], y = ROC_statsbomb[1], mode = 'lines', name = 'StatsBomb model', line = dict(color = 'darkturquoise')))
                        fig.add_trace(go.Scatter(x = [0, 1], y = [0, 1], mode = 'lines', line = dict(color = 'black', dash = 'dot'), name = 'Reference Line'))

                        fig.update_layout(title = {'text': "ROC curve",
                                                'y': 0.9},
                                        xaxis = dict(title = 'Specificity', range = [0, 1], zeroline = False),
                                        yaxis = dict(title = 'Sensitivity', range = [0, 1]),
                                        legend = dict(x = 0.9, y = 0.1, xanchor = 'center'),
                                        )

                        fig.update_yaxes(showgrid = False)

                        col1.plotly_chart(fig, use_container_width = True)
                        with col3:
                            add_vertical_space(1)
                            st.markdown("Confusion Matrix")
                            st.dataframe(confusionMatrix, use_container_width = True)

                except Exception as e:
                    st.write(e)

        except ValueError:
            st.warning("You have qualitative variables in your data to specify", icon = '⚠️')
        #except Exception as e:
            #st.write(e)
