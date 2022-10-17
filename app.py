from st_aggrid import AgGrid
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import time
import pickle
import json
import streamlit as st
import streamlit_authenticator as stauth
from streamlit import session_state as ss
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_breast_cancer
from xgboost import XGBClassifier, plot_importance
import database as db


# settings
# st.set_page_config(layout="wide")

# utility functions
def get_ohe_col_name(ohe, category_cols, drop='if_binary'):
    """
    Get col names after one hot encoding

    :param ohe: OneHotEncoder transforer
    :param category_cols: categorical columns that need to be encoded
    :param drop: whether encode binary cols
    """

    cat_cols_new = []
    col_values = ohe.categories_

    for i, j in enumerate(category_cols):
        if (drop == 'if_binary') and (len(col_values[i]) == 2):
            cat_cols_new.append(j)
        else:
            for v in col_values[i]:
                feature_name = j + '_' + str(v)
                cat_cols_new.append(feature_name)
    return cat_cols_new


def get_metrics(model, X_test, y_test):
    """
    Pass model and testing set to get metrics

    :param model: a sklearn model or XGBoost model
    :param X: data set of features
    :param y: data set of label
    """
    # Make predictions
    y_pred = model.predict(X_test)
    try:
        probab = model.decision_function(X_test)
    except:
        probab = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    roc = roc_curve(y_test, probab)

    return precision, recall, f1, accuracy, roc


# Cached data
@st.experimental_memo
def load_data(file):
    df = pd.read_csv(file, encoding='utf-8')
    return df


@st.experimental_memo
def load_example_data():
    df = pd.read_csv('credit.csv', encoding='utf-8')
    return df


# Authentication
users = db.fetch_all_users()
usernames = [user['_id'] for user in users]
names = [user['name'] for user in users]
passwords = [user['password'] for user in users]
hashed_passwords = stauth.Hasher(passwords).generate()

authenticator = stauth.Authenticate(names, usernames, hashed_passwords,
                                    "no-code-xgboost ", "abcdef", cookie_expiry_days=30)

name, authentication_status, username = authenticator.login('Login', 'sidebar')

if authentication_status == False:
    st.sidebar.error("Username/password is incorrect")

if authentication_status == None:
    st.sidebar.warning("You can still use the app without login, "
                       "but the model results won't be saved for later retrieval.")

if authentication_status:
    authenticator.logout("Logout", "sidebar")
    if ss['logout']:
        use_example_data = False
        for key in ss.keys():
            del ss[key]
    st.sidebar.subheader(f"Welcome, {name} 👨‍💻")

# Newly defined session states
ss['filename'] = ss.get('filename', None)
ss['split'] = ss.get('split', False)
# ss['feature_cols'] = ss.get('feature_cols', col_list)
ss['label_col'] = ss.get('label_col', None)
# ss['label_index'] = ss.get('label_index', default_label_index)
ss['seed'] = ss.get('seed', 1024)
ss['test_size'] = ss.get('test_size', 0.33)
ss['X_train'] = ss.get('X_train', None)
ss['X_test'] = ss.get('X_test', None)
ss['y_train'] = ss.get('y_train', None)
ss['y_test'] = ss.get('y_test', None)
ss['xgb'] = ss.get('xgb', None)
ss['run_model'] = ss.get('run_model', False)
ss['model_metrics'] = ss.get('model_metrics', None)
ss['fp_r'] = ss.get('fp_r', None)
ss['tp_r'] = ss.get('tp_r', None)
ss['model_date'] = ss.get('model_date', None)
ss['model_time'] = ss.get('model_time', None)
ss['pickled_model'] = ss.get('pickled_model', None)

st.header('No Code XGBoost')
st.info('This web app allows the user to run XGBoost models without writing a single line of code.', icon="ℹ")

uploaded_file = st.file_uploader('Upload a CSV file', type="csv", key='file_uploader')
use_example_data = st.checkbox('Use example dataset (Sklearn datasets will be supported in later development)',
                               value=True)

if use_example_data:
    data = load_example_data()
    ss['filename'] = 'credit.csv'
    col_names = data.columns

if uploaded_file is not None:
    data = load_data(uploaded_file)
    ss['filename'] = uploaded_file.name
    col_names = data.columns

if use_example_data or uploaded_file is not None:
    st.sidebar.subheader('Navigation')
    option_list = ['📊 Data Exploration', '⏳ Parameter Tuning', '🚀 Run Model', '⚡ Modeling History']

    navigation_vertical = st.sidebar.radio('go to', option_list)
    df = data.copy()
    if navigation_vertical == '📊 Data Exploration':
        col_options = st.multiselect('Specify the categorical columns', col_names)
        df[col_options] = df[col_options].astype(str)

        st.subheader('Explore the data')
        with st.expander('Click to show the data'):
            st.dataframe(df)
        with st.expander('Click to show the data description'):
            st.dataframe(df.describe())

    if navigation_vertical == '⏳ Parameter Tuning':
        st.write('Will be implemented shortly using Optuna')

    if navigation_vertical == '🚀 Run Model':
        st.subheader('Prepare training and testing data')
        col_list = list(col_names)
        default_label_index = len(col_names) - 1
        ss['feature_cols'] = ss.get('feature_cols', col_list)
        ss['label_index'] = ss.get('label_index', default_label_index)

        with st.form("split_data_form"):
            feature_cols = st.multiselect('Exclude features not used in the model by clicking the X',
                                          col_names, ss['feature_cols'])

            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                label_col = st.selectbox('Select the label variable', col_names, index=ss['label_index'],
                                         help='One-hot encoding will be don on categorical variables')
                label_index = col_list.index(label_col)
            with p_col2:
                seed = st.number_input('seed', value=ss['seed'],
                                       help='The random seed help you reproduce the dataset later.')
            with p_col3:
                test_size = st.number_input('test_size', min_value=0.0, max_value=1.0, value=ss['test_size'])

            split_button = st.form_submit_button("Split the data")
            if split_button:
                ss['split'] = True
                ss['feature_cols'] = feature_cols
                ss['label_col'] = label_col
                ss['label_index'] = label_index
                ss['seed'] = seed
                ss['test_size'] = test_size

                features_df = df[ss['feature_cols']]
                cat_cols = list(features_df.select_dtypes(include='object').columns)
                ohe = OneHotEncoder(drop='if_binary')
                df_cat = df[cat_cols]
                ohe.fit(df_cat)

                df_cat = pd.DataFrame(ohe.transform(df_cat).toarray(), columns=get_ohe_col_name(ohe, cat_cols))

                none_cat_cols = list(features_df.select_dtypes(exclude='object').columns)
                df_non_cat = features_df[features_df[none_cat_cols]]

                # X = df[ss['feature_cols']]
                X = pd.concat([df_cat, df_non_cat], axis=1)
                y = df[ss['label_col']]
                ss['X_train'], ss['X_test'], ss['y_train'], ss['y_test'] = train_test_split(X, y,
                                                                                            test_size=ss['test_size'],
                                                                                            random_state=ss['seed'],
                                                                                            stratify=y)

        if ss['X_train'] is not None:
            X_train, X_test, y_train, y_test = ss['X_train'], ss['X_test'], ss['y_train'], ss['y_test']
            st.subheader('Check the training and testing data')
            with st.expander('Click to fold/unfold', expanded=True):
                X_train_tab, X_test_tab, y_train_tab, y_test_tab = st.tabs(['X_train', 'X_test', 'y_train', 'y_test'])
                X_train_tab.dataframe(X_train.head())
                X_test_tab.dataframe(X_test.head())
                y_train_tab.dataframe(y_train.head())
                y_test_tab.dataframe(y_test.head())

        st.subheader('Specify parameters (optional)')
        with st.expander('Click to fold/unfold', expanded=True):
            with st.form("params_form"):
                col1, col2, col3, col4 = st.columns(4)

                params = {}
                with col1:
                    params['alpha'] = st.number_input('alpha', value=0)
                    params['lambda'] = st.number_input('lambda', value=1)

                with col2:
                    params['eta'] = st.number_input('eta', min_value=0.0, max_value=1.0, value=0.3)
                    params['gamma'] = st.number_input('gamma', min_value=0, value=0)

                with col3:
                    params['max_depth'] = st.number_input('max_depth', min_value=0, value=6)
                    params['min_child_weight'] = st.number_input('min_child_weight', min_value=0, value=1)

                with col4:
                    params['scale_pos_weight'] = st.number_input('scale_pos_weight', value=1)
                    params['grow_policy'] = st.selectbox('grow_policy', ['depthwise', 'lossguide'])

                param_button = st.form_submit_button("Done")
                if param_button:
                    st.markdown('**Here are the parameters you specified:**')
                    params

        st.subheader('Click the button below to run the model')
        # ss['xgb'] = ss.get('xgb', None)
        # ss['run_model'] = ss.get('run_model', False)
        # ss['model_metrics'] = ss.get('model_metrics', None)
        # ss['fp_r'] = ss.get('fp_r', None)
        # ss['tp_r'] = ss.get('tp_r', None)

        model_metrics = {}
        if st.button('Run XGBoost 🚀'):
            ss['run_model'] = True
            # st.write(ss['X_train'].head())
            X_train, X_test, y_train, y_test = ss['X_train'], ss['X_test'], ss['y_train'], ss['y_test']


            @st.experimental_singleton
            def run_xgb(X_train, y_train, params):
                xgb = XGBClassifier(objective="binary:logistic", eval_metric="auc", use_label_encoder=False)
                xgb.set_params(**params)
                xgb.fit(X_train, y_train)

                return xgb


            xgb = run_xgb(X_train, y_train, params)

            st.success('Model runs successfully!')

            model_metrics['precision'], model_metrics['recall'], model_metrics['f1'], model_metrics[
                'accuracy'], roc = get_metrics(xgb, X_test, y_test)
            fp_r, tp_r, thresholds = roc
            model_metrics['auc_score'] = metrics.auc(fp_r, tp_r)
            ss['model_metrics'] = model_metrics
            ss['fp_r'], ss['tp_r'] = fp_r, tp_r
            ss['xgb'] = xgb
            pickled_model = pickle.dumps(xgb)
            ss['pickled_model'] = pickled_model

            model_datetime = str(datetime.now())
            model_date = model_datetime[:10]
            model_time = model_datetime[11:19]
            ss['model_date'] = model_date
            ss['model_time'] = model_time

            if authentication_status:
                db.insert_model(username, ss['filename'], pickled_model, model_date, model_time,
                                ss['feature_cols'], ss['label_col'], params, model_metrics)
                st.success('The model results have been saved!')

        if ss['run_model']:
            model_metrics = ss['model_metrics']
            pickled_model = ss['pickled_model']
            model_date, model_time = ss['model_date'], ss['model_time']
            pickled_file_name = f'xgboost{model_date}_{model_time}.pkl'


            def get_json_info_file(filename, model_date, model_time, used_features, label_name, params, metrics):
                """Get the model info in json"""

                data = {'filename': filename, 'model_date': model_date, 'model_time': model_time,
                        'used_features': used_features, 'label_name': label_name, 'params': params, 'metrics': metrics}
                model_string = json.dumps(data)
                return model_string


            model_string = get_json_info_file(ss['filename'], model_date, model_time, ss['feature_cols'],
                                              ss['label_col'], params, model_metrics)
            json_file_name = f'xgboost{model_date}_{model_time}.json'

            download_col1, download_col2 = st.columns(2)

            with download_col1:
                st.download_button(
                    'Download Model',
                    data=pickled_model,
                    file_name=pickled_file_name
                )

            with download_col2:
                st.download_button(
                    'Download Model Info as a Json file',
                    data=model_string,
                    file_name=json_file_name,
                    mime='application/json',
                )
            _, X_test, _, y_test = ss['X_train'], ss['X_test'], ss['y_train'], ss['y_test']

            fp_r, tp_r = ss['fp_r'], ss['tp_r']
            st.subheader('Check the model results')
            tab1, tab2, tab3 = st.tabs(["🔢 Metrics", "⭐ ROC Curve", "🎯 Feature Importance Chart"])

            with tab1:
                m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
                m_col1.metric("AUC", round(model_metrics['auc_score'], 3))
                m_col2.metric("Precision", round(model_metrics['precision'], 3))
                m_col3.metric("Recall", round(model_metrics['recall'], 3))
                m_col4.metric("F1 Score", round(model_metrics['f1'], 3))
                m_col5.metric("Accuracy", round(model_metrics['accuracy'], 3))

            with tab2:
                fig1 = plt.figure(figsize=(8, 6))
                plt.plot(fp_r, tp_r, label="AUC = %.3f" % model_metrics['auc_score'])
                plt.plot([0, 1], [0, 1], "r--")
                plt.ylabel("TP rate")
                plt.xlabel("FP rate")
                plt.legend(loc=4)
                plt.title("ROC Curve")
                st.pyplot(fig1)

            with tab3:
                fig2, ax = plt.subplots(figsize=(6, 9))
                plot_importance(ss['xgb'], max_num_features=50, height=0.8, ax=ax)
                st.pyplot(fig2)

    if navigation_vertical == '⚡ Modeling History':
        if authentication_status:
            models = db.fetch_all_models(username)
            # st.write(models[0])
            # st.json(models, expanded=False)
            # st.json(models, expanded=True)
            model_df = pd.DataFrame(models)
            del model_df['_id']
            del model_df['username']

            st.markdown('**Here are all the previous models.**')
            st.dataframe(model_df)

            st.markdown('**Pick a specific model to check out**')
            filter_col1, filter_col2 = st.columns(2)

            with filter_col1:
                selected_day = st.date_input('which day to choose')

            type(selected_day)
            time_list = model_df.loc[model_df['model_date'] == str(selected_day), 'model_time']

            default_time_index = len(time_list) - 1

            with filter_col2:
                selected_time = st.selectbox('pick a time', time_list, index=default_time_index)


            # filter the model json
            def get_selected_model(model_day, model_time):
                selected_model = None
                for model in models:
                    if model['model_date'] == model_day and model['model_time'] == model_time:
                        selected_model = model
                        break
                return selected_model


            # st.write(models[0]['model_date'])
            # selected_model = None
            # for model in models:
            #     if model['model_date'] == str(selected_day) and model['model_time'] == str(selected_time):
            #         selected_model = model
            #         break
            #
            selected_model = get_selected_model(str(selected_day), str(selected_time))
            selected_pickled_model = selected_model['pickled_model']
            del selected_model['_id']
            del selected_model['pickled_model']
            st.write(selected_model)

            # AgGrid(model_df)

        else:
            st.warning('You need to log in to retrieve previous models', icon="⚠️")
