from st_aggrid import AgGrid
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import time
import pickle
import json
import streamlit as st
import streamlit_authenticator as stauth
from streamlit_option_menu import option_menu
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


# navigation menu
option_list = ['Data Exploration', 'Parameter Tuning', 'Run Model', 'Modeling History']

with st.sidebar:
    st.title('No Code XGBoost')
    st.markdown("""by Feng Zhao [![](https://i.stack.imgur.com/gVE0j.png) ](https://www.linkedin.com/in/felixfengzhao/)
                &nbsp; [![](https://i.stack.imgur.com/tskMh.png) ](https://github.com/iFengZhao/no-code-xgboost)""")

    navigation_vertical = option_menu(
        menu_title=None,
        options=option_list,
        icons=['activity', 'body-text', 'caret-right-square', 'cloud-arrow-down'],  # https://icons.getbootstrap.com/
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"font-size": "15px"},
            "nav-link": {"font-size": "15px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
        }
    )

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
    st.sidebar.subheader(f"Welcome, {name} ???????????")
    authenticator.logout("Logout", "sidebar")
    if ss['logout']:
        ss['use_example_dataset'] = False
        for key in ss.keys():
            del ss[key]

# Newly defined session states
ss['filename'] = ss.get('filename', None)
ss['data'] = ss.get('data', None)
ss['col_names'] = ss.get('col_names', None)
ss['split'] = ss.get('split', False)
ss['use_example_dataset'] = ss.get('use_example_dataset', False)
ss['label_col'] = ss.get('label_col', None)
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

ss_list = ['data', 'col_names', 'split', 'use_example_dataset',
           'feature_cols', 'label_col', 'label_index', 'seed',
           'test_size', 'X_train', 'X_test', 'y_train', 'y_test',
           'xgb', 'run_model', 'model_metrics','fp_r', 'tp_r',
           'model_date', 'model_time', 'pickled_model']


def clear_session_states():
    """
    clear session states when needed
    :return:
    """
    for key in ss.keys():
        if key in ss_list:
            del ss[key]


if navigation_vertical == 'Data Exploration':
    st.header('???? Data Exploration')
    uploaded_file = st.file_uploader('Upload a CSV file', type="csv", key='file_uploader')
    use_example_data = st.checkbox('Use example dataset (Sklearn datasets will be supported in later development.)',
                                   value=ss['use_example_dataset'])
    if use_example_data:
        ss['data'] = load_example_data()
        ss['filename'] = 'credit.csv'
        ss['col_names'] = ss['data'].columns
    else: ss['use_example_dataset'] = False

    if uploaded_file is not None:
        ss['use_example_dataset'] = False
        ss['data'] = load_data(uploaded_file)
        ss['filename'] = uploaded_file.name
        ss['col_names'] = ss['data'].columns

    if not use_example_data and uploaded_file is None:
        ss['data'] = None
        clear_session_states()

    if use_example_data or uploaded_file is not None:
        df = ss['data'].copy()
        col_names = ss['col_names']
        col_options = st.multiselect('Specify the categorical columns', col_names)
        df[col_options] = df[col_options].astype(str)
        ss['data'] = df

        st.subheader('Explore the data')
        with st.expander('Click to show the data'):
            st.dataframe(df)
        with st.expander('Click to show the data description'):
            st.dataframe(df.describe())

if navigation_vertical == 'Parameter Tuning':
    st.header('??? Parameter Tuning')
    st.write('Will be implemented shortly using Optuna')

if navigation_vertical == 'Run Model':
    st.header('???? Run Model')
    if ss['data'] is None:
        st.info('Please upload a dataset or use the example dataset', icon="??????")
    else:
        df = ss['data']
        col_names = ss['col_names']
        st.subheader('Prepare training and testing data')
        col_list = list(col_names)
        default_label_index = len(col_names) - 1
        ss['feature_cols'] = ss.get('feature_cols', col_list)
        ss['label_index'] = ss.get('label_index', default_label_index)

        with st.form("split_data_form"):
            feature_cols = st.multiselect('Exclude features not used in the model by clicking the X',
                                          col_names, ss['feature_cols'],
                                          help='Categorical variables will be one-hot encoded')

            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                label_col = st.selectbox('Select the label variable', col_names, index=ss['label_index'],
                                         help='Select the label variables')
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
                df_non_cat = features_df[none_cat_cols]

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
        model_metrics = {}
        if st.button('Run XGBoost ????'):
            ss['run_model'] = True
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
                    '?????? Download Pickled Model ',
                    data=pickled_model,
                    file_name=pickled_file_name
                )

            with download_col2:
                st.download_button(
                    '?????? Download Model Info as a Json file ',
                    data=model_string,
                    file_name=json_file_name,
                    mime='application/json',
                )

            _, X_test, _, y_test = ss['X_train'], ss['X_test'], ss['y_train'], ss['y_test']

            fp_r, tp_r = ss['fp_r'], ss['tp_r']
            st.subheader('Check the model results')
            tab1, tab2, tab3 = st.tabs(["???? Metrics", "??? ROC Curve", "???? Feature Importance Chart"])

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

if navigation_vertical == 'Modeling History':
    st.header('??? Modeling History')
    if authentication_status:
        models = db.fetch_all_models(username)

        model_df = pd.DataFrame(models)
        del model_df['_id']
        del model_df['username']

        st.markdown('**Here are all the previous models.**')
        st.dataframe(model_df)

        st.markdown('**Pick a specific model to check out**')
        filter_col1, filter_col2 = st.columns(2)

        latest_day = max(pd.to_datetime(model_df['model_date']))

        with filter_col1:
            selected_day = st.date_input('which day to choose', latest_day)

        type(selected_day)
        time_list = model_df.loc[model_df['model_date'] == str(selected_day), 'model_time']

        default_time_index = len(time_list) - 1

        with filter_col2:
            selected_time = st.selectbox('pick a time', time_list, index=default_time_index)


        # filter the model json
        def get_selected_model(model_day, model_time):
            """
            Select model from the database
            :param model_day: selected model day
            :param model_time: selected model time
            :return: selected model
            """
            selected_model = None
            for model in models:
                if model['model_date'] == model_day and model['model_time'] == model_time:
                    selected_model = model
                    break
            return selected_model


        selected_model = get_selected_model(str(selected_day), str(selected_time))
        selected_pickled_model = selected_model['pickled_model']
        del selected_model['_id']
        del selected_model['pickled_model']
        st.write(selected_model)

        download_file_name = f'xgboost{selected_day}_{selected_time}.pkl'
        st.download_button(
            '?????? Download Pickled Model ',
            data=selected_pickled_model,
            file_name=download_file_name
        )

        # AgGrid(model_df)

    else:
        st.warning('You need to log in to retrieve previous models', icon="??????")
