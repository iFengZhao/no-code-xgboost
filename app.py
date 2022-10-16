from st_aggrid import AgGrid
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import time
import streamlit as st
import streamlit_authenticator as stauth
from streamlit import session_state as ss
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from xgboost import XGBClassifier, plot_importance
import database as db

# settings
st.set_page_config(layout="wide")

# utility functions
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

st.header('No Code XGBoost')
st.info('This web app allows the user to run XGBoost models without writing a single line of code.', icon="ℹ")

uploaded_file = st.file_uploader('Upload a CSV file', type="csv", key='file_uploader')
use_example_data = st.checkbox('Use example dataset (Sklearn datasets will be supported in later development)')

if use_example_data:
    df = load_example_data()
    ss['filename'] = 'credit.csv'
    col_names = df.columns

if uploaded_file is not None:
    df = load_data(uploaded_file)
    ss['filename'] = uploaded_file.name
    col_names = df.columns

if use_example_data or uploaded_file is not None:
    st.sidebar.subheader('Navigation')
    option_list = ['📊 Data Exploration', '⏳ Parameter Tuning', '🚀 Run Model', '⚡ Modeling History']

    navigation_vertical = st.sidebar.radio('go to', option_list)

    # st.subheader("Navigation")
    main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs(option_list)
    #
    # with main_tab1:
    if navigation_vertical == '📊 Data Exploration':
        st.subheader('Explore the data')
        with st.expander('Click to show the data'):
            st.dataframe(df)
        # with st.expander('Click to show the data description'):
        #     st.dataframe(df.describe(include='all'))

        col_options = st.multiselect('Specify the categorical columns', col_names)
        df[col_options] = df[col_options].astype(str)

    # with main_tab2:
    if navigation_vertical == '⏳ Parameter Tuning':
        st.write('Will be implemented shortly using Optuna')

    # with main_tab3:
    if navigation_vertical == '🚀 Run Model':
        st.subheader('Prepare training and testing data')
        col_list = list(col_names)
        default_label_index = len(col_names)-1
        ss['split'] = ss.get('split', False)
        ss['feature_cols'] = ss.get('feature_cols', col_list)
        ss['label_col'] = ss.get('label_col', None)
        ss['label_index'] = ss.get('label_index', default_label_index)
        ss['seed'] = ss.get('seed', 1024)
        ss['test_size'] = ss.get('test_size', 0.33)

        with st.form("split_data_form"):
            # feature_cols, label_index, seed, test_size = split_data()
            feature_cols = st.multiselect('Exclude features not used in the model by cliking the X', col_names, ss['feature_cols'])

            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                label_col = st.selectbox('Select the lable variable', col_names, index=ss['label_index'])
                label_index = col_list.index(label_col)
            with p_col2:
                seed = st.number_input('seed', value=ss['seed'], help='The random seed help you reproduce the dataset later.')
            with p_col3:
                test_size = st.number_input('test_size', min_value=0.0, max_value=1.0, value=ss['test_size'])

            ss['X_train'] = ss.get('X_train', None)
            ss['X_test'] = ss.get('X_test', None)
            ss['y_train'] = ss.get('y_train', None)
            ss['y_test'] = ss.get('y_test', None)

            split_button = st.form_submit_button("Split the data")
            if split_button:
                ss['split'] = True
                ss['feature_cols'] = feature_cols
                ss['label_col'] = label_col
                ss['label_index'] = label_index
                ss['seed'] = seed
                ss['test_size'] = test_size

                X = df[ss['feature_cols']]
                y = df[ss['label_col']]
                ss['X_train'], ss['X_test'], ss['y_train'], ss['y_test'] = train_test_split(X, y, test_size=ss['test_size'],
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
        # with st.expander('Click to fold/unfold', expanded=True):
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
        ss['xgb'] = ss.get('xgb', None)
        ss['run_model'] = ss.get('run_model', False)
        ss['model_metrics'] = ss.get('model_metrics', None)
        ss['fp_r'] = ss.get('fp_r', None)
        ss['tp_r'] = ss.get('tp_r', None)

        # run_model = st.button('Run XGBoost 🚀')
        if st.button('Run XGBoost 🚀'):
            ss['run_model'] = True
            st.write(ss['X_train'].head())
            X_train, X_test, y_train, y_test = ss['X_train'], ss['X_test'], ss['y_train'], ss['y_test']

            @st.experimental_singleton
            def run_xgb():
                xgb = XGBClassifier(objective="binary:logistic", eval_metric="auc", use_label_encoder=False)
                xgb.set_params(**params)
                xgb.fit(X_train, y_train)
                return xgb

            ss['xgb'] = run_xgb()

            model_bar = st.progress(0)

            for percent_complete in range(100):
                time.sleep(0.1)
                model_bar.progress(percent_complete + 1)

            st.success('Model runs successfully!')

            model_metrics = {}
            model_metrics['precision'], model_metrics['recall'], model_metrics['f1'], model_metrics['accuracy'], roc = get_metrics(ss['xgb'], X_test, y_test)
            fp_r, tp_r, thresholds = roc
            model_metrics['auc_score'] = metrics.auc(fp_r, tp_r)
            ss['model_metrics'] = model_metrics
            ss['fp_r'], ss['tp_r'] = fp_r, tp_r
            model_datetime = str(datetime.now())
            model_date = model_datetime[:10]
            model_time = model_datetime[11:19]

            if authentication_status:
                db.insert_model(username, ss['filename'], model_date, model_time,
                                ss['feature_cols'], ss['label_col'], params, model_metrics)
                st.success('The model results have been saved!')

        if ss['run_model']:
            # _, X_test, _, y_test = ss['X_train'], ss['X_test'], ss['y_train'], ss['y_test']
            st.write(ss['model_metrics'])
            model_metrics = ss['model_metrics']
            fp_r, tp_r = ss['fp_r'], ss['tp_r']
            st.subheader('Check the model results')
            tab1, tab2, tab3 = st.tabs(["🔢 Metrics", "ROC Curve", "🎯 Feature Importance Chart"])

            with tab1:
                m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
                m_col1.metric("AUC", round(model_metrics['auc_score'], 3))
                m_col2.metric("Precision", round(model_metrics['precision'], 3))
                m_col3.metric("Recall", round(model_metrics['recall'], 3))
                m_col4.metric("F1 Score", round(model_metrics['f1'], 3))
                m_col5.metric("Accuracy", round(model_metrics['accuracy'], 3))

            with tab2:
                fig = plt.figure(figsize=(8, 6))
                plt.plot(fp_r, tp_r, label="AUC = %.3f" % model_metrics['auc_score'])
                plt.plot([0, 1], [0, 1], "r--")
                plt.ylabel("TP rate")
                plt.xlabel("FP rate")
                plt.legend(loc=4)
                plt.title("ROC Curve")
                st.pyplot(fig)

            with tab3:
                fig, ax = plt.subplots(figsize=(6, 9))
                plot_importance(ss['xgb'], max_num_features=50, height=0.8, ax=ax)
                st.pyplot(fig)


    # with main_tab4:
    if navigation_vertical == '⚡ Modeling History':
        if authentication_status:
            models = db.fetch_all_models(username)
            # st.write(models)
            # st.json(models, expanded=False)
            # st.json(models, expanded=True)
            model_df = pd.DataFrame(models)
            # model_df.drop([['_id', 'username']], axis=1, inplace=True)
            del model_df['_id']
            del model_df['username']
            st.dataframe(model_df)
            # AgGrid(model_df)
            # m_df_t = model_df.T
            # AgGrid(m_df_t)
            # model_list = []
        else:
            st.warning('You need to log in to retrieve previous models', icon="⚠️")







