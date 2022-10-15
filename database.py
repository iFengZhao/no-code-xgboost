from pymongo import MongoClient
import streamlit as st


connection_string = f"mongodb+srv://{st.secrets.DB_USERNAME}:{st.secrets.DB_PWD}@hobby.i32puaj.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(connection_string)
xgboost_db = client.xgboost

user_collection = xgboost_db.users
model_collection = xgboost_db.models


def insert_user(username, name, password, email):
    """Insert the info of a registered user into the database"""
    model_collection.insert_one({'_id': username, 'name': name,
                                 'password': password, 'email': email})


def fetch_all_users():
    """
    Retrive all users from the database
    """
    results = user_collection.find({})
    users = [user for user in results]
    return users


def insert_model(username, filename, model_date, model_time,
                 used_features, label_name, params, metrics):

    """Insert the model info into the database"""

    model_collection.insert_one({'username': username, 'filename': filename,
                                 'model_date': model_date, 'model_time': model_time,
                                 'used_features': used_features, 'label_name': label_name,
                                 'params': params, 'metrics': metrics})


def fetch_all_models(username):
    """Retrieve a user's all models"""
    results = model_collection.find({'username': username})
    models = [model for model in results]
    return models

