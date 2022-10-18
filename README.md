## Building a web app that runs the XGBoost model without coding using Streamlit and MongoDB
 
This web app is built using streamlit and MongoDB. It allows the user to run the XGBoost models without writing a single line of code. For the logged-in user, the models they run will be saved in the database and can be retrieved later. After the model runs successfully, the user can download the picked model for later use. They can also download a json file documenting the critical model info.

 The app is hosted using streamlit share, and it can be accessed at  [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ifengzhao-no-code-xgboost-app-piv9vi.streamlitapp.com).
 
 **The web app has four tabs on the left sidebar**:
 - 📊 **Data Exploration**: users can upload the data or use the example dataset; they can also check the data they uploaded and check the basic statistics of the numeric variables. I will add data visualization features later on.
 - ⏳ **Parameter Tuning**: users can do hyperparameter tuning here. I will implemeted it in the next couple of days.
 - 🚀 **Run Model**: users can specify which features to be included in the model, the lable variable, and how to split the data into training and testing datasets. They can also specify values for a few important parameters of the XGBoost model. After the model runs successfully, the model will be automatically saved into the database for later retrieval. The user can download the picked model for later use. They can also download a json file documenting the critical model info. The user can also check the model results in different tabs.
 - ⚡ **Modeling History**: In this tab, the user can retieve all previous models. They can also check a specific model using the date and time selectboxes. The picked model of the selected can also be downloaded.

**To-do list**
- support univariate & bivariate data vaisualization
- implement hyperparameter tuning
- support XGBoost Regression

 
### Video Demo
[![YouTube Video](https://i9.ytimg.com/vi/UlyUIFzEMhk/mq1.jpg?sqp=CMzeuJoG&rs=AOn4CLB0tZbwpcs7evrfGxkUAuPCDPfCZQ)](https://www.youtube.com/watch?v=UlyUIFzEMhk&t=89s&ab_channel=FengZhao)
 
