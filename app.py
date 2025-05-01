import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model
model = joblib.load('bank_marketing_model.joblib')

# Set page title and description
st.title('Bank Marketing Prediction App')
st.write("""
This app predicts whether a client will subscribe to a term deposit based on input features.
Enter the client's information below and click 'Predict' to get a prediction.
""")

# Create input form
st.header('Client Information')

# Create two columns for inputs
col1, col2 = st.columns(2)

with col1:
    age = st.slider('Age', 18, 100, 40)
    job = st.selectbox('Job', ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
                            'retired', 'self-employed', 'services', 'student', 'technician', 
                            'unemployed', 'unknown'])
    marital = st.selectbox('Marital Status', ['divorced', 'married', 'single', 'unknown'])
    education = st.selectbox('Education', ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 
                                        'illiterate', 'professional.course', 'university.degree', 
                                        'unknown'])
    default = st.selectbox('Has Credit in Default?', ['no', 'yes', 'unknown'])
    housing = st.selectbox('Has Housing Loan?', ['no', 'yes', 'unknown'])
    loan = st.selectbox('Has Personal Loan?', ['no', 'yes', 'unknown'])

with col2:
    contact = st.selectbox('Contact Communication Type', ['cellular', 'telephone'])
    month = st.selectbox('Last Contact Month', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                                            'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    day_of_week = st.selectbox('Last Contact Day of Week', ['mon', 'tue', 'wed', 'thu', 'fri'])
    duration = st.slider('Last Contact Duration (seconds)', 0, 5000, 250)
    campaign = st.slider('Number of Contacts During Campaign', 1, 20, 2)
    pdays = st.slider('Days Since Last Contact', 0, 999, 999, help="999 means client was not previously contacted")
    previous = st.slider('Number of Contacts Before Campaign', 0, 10, 0)
    poutcome = st.selectbox('Outcome of Previous Campaign', ['failure', 'nonexistent', 'success'])

# Economic indicators (average values as defaults)
st.header('Economic Indicators')
col3, col4 = st.columns(2)

with col3:
    emp_var_rate = st.slider('Employment Variation Rate', -3.0, 2.0, 0.0, step=0.1)
    cons_price_idx = st.slider('Consumer Price Index', 90.0, 95.0, 93.5, step=0.1)
    cons_conf_idx = st.slider('Consumer Confidence Index', -50.0, -25.0, -40.0, step=0.5)

with col4:
    euribor3m = st.slider('Euribor 3 Month Rate', 0.5, 5.0, 3.0, step=0.1)
    nr_employed = st.slider('Number of Employees', 4900.0, 5300.0, 5100.0, step=10.0)

# Create a DataFrame with the input features
def create_input_df():
    data = {'age': age, 
            'job': job, 
            'marital': marital, 
            'education': education, 
            'default': default, 
            'housing': housing, 
            'loan': loan, 
            'contact': contact, 
            'month': month, 
            'day_of_week': day_of_week, 
            'duration': duration,
            'campaign': campaign, 
            'pdays': pdays,
            'previous': previous, 
            'poutcome': poutcome, 
            'emp.var.rate': emp_var_rate, 
            'cons.price.idx': cons_price_idx, 
            'cons.conf.idx': cons_conf_idx, 
            'euribor3m': euribor3m, 
            'nr.employed': nr_employed}

    return pd.DataFrame(data, index=[0])

# When the user clicks the Predict button
if st.button('Predict'):
    # Create the input DataFrame
    input_df = create_input_df()

    # Make prediction
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]

    # Display result
    st.header('Prediction Result')

    if prediction == 1:
        st.success(f'The client is likely to subscribe to a term deposit! (Probability: {prediction_proba:.2%})')
    else:
        st.error(f'The client is not likely to subscribe to a term deposit. (Probability: {prediction_proba:.2%})')

    # Display feature importance if available
    if hasattr(model[-1], 'feature_importances_'):
        st.subheader('Feature Importance for this Prediction')

        # Get feature names after preprocessing
        preprocessor = model[0]
        if hasattr(preprocessor, 'transformers_'):
            numeric_features = preprocessor.transformers_[0][2]
            categorical_features = preprocessor.transformers_[1][2]

            # Get categorical feature names after one-hot encoding
            categorical_transformer = preprocessor.transformers_[1][1]
            if hasattr(categorical_transformer, 'get_feature_names_out'):
                cat_feature_names = categorical_transformer.get_feature_names_out(categorical_features)
                feature_names = np.concatenate([numeric_features, cat_feature_names])

                # Get feature importances
                importances = model[-1].feature_importances_

                # Create DataFrame for feature importances
                feature_importance_df = pd.DataFrame({
                    'Feature': feature_names, 
                    'Importance': importances
                }).sort_values(by='Importance', ascending=False).head(10)

                # Display feature importances
                st.bar_chart(feature_importance_df.set_index('Feature'))
            else:
                st.write('Feature importance visualization not available for this model.')
        else:
            st.write('Feature importance visualization not available for this model.')
    else:
        st.write('Feature importance visualization not available for this model.')

# Instructions for running the app
st.sidebar.header('About')
st.sidebar.info("""
This app predicts whether a client will subscribe to a term deposit based on their demographic information and economic indicators.

The model was trained on data from a Portuguese banking institution's direct marketing campaigns.
""")

st.sidebar.header('Instructions')
st.sidebar.info("""
1. Enter the client's personal information
2. Set the economic indicators
3. Click 'Predict' to get the result
""")
