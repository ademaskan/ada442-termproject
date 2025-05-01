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
    
    # Job options with more descriptive labels
    job_options = {
        'admin.': 'Administrative',
        'blue-collar': 'Blue Collar Worker',
        'entrepreneur': 'Entrepreneur',
        'housemaid': 'Housemaid/Cleaning',
        'management': 'Management',
        'retired': 'Retired',
        'self-employed': 'Self-Employed',
        'services': 'Services Sector',
        'student': 'Student',
        'technician': 'Technical Professional',
        'unemployed': 'Unemployed',
        'unknown': 'Not Specified'
    }
    job_display = list(job_options.keys())
    job_labels = list(job_options.values())
    job = st.selectbox('Occupation', options=job_display, format_func=lambda x: job_options[x])
    
    # Marital status with better labels
    marital_options = {
        'divorced': 'Divorced/Separated',
        'married': 'Married/Cohabiting',
        'single': 'Single',
        'unknown': 'Not Specified'
    }
    marital = st.selectbox('Marital Status', options=list(marital_options.keys()), 
                          format_func=lambda x: marital_options[x])
    
    # Education with clearer descriptions
    education_options = {
        'basic.4y': 'Basic Education (4 years)',
        'basic.6y': 'Basic Education (6 years)',
        'basic.9y': 'Basic Education (9 years)',
        'high.school': 'High School Diploma',
        'illiterate': 'No Formal Education',
        'professional.course': 'Professional Certificate',
        'university.degree': 'University Degree',
        'unknown': 'Not Specified'
    }
    education = st.selectbox('Education Level', options=list(education_options.keys()), 
                            format_func=lambda x: education_options[x])
    
    # Yes/No options with clearer labels
    yes_no_options = {
        'no': 'No',
        'yes': 'Yes',
        'unknown': 'Not Specified'
    }
    default = st.selectbox('Credit Default History', options=list(yes_no_options.keys()), 
                          format_func=lambda x: yes_no_options[x])
    housing = st.selectbox('Housing Loan', options=list(yes_no_options.keys()), 
                          format_func=lambda x: yes_no_options[x])
    loan = st.selectbox('Personal Loan', options=list(yes_no_options.keys()), 
                       format_func=lambda x: yes_no_options[x])

with col2:
    # Contact method with better descriptions
    contact_options = {
        'cellular': 'Mobile Phone',
        'telephone': 'Landline Phone'
    }
    contact = st.selectbox('Contact Method', options=list(contact_options.keys()), 
                          format_func=lambda x: contact_options[x])
    
    # Full month names instead of abbreviations
    month_options = {
        'jan': 'January',
        'feb': 'February',
        'mar': 'March',
        'apr': 'April',
        'may': 'May',
        'jun': 'June', 
        'jul': 'July',
        'aug': 'August',
        'sep': 'September',
        'oct': 'October',
        'nov': 'November',
        'dec': 'December'
    }
    month = st.selectbox('Month of Last Contact', options=list(month_options.keys()), 
                        format_func=lambda x: month_options[x])
    
    # Full day names instead of abbreviations
    day_options = {
        'mon': 'Monday',
        'tue': 'Tuesday',
        'wed': 'Wednesday',
        'thu': 'Thursday',
        'fri': 'Friday'
    }
    day_of_week = st.selectbox('Day of Last Contact', options=list(day_options.keys()), 
                              format_func=lambda x: day_options[x])
    
    duration = st.slider('Last Call Duration (seconds)', 0, 5000, 250)
    campaign = st.slider('Number of Contacts in Current Campaign', 1, 20, 2)
    pdays = st.slider('Days Since Previous Contact', 0, 999, 999, help="999 means client was not previously contacted")
    previous = st.slider('Previous Campaign Contacts', 0, 10, 0)
    
    # More descriptive outcome options
    outcome_options = {
        'failure': 'Unsuccessful',
        'nonexistent': 'No Previous Campaign',
        'success': 'Successful'
    }
    poutcome = st.selectbox('Previous Campaign Outcome', options=list(outcome_options.keys()),
                           format_func=lambda x: outcome_options[x])

# Economic indicators (average values as defaults)
st.header('Economic Indicators')
col3, col4 = st.columns(2)

with col3:
    emp_var_rate = st.slider('Employment Variation Rate (quarterly)', -3.0, 2.0, 0.0, step=0.1, 
                            help="Quarterly indicator of employment change")
    cons_price_idx = st.slider('Consumer Price Index (monthly)', 90.0, 95.0, 93.5, step=0.1,
                              help="Monthly indicator of inflation")
    cons_conf_idx = st.slider('Consumer Confidence Index (monthly)', -50.0, -25.0, -40.0, step=0.5,
                             help="Monthly indicator of consumer confidence")

with col4:
    euribor3m = st.slider('Euribor 3-Month Rate (%)', 0.5, 5.0, 3.0, step=0.1,
                         help="Daily indicator of Euro interbank interest rate")
    nr_employed = st.slider('Number of Employees (quarterly)', 4900.0, 5300.0, 5100.0, step=10.0,
                          help="Quarterly indicator of total workforce")

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
