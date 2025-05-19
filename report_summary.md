# Bank Marketing Analysis Report Summary

## Key Findings:

1.  **Model Performance**:
    *   We compared three machine learning models: Random Forest, Logistic Regression, and Gradient Boosting.
    *   Gradient Boosting generally performed the best across metrics, followed closely by Random Forest.
    *   All models showed good discriminative ability with AUC-ROC scores above 0.90.

2.  **Most Important Features**:
    *   Economic indicators (like euribor3m, nr.employed, emp.var.rate) were consistently among the most important features.
    *   Contact type and month of contact also had significant impacts on subscription likelihood.
    *   Previous campaign outcome was highly predictive of current campaign success.

3.  **Target Imbalance**:
    *   The dataset is imbalanced, with a much smaller number of 'yes' responses compared to 'no' responses.
    *   Despite this imbalance, our models performed reasonably well in predicting both classes.

4.  **Economic Context**:
    *   The social and economic attributes added significant value to the prediction model, confirming findings from previous studies.
    *   Market conditions strongly influence a client's decision to subscribe to a term deposit.

## Recommendations for the Bank's Marketing Strategy:

1.  **Timing**: Focus campaigns during periods with favorable economic indicators, particularly when euribor rates are within specific ranges.
2.  **Targeting**: Prioritize customer segments identified as more likely to subscribe based on feature importance analysis.
3.  **Contact Method**: Optimize the contact approach based on the findings about contact types and their effectiveness.
4.  **Follow-up Strategy**: Pay special attention to clients with successful previous campaign outcomes as they have a higher likelihood of subscribing again.
5.  **Model Deployment**: Implement the Gradient Boosting model in the marketing decision process to prioritize potential clients by their likelihood to subscribe, as demonstrated with the Streamlit app.

By leveraging both the insights from our analysis and the deployed model application, the bank can significantly improve the efficiency of its marketing campaigns and increase the success rate of term deposit subscriptions.

Streamlit link
https://ada442-termproject-hdnnmaz3zupemanhtwd9sa.streamlit.app/

Adem Aşkan
Mehmet Bektaş
Koray Başkoca
