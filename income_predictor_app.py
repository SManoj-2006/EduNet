import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Advanced Income Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def get_data():
    """Loads, cleans, and caches the data for UI and visualizations."""
    df = pd.read_csv('adult 3.csv')
    df.columns = df.columns.str.strip()
    df.replace('?', 'Others', inplace=True)
    df['income_cat'] = df['income'].apply(lambda x: '>50K' if '>50K' in x else '<=50K')
    return df

@st.cache_resource
def get_model():
    """
    Loads the model from disk, or trains and saves it if it doesn't exist.
    Returns the model and the feature columns it was trained on.
    """

    if not os.path.exists('random_forest_model.pkl'):
        df_train = pd.read_csv('adult 3.csv')
        df_train.columns = df_train.columns.str.strip()
        df_train.replace('?', 'Others', inplace=True)
        

        df_train['income_num'] = df_train['income'].apply(lambda x: 1 if '>50K' in x else 0)
        target = 'income_num'

        features_to_drop = ['income_num', 'income', 'education', 'fnlwgt']
        features = [col for col in df_train.columns if col not in features_to_drop]
        X = df_train[features]
        y = df_train[target]

        X_encoded = pd.get_dummies(X, drop_first=True)
        model_columns = X_encoded.columns
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_encoded, y)


        joblib.dump(model, 'random_forest_model.pkl')
        joblib.dump(model_columns, 'model_columns.pkl')
    
    model = joblib.load('random_forest_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, model_columns


df = get_data()
model, model_columns = get_model()





st.title("👨‍💼 Advanced Income Prediction Dashboard")
st.markdown("An interactive tool to predict income levels and explore the underlying data.")



tab1, tab2 = st.tabs(["**🚀 Income Predictor**", "**📊 Project Insights**"])



with tab1:
    st.header("Employee Salary Prediction")

    if 'prediction' not in st.session_state:
        st.session_state.prediction = None
        st.session_state.confidence = None
        st.session_state.probabilities = None
        st.session_state.last_input = None

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("Input Features")
        st.markdown("Adjust the sliders and dropdowns to match an individual's profile.")


        age = st.slider("Age", 17, 90, 38)
        workclass = st.selectbox("Work Class", df['workclass'].unique())
        education_num = st.slider("Educational Number", 1, 16, 10)
        marital_status = st.selectbox("Marital Status", df['marital-status'].unique())
        occupation = st.selectbox("Occupation", df['occupation'].unique())
        relationship = st.selectbox("Relationship", df['relationship'].unique())
        race = st.selectbox("Race", df['race'].unique())
        gender = st.selectbox("Gender", df['gender'].unique())
        capital_gain = st.number_input("Capital Gain", 0, 100000, 0)
        capital_loss = st.number_input("Capital Loss", 0, 5000, 0)
        hours_per_week = st.slider("Hours per Week", 1, 99, 40)
        native_country = st.selectbox("Native Country", df['native-country'].unique())

    with col2:
        st.subheader("Prediction Result")
        st.markdown("Click the button below to see the model's prediction.")

        if st.button("Predict Income", use_container_width=True, type="primary"):

            input_data = {
                'age': [age], 'workclass': [workclass], 'educational-num': [education_num],
                'marital-status': [marital_status], 'occupation': [occupation],
                'relationship': [relationship], 'race': [race], 'gender': [gender],
                'capital-gain': [capital_gain], 'capital-loss': [capital_loss],
                'hours-per-week': [hours_per_week], 'native-country': [native_country]
            }
            input_df = pd.DataFrame(input_data)
            st.session_state.last_input = input_df # Store for debugging
            
            input_encoded = pd.get_dummies(input_df).reindex(columns=model_columns, fill_value=0)

            prediction_code = model.predict(input_encoded)[0]
            prediction_proba = model.predict_proba(input_encoded)[0]
            

            st.session_state.prediction = prediction_code
            st.session_state.confidence = prediction_proba[prediction_code]
            st.session_state.probabilities = prediction_proba


        if st.session_state.prediction is not None:
            if st.session_state.prediction == 1:
                st.success(f"**Predicted Income: >$50K**", icon="✅")
            else:
                st.error(f"**Predicted Income: <=$50K**", icon="❌")
            
            st.metric(label="Confidence", value=f"{st.session_state.confidence:.2%}")
            
            with st.expander("Show Prediction Details"):
                st.write("Probabilities:")
                st.write(f"Probability of earning **<=$50K**: {st.session_state.probabilities[0]:.2%}")
                st.write(f"Probability of earning **>$50K**: {st.session_state.probabilities[1]:.2%}")
            
            with st.expander("Show Input Data Used for This Prediction"):
                st.dataframe(st.session_state.last_input)



with tab2:
    st.header("Data and Model Insights")
    st.markdown("Visualizations to help understand the dataset and model behavior.")

    plot_col1, plot_col2 = st.columns(2)

    with plot_col1:

        st.subheader("Income by Education Level")
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        sns.countplot(data=df, y='education', hue='income_cat', ax=ax1, order=df['education'].value_counts().index)
        ax1.set_title('Income Distribution by Education Level')
        ax1.set_xlabel('Count')
        ax1.set_ylabel('Education Level')
        ax1.legend(title='Income')
        st.pyplot(fig1)

    with plot_col2:

        st.subheader("Age Distribution by Income")
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        sns.histplot(data=df, x='age', hue='income_cat', multiple='stack', bins=30, ax=ax2, palette='viridis')
        ax2.set_title('Age Distribution by Income Level')
        ax2.set_xlabel('Age')
        ax2.set_ylabel('Count')
        st.pyplot(fig2)

    st.divider()


    plot_col3, plot_col4 = st.columns(2)

    with plot_col3:

        st.subheader("Income by Work Class")
        fig3, ax3 = plt.subplots(figsize=(10, 8))
        sns.countplot(data=df, y='workclass', hue='income_cat', ax=ax3, order=df['workclass'].value_counts().index, palette='magma')
        ax3.set_title('Income Distribution by Work Class')
        ax3.set_xlabel('Count')
        ax3.set_ylabel('Work Class')
        ax3.legend(title='Income')
        st.pyplot(fig3)

    with plot_col4:

        st.subheader("Hours per Week vs. Income")
        fig4, ax4 = plt.subplots(figsize=(10, 8))
        sns.boxplot(data=df, x='income_cat', y='hours-per-week', ax=ax4, palette='coolwarm')
        ax4.set_title('Distribution of Hours per Week by Income')
        ax4.set_xlabel('Income Bracket')
        ax4.set_ylabel('Hours per Week')
        st.pyplot(fig4)

    st.divider()


    st.subheader("Model Feature Importance")
    st.markdown("This chart shows which features the model found most influential for making predictions.")
    feature_importances = pd.Series(model.feature_importances_, index=model_columns)
    top_features = feature_importances.nlargest(15)

    fig5, ax5 = plt.subplots(figsize=(12, 8))
    sns.barplot(x=top_features.values, y=top_features.index, ax=ax5, palette='plasma')
    ax5.set_title('Top 15 Most Important Features')
    ax5.set_xlabel('Importance Score')
    ax5.set_ylabel('Features')
    st.pyplot(fig5)
