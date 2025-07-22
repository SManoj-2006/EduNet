Employee Salary Prediction
An interactive web application built with Streamlit to predict whether an individual's annual income is likely to be more or less than $50,000 based on US census data. This project demonstrates a complete machine learning workflow from data cleaning and model training to deployment as a user-friendly tool.

🚀 Live Demo & Screenshot
The application provides an interactive dashboard with a prediction tool and data visualizations.

✨ Features
Interactive Prediction Tool: Users can input demographic data through sliders and dropdowns to get an instant income prediction.

Prediction Confidence: Displays the model's confidence score for each prediction.

Data Insights Dashboard: A dedicated tab with interactive charts to explore the dataset.

Feature Importance: Visualizes which factors are most influential in the model's predictions.

Dynamic Model Training: The application automatically trains and saves the machine learning model on the first run.

📊 Dataset
This project uses the "Adult" dataset from the UCI Machine Learning Repository, which was extracted from the 1994 US Census database.

Source: UCI Machine Learning Repository

Features: Includes 14 attributes such as age, workclass, education, marital status, occupation, race, gender, and hours worked per week.

Target Variable: income (classified as <=50K or >50K).

🛠️ Tech Stack
Language: Python

Core Libraries:

Streamlit: For building the interactive web user interface.

Pandas: For data manipulation and cleaning.

Scikit-learn: For building and evaluating the Random Forest classification model.

Matplotlib & Seaborn: For creating data visualizations.

Joblib: For saving and loading the trained model.

⚙️ Setup and Installation
To run this project locally, please follow these steps:

1. Clone the repository:

git clone https://github.com/SManoj-2006/EduNet
cd EduNet

2. Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install the required dependencies:

pip install -r requirements.txt

(Note: You will need to create a requirements.txt file. See below.)

4. Create requirements.txt:
Create a file named requirements.txt in your project folder and add the following lines:

streamlit
pandas
scikit-learn
matplotlib
seaborn

5. Run the Streamlit application:

streamlit run app.py

(Assuming your main Python script is named app.py)

The application will open in your web browser at http://localhost:8501.

📂 File Structure
.
├── .gitignore          # Tells Git which files to ignore (e.g., model files)
├── app.py              # The main Streamlit application script
├── adult 3.csv         # The dataset file
├── requirements.txt    # List of Python dependencies
└── README.md           # This file

Important: The trained model file (random_forest_model.pkl) is not stored in the repository. The application will generate this file automatically the first time it is run.

📈 Model Performance
The prediction model is a Random Forest Classifier which achieved the following performance on the test set:

Overall Accuracy: ~85%

The model shows high precision and recall for the majority class (<=50K) and is a strong baseline for this classification task. Further details can be seen in the confusion matrix and feature importance charts in the app's "Project Insights" tab.

💡 Future Improvements
Hyperparameter Tuning: Optimize the Random Forest model using techniques like GridSearchCV to improve performance.

Experiment with Other Models: Test other classification algorithms like XGBoost or LightGBM.

Cloud Deployment: Deploy the Streamlit application to a cloud service like Streamlit Community Cloud or Heroku for public access.
