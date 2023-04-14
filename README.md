![Alt text for the image](https://raw.githubusercontent.com/allabe93/employee_attrition_predictor/main/quitting.jpg)

# employee_attrition_predictor

This project aims to predict employee attrition based on various features such as age, salary, job satisfaction, work-life balance, years at the company, and others. The main objective is to help organizations identify potential employee attrition risks and take appropriate actions to retain valuable employees.

To achieve this goal, and after the work done in the Jupiter Notebook, we created a Streamlit app which predicts the probability of employee attrition using a logistic regression model that was trained on preprocessed data. The app starts by importing necessary libraries and loading the files (data, scaler, and model) and sets up the structure of the web page with headers, an image, and text. It then generates input fields for each feature using the generate_input() function and the st.selectbox() function for the overtime feature. The user inputs are collected and preprocessed (e.g., encoding the overtime feature and scaling the input data) before being fed into the saved logistic regression model to generate a prediction. The prediction is then displayed as a probability, with different colors depending on the risk level of employee attrition. The "magic" happens when the user's input data is gathered, preprocessed, and the trained model is used to make a prediction. Once the prediction is achieved, it can be displayed it to the user with appropriate formatting

## Overview
The project consists of several components:
1.	Data Analysis and Feature Engineering: exploratory data analysis was performed on the IBM HR Analytics Employee Attrition & Performance dataset from Kaggle, and features were selected for better model performance.
2.	Machine Learning Model: Logistic Regression was chosen as the best performing model in terms of recall for the minority class (attrition). SMOTE was used to balance the dataset, and hyperparameter tuning was applied to improve the model further.
3.	Web Application: a Streamlit web application was developed to provide an interactive user interface for inputting employee data and predicting the probability of attrition.
4.	Tableau visualizations and a dashboard: included to further explore the relationship between features and the target variable (the visualizations in Python focus solely on univariate distribution of numerical and categorical features). Therefore, for more advanced and interactive visualizations that focus on the relationships between features and the target variable, 'attrition', we include some work in Tableau.

## Project Structure
•	data/: contains the raw dataset used for the project and the tranformed versions of it.

•	pickle_files/: contains the saved scaler and the Logistic Regression model files.

•	employee_attrition_predictor_app.py: the Streamlit web application.

•	employee_attrition_predictor.ipynb: the Jupyter Notebook containing mainly the data analysis, some necessary cleaning, feature selection, the machine learning model and the refinement of it.

•	employee_attrition.twbx: the Tableau file.

• presentation.pptx: some slides to present the project. 

• quitting.jpg: an image referring to attrition.

## Key Findings
1.	Logistic Regression was chosen as the best model due to its higher recall for the minority class (attrition), which is important for this project's objective of predicting employee attrition.
2.	The use of SMOTE (oversampling technique) and hyperparameter tuning resulted in improved model performance.
3.	The analysis revealed key factors such as overtime and compensation and benefits as significant contributors to employee attrition. Additionally, there is an association between job satisfaction and employee turnover. By addressing these areas, organizations can focus their efforts on enhancing employee satisfaction and reducing turnover.

## Libraries Used
•	Numpy: https://numpy.org/doc/stable/

•	Pandas: https://pandas.pydata.org/pandas-docs/stable/index.html

•	Seaborn: https://seaborn.pydata.org/

•	Matplotlib: https://matplotlib.org/stable/contents.html

•	Scikit-learn: https://scikit-learn.org/stable/user_guide.html

•	Imbalanced-learn: https://imbalanced-learn.org/stable/

•	Streamlit: https://docs.streamlit.io/en/stable/
