import streamlit as st
import pandas as pd
import numpy as np
import pickle


# Loading the data
transf_data = pd.read_csv('data/transformed_data_final.csv')


# Loading the saved scaler and model
with open("pickle_files/standard_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("pickle_files/best_logistic.pkl", "rb") as f:
    model = pickle.load(f)


st.write("<h2 style='text-align: center;'>HR Predictive Solutions: Attrition</h2>", unsafe_allow_html=True)


# Adding an image
image_path = "quitting.jpg"
st.image(image_path, width=None, use_column_width="always")


# Adding some text
st.markdown("<strong>Attrition</strong> is the departure of employees from the organization due to different reasons. This increases the turnover of the company and its cost is really high. It is estimated that losing an employee can cost a company 1.5-2 times the employee's salary -depending on the individual's seniority-, so reducing this cost to the minimum possible is vital to contribute to the economic efficiency of the organization from the point of view of personnel costs.", unsafe_allow_html=True)


st.write("Certainly there are personal, psychological, family, social and even force majeure and other factors that are beyond the company's control that can lead to employee attrition, <strong>but can you imagine having a solution that allows you to identify the likelihood of employee attrition based on variables that the company can control?</strong>", unsafe_allow_html=True)


st.write("This is precisely what we present to you here, a tool in which you enter data about the employee so that we can inform you of the probability of employee attrition. What you then do with the information you find here is up to you.") 


st.write("This solution is:")


# Solution title
st.write('<h1 style="text-align: center;">Employee Attrition Predictor</h1>', unsafe_allow_html=True)


# Adding a "call to action"
st.write("Please fill in all the fields below to see the probability of employee attrition at the end.")


# Dictionary to map the column names to their readable versions
readable_column_names = {
    "age": "Age",
    "dailyrate": "Daily Rate",
    "distancefromhome": "Distance from Home",
    "environmentsatisfaction": "Environment Satisfaction",
    "hourlyrate": "Hourly Rate",
    "jobinvolvement": "Job Involvement",
    "jobsatisfaction": "Job Satisfaction",
    "monthlyincome": "Monthly Income",
    "monthlyrate": "Monthly Rate",
    "numcompaniesworked": "Number of Companies Worked at",
    "overtime": "Overtime",
    "percentsalaryhike": "Percent Salary Hike",
    "relationshipsatisfaction": "Relationship Satisfaction",
    "stockoptionlevel": "Stock Option Level",
    "totalworkingyears": "Total Working Years",
    "trainingtimeslastyear": "Number of  Trainings last Year",
    "worklifebalance": "Work-Life Balance",
    "yearsatcompany": "Number of  Years at the Company",
    "yearsincurrentrole": "Number of  Years in the current Role",
    "yearssincelastpromotion": "Number of  Years since last Promotion",
    "yearswithcurrmanager": "Number of  Years with the current Manager",
}


# Function to generate input fields
def generate_input(column_name, df):
    min_value = df[column_name].min()
    max_value = df[column_name].max()
    mean_value = df[column_name].mean()
    input_value = st.number_input(
        readable_column_names[column_name],
        min_value=min_value,
        max_value=max_value,
        value=int(mean_value),
    )
    return input_value


# Creating input fields for each feature
age = generate_input("age", transf_data)
hourly_rate = generate_input("hourlyrate", transf_data)
daily_rate = generate_input("dailyrate", transf_data)
monthly_rate = generate_input("monthlyrate", transf_data)
monthly_income = generate_input("monthlyincome", transf_data)
percent_salary_hike = generate_input("percentsalaryhike", transf_data)
job_involvement = generate_input("jobinvolvement", transf_data)
job_satisfaction = generate_input("jobsatisfaction", transf_data)
environment_satisfaction = generate_input("environmentsatisfaction", transf_data)
relationship_satisfaction = generate_input("relationshipsatisfaction", transf_data)
overtime = st.selectbox(readable_column_names["overtime"], options=["No", "Yes"])
work_life_balance = generate_input("worklifebalance", transf_data)
distance_from_home = generate_input("distancefromhome", transf_data)
num_companies_worked = generate_input("numcompaniesworked", transf_data)
total_working_years = generate_input("totalworkingyears", transf_data)
training_times_last_year = generate_input("trainingtimeslastyear", transf_data)
years_in_current_role = generate_input("yearsincurrentrole", transf_data)
years_with_curr_manager = generate_input("yearswithcurrmanager", transf_data)
years_since_last_promotion = generate_input("yearssincelastpromotion", transf_data)
years_at_company = generate_input("yearsatcompany", transf_data)
stock_option_level = generate_input("stockoptionlevel", transf_data)


# Encoding the 'overtime' feature
if overtime == "Yes":
    overtime_encoded = 0
else:
    overtime_encoded = 1


# Creating a NumPy array with input values
input_data = np.array([age, daily_rate, distance_from_home, environment_satisfaction, hourly_rate, job_involvement, job_satisfaction, monthly_income, monthly_rate, num_companies_worked, overtime_encoded, percent_salary_hike, relationship_satisfaction, stock_option_level, total_working_years, training_times_last_year, work_life_balance, years_at_company, years_in_current_role, years_since_last_promotion, years_with_curr_manager]).reshape(1, -1)


# Scaling the input data and making a prediction
input_data_scaled = scaler.transform(input_data)
prediction = model.predict_proba(input_data_scaled)


# Displaying the prediction
probability = prediction[0][1] * 100

if probability <= 33:
    color = "green"
elif 33 < probability <= 66:
    color = "#FFA500"  # Orange color
else:
    color = "red"

st.markdown(f"<h2 style='color: {color}; font-weight: bold;'>The probability of attrition is: {probability:.2f}%</h2>", unsafe_allow_html=True)


st.write("Now that you know the probability of the employee leaving the company, and especially if he/she is in the red zone (probability higher than 66%), you can take the appropriate actions to avoid it.")
st.text("It lies in your hands!")