from sklearn.ensemble import StackingClassifier
import pickle
clf = pickle.load(open('../models/stacking_classifier_optimized.pkl', 'rb'))

import numpy as np
import pandas as pd
import json
X = pd.read_csv('../data/ibm_hr_analytics_cleaned.csv')
raw_map = json.load(open('../data/mappings.json', 'rb'))
map = {}
for feature in raw_map.keys():
    map.update({feature: {value: int(key) for key, value in raw_map[feature].items() if value is not None}})

import ipywidgets as widgets
from IPython.display import display
from ipywidgets import Layout

def DisplayApp():
    # Apply it to the model with prediction probability
    bar = widgets.FloatProgress(value=0.0, 
                                min=0.0, 
                                max=100.0, 
                                description='Probability:', 
                                bar_style='danger',
                                style={'bar_color': 'red',
                                       'description_width': '100px',
                                       'font_size': '20px'},
                                layout=Layout(width='70%', height='50px', margin='15px 10px 5px 0'))
    bar_label = widgets.Label(value='0% likely to leave the company', layout=Layout(margin='15px 0 0 0'))
    
    all_inputs = []
    
    # There is certainly a more object-oriented approach for creating all these inputs, 
    # but I think this approach is best for this specific project. If I were to approach
    # this again on a larger project, I would calculate/store the default values once 
    # then use a loop with a function call to create these.
    age_slider = widgets.IntSlider(
        description='Age', 
        value=round(X['Age'].median()), 
        min=X['Age'].min(), 
        max=X['Age'].max()
    )
    all_inputs.append(age_slider)
    
    business_travel_drop = widgets.Dropdown(
        description='BusinessTravel',
        options=map['BusinessTravel']
    )
    all_inputs.append(business_travel_drop)
    
    dailyrate_slider = widgets.IntSlider(
        description='DailyRate',
        value=round(X['DailyRate'].median()),
        min=X['DailyRate'].min(),
        max=X['DailyRate'].max()
    )
    all_inputs.append(dailyrate_slider)
    
    dept_drop = widgets.Dropdown(
        description='Department',
        options=map['Department']
    )
    all_inputs.append(dept_drop)
    
    dist_slider = widgets.IntSlider(
        description='DistanceFromHome',
        value=round(X['DistanceFromHome'].median()),
        min=X['DistanceFromHome'].min(),
        max=X['DistanceFromHome'].max()
    )
    all_inputs.append(dist_slider)
    
    edu_slider = widgets.IntSlider(
        description='Education',
        value=round(X['Education'].median()),
        min=X['Education'].min(),
        max=X['Education'].max()
    )
    all_inputs.append(edu_slider)
    
    edu_drop = widgets.Dropdown(
        description='EducationField',
        options=map['EducationField']
    )
    all_inputs.append(edu_drop)
    
    env_slider = widgets.IntSlider(
        description='EnvironmentSatisfaction',
        value=round(X['EnvironmentSatisfaction'].median()),
        min=X['EnvironmentSatisfaction'].min(),
        max=X['EnvironmentSatisfaction'].max()
    )
    all_inputs.append(env_slider)
    
    gender_drop = widgets.Dropdown(
        description='Gender',
        options=map['Gender']
    )
    all_inputs.append(gender_drop)
    
    hourly_rate_slider = widgets.IntSlider(
        description='HourlyRate',
        value=round(X['HourlyRate'].median()),
        min=X['HourlyRate'].min(),
        max=X['HourlyRate'].max()
    )
    all_inputs.append(hourly_rate_slider)
    
    job_involvement_slider = widgets.IntSlider(
        description='JobInvolvement',
        value=round(X['JobInvolvement'].median()),
        min=X['JobInvolvement'].min(),
        max=X['JobInvolvement'].max()
    )
    all_inputs.append(job_involvement_slider)
    
    job_level_slider = widgets.IntSlider(
        description='JobLevel',
        value=round(X['JobLevel'].median()),
        min=X['JobLevel'].min(),
        max=X['JobLevel'].max()
    )
    all_inputs.append(job_level_slider)
    
    job_role_drop = widgets.Dropdown(
        description='JobRole',
        options=map['JobRole']
    )
    all_inputs.append(job_role_drop)
    
    job_satisfaction_slider = widgets.IntSlider(
        description='JobSatisfaction',
        value=round(X['JobSatisfaction'].median()),
        min=X['JobSatisfaction'].min(),
        max=X['JobSatisfaction'].max()
    )
    all_inputs.append(job_satisfaction_slider)
    
    marital_status_drop =  widgets.Dropdown(
        description='MaritalStatus',
        options=map['MaritalStatus']
    )
    all_inputs.append(marital_status_drop)
    
    monthly_income_slider = widgets.IntSlider(
        description='MonthlyIncome',
        value=round(X['MonthlyIncome'].median()),
        min=X['MonthlyIncome'].min(),
        max=X['MonthlyIncome'].max()
    )
    all_inputs.append(monthly_income_slider)
    
    monthly_rate_slider = widgets.IntSlider(
        description='MonthlyRate',
        value=round(X['MonthlyRate'].median()),
        min=X['MonthlyRate'].min(),
        max=X['MonthlyRate'].max()
    )
    all_inputs.append(monthly_rate_slider)
    
    num_companies_worked_slider = widgets.IntSlider(
        description='NumCompaniesWorked',
        value=round(X['NumCompaniesWorked'].median()),
        min=X['NumCompaniesWorked'].min(),
        max=X['NumCompaniesWorked'].max()
    )
    all_inputs.append(num_companies_worked_slider)
    
    overtime_button = widgets.Checkbox(
        description='Overtime',
        value=False
    )
    all_inputs.append(overtime_button)
    
    percent_salary_hike_slider = widgets.IntSlider(
        description='PercentSalaryHike',
        value=round(X['PercentSalaryHike'].median()),
        min=X['PercentSalaryHike'].min(),
        max=X['PercentSalaryHike'].max()
    )
    all_inputs.append(percent_salary_hike_slider)
    
    performance_rating_slider = widgets.IntSlider(
        description='PerformanceRating',
        value=round(X['PerformanceRating'].median()),
        min=X['PerformanceRating'].min(),
        max=X['PerformanceRating'].max()
    )
    all_inputs.append(performance_rating_slider)
    
    relationship_satisfaction_slider = widgets.IntSlider(
        description='RelationshipSatisfaction',
        value=round(X['RelationshipSatisfaction'].median()),
        min=X['RelationshipSatisfaction'].min(),
        max=X['RelationshipSatisfaction'].max()
    )
    all_inputs.append(relationship_satisfaction_slider)
    
    stock_option_level_slider = widgets.IntSlider(
        description='StockOptionLevel',
        value=round(X['StockOptionLevel'].median()),
        min=X['StockOptionLevel'].min(),
        max=X['StockOptionLevel'].max()
    )
    all_inputs.append(stock_option_level_slider)
    
    total_working_years_slider = widgets.IntSlider(
        description='TotalWorkingYears',
        value=round(X['TotalWorkingYears'].median()),
        min=X['TotalWorkingYears'].min(),
        max=X['TotalWorkingYears'].max()
    )
    all_inputs.append(total_working_years_slider)
    
    training_times_last_year_slider = widgets.IntSlider(
        description='TrainingTimesLastYear',
        value=round(X['TrainingTimesLastYear'].median()),
        min=X['TrainingTimesLastYear'].min(),
        max=X['TrainingTimesLastYear'].max()
    )
    all_inputs.append(training_times_last_year_slider)
    
    work_life_balance_slider = widgets.IntSlider(
        description='WorkLifeBalance',
        value=round(X['WorkLifeBalance'].median()),
        min=X['WorkLifeBalance'].min(),
        max=X['WorkLifeBalance'].max()
    )
    all_inputs.append(work_life_balance_slider)
    
    years_at_company_slider = widgets.IntSlider(
        description='YearsAtCompany',
        value=round(X['YearsAtCompany'].median()),
        min=X['YearsAtCompany'].min(),
        max=X['YearsAtCompany'].max()
    )
    all_inputs.append(years_at_company_slider)
    
    years_in_current_role_slider = widgets.IntSlider(
        description='YearsInCurrentRole',
        value=round(X['YearsInCurrentRole'].median()),
        min=X['YearsInCurrentRole'].min(),
        max=X['YearsInCurrentRole'].max()
    )
    all_inputs.append(years_in_current_role_slider)
    
    years_since_last_promotion_slider = widgets.IntSlider(
        description='YearsSinceLastPromotion',
        value=round(X['YearsSinceLastPromotion'].median()),
        min=X['YearsSinceLastPromotion'].min(),
        max=X['YearsSinceLastPromotion'].max()
    )
    all_inputs.append(years_since_last_promotion_slider)
    
    years_with_curr_manager_slider = widgets.IntSlider(
        description='YearsWithCurrManager',
        value=round(X['YearsWithCurrManager'].median()),
        min=X['YearsWithCurrManager'].min(),
        max=X['YearsWithCurrManager'].max()
    )
    all_inputs.append(years_with_curr_manager_slider)
    
    u_input = pd.Series({
        'Age': None,
        'BusinessTravel': None,
        'DailyRate': None,
        'Department': None,
        'DistanceFromHome': None,
        'Education': None,
        'EducationField': None,
        'EnvironmentSatisfaction': None,
        'Gender': None,
        'HourlyRate': None,
        'JobInvolvement': None,
        'JobLevel': None,
        'JobRole': None,
        'JobSatisfaction': None,
        'MaritalStatus': None,
        'MonthlyIncome': None,
        'MonthlyRate': None,
        'NumCompaniesWorked': None,
        'OverTime': None,
        'PercentSalaryHike': None,
        'PerformanceRating': None,
        'RelationshipSatisfaction': None,
        'StockOptionLevel': None,
        'TotalWorkingYears': None,
        'TrainingTimesLastYear': None,
        'WorkLifeBalance': None,
        'YearsAtCompany': None,
        'YearsInCurrentRole': None,
        'YearsSinceLastPromotion': None,
        'YearsWithCurrManager': None
    })
    def update_bar():
        for i in range(len(u_input)):
            u_input.iloc[i] = all_inputs[i].get_interact_value()
        input = pd.DataFrame(u_input).transpose()
        value = clf.predict_proba(input)[0, 1] * 100
        bar.value = value
        bar_label.value = f'{value:.2f}% likely to leave the company'
    
    # Make interactive and display
    for elem in all_inputs:
        elem.on_trait_change(update_bar)
        elem.style = {'description_width': '150px'}
        elem.layout = Layout(width='50%')
    update_bar()
    display(*all_inputs, widgets.HBox([bar, bar_label]))