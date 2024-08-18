import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import flet as ft
from flet import Page, TextField, Slider, ElevatedButton, Text, Row, Column, Dropdown, AlertDialog
import json
from datetime import datetime
import sqlite3

# Initialize global DataFrames
investor_df = pd.DataFrame()
entrepreneur_df = pd.DataFrame()
project_df = pd.DataFrame()

def init_db():
    with sqlite3.connect('app_data.db') as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (username TEXT PRIMARY KEY, password TEXT)''')
        c.execute('''CREATE TABLE IF NOT EXISTS feedback
                     (profile_id TEXT, rating INTEGER, timestamp TEXT)''')

def load_settings():
    if os.path.exists('settings.json'):
        with open('settings.json', 'r') as f:
            return json.load(f)
    return {
        "criteria_weights": {
            "industry_match": 0.3,
            "technology_fit": 0.4,
            "impact_area": 0.3
        },
        "minimum_match_score": 0.5
    }

settings = load_settings()

def save_settings():
    with open('settings.json', 'w') as f:
        json.dump(settings, f)

def load_data(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()

def initialize_data():
    global investor_df, entrepreneur_df, project_df
    investor_df = load_data('data/Investor_profiles.csv')
    entrepreneur_df = load_data('data/Entrepreneur_profiles.csv')
    project_df = load_data('data/Research_project_data.csv')

def extract_combined_features(df):
    df = df.fillna('')
    return df['industry'] + ' ' + df['technology'] + ' ' + df['impact_area']

def train_and_cache_model():
    initialize_data()
    
    combined_features = extract_combined_features(entrepreneur_df)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(combined_features)
    
    y = entrepreneur_df['target_label']  # Replace with your actual target label column
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    
    rf_model = RandomForestClassifier()
    clf = GridSearchCV(rf_model, param_grid, cv=5)
    clf.fit(X, y)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(clf.best_estimator_, 'models/investor_matching_model.pkl')
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    
    return clf.best_estimator_, vectorizer

def load_or_train_model():
    if os.path.exists('models/investor_matching_model.pkl') and os.path.exists('models/tfidf_vectorizer.pkl'):
        model = joblib.load('models/investor_matching_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    else:
        model, vectorizer = train_and_cache_model()
    return model, vectorizer

model, vectorizer = load_or_train_model()

def match_profiles(investor_profile_text):
    investor_features = vectorizer.transform([investor_profile_text])
    matches = model.predict_proba(investor_features)[0]
    
    sorted_indices = np.argsort(matches)[::-1]
    return [(entrepreneur_df.iloc[i], matches[i]) for i in sorted_indices if matches[i] >= settings['minimum_match_score']]

def save_feedback(profile_id, rating):
    with sqlite3.connect('app_data.db') as conn:
        c = conn.cursor()
        c.execute('INSERT INTO feedback (profile_id, rating, timestamp) VALUES (?, ?, ?)',
                  (profile_id, rating, datetime.now().isoformat()))

def generate_explanations(matches):
    return [f"Matched {match['name']} with a score of {score:.2f} based on:\n"
            f"- Industry match: {match['industry']}\n"
            f"- Technology fit: {match['technology']}\n"
            f"- Impact area alignment: {match['impact_area']}"
            for match, score in matches]

def update_data(data_type, new_data):
    global investor_df, entrepreneur_df, project_df
    if data_type == 'investor':
        investor_df = pd.concat([investor_df, new_data])
    elif data_type == 'entrepreneur':
        entrepreneur_df = pd.concat([entrepreneur_df, new_data])
    elif data_type == 'project':
        project_df = pd.concat([project_df, new_data])
    
    train_and_cache_model()

def main(page: Page):
    init_db()

    def show_screen(screen_name):
        page.controls.clear()
        if screen_name == 'login':
            show_login_screen()
        elif screen_name == 'register':
            show_register_screen()
        elif screen_name == 'main':
            show_main_screen()
        page.update()

    def show_login_screen():
        username_input = TextField(label="Username")
        password_input = TextField(label="Password", password=True)
        
        def on_login(e):
            with sqlite3.connect('app_data.db') as conn:
                c = conn.cursor()
                c.execute('SELECT * FROM users WHERE username = ? AND password = ?', 
                          (username_input.value, password_input.value))
                if c.fetchone():
                    show_screen('main')
                else:
                    page.dialog = AlertDialog(title="Login Failed", content=Text("Invalid username or password"))
                    page.dialog.open = True
                    page.update()
        
        login_button = ElevatedButton("Login", on_click=on_login)
        register_button = ElevatedButton("Register", on_click=lambda _: show_screen('register'))
        page.add(Column([username_input, password_input, login_button, register_button]))

    def show_register_screen():
        username_input = TextField(label="Username")
        password_input = TextField(label="Password", password=True)
        
        def on_register(e):
            with sqlite3.connect('app_data.db') as conn:
                c = conn.cursor()
                c.execute('SELECT * FROM users WHERE username = ?', (username_input.value,))
                if c.fetchone():
                    page.dialog = AlertDialog(title="Registration Failed", content=Text("Username already exists"))
                    page.dialog.open = True
                    page.update()
                else:
                    c.execute('INSERT INTO users (username, password) VALUES (?, ?)', 
                              (username_input.value, password_input.value))
                    show_screen('login')
        
        register_button = ElevatedButton("Register", on_click=on_register)
        back_button = ElevatedButton("Back to Login", on_click=lambda _: show_screen('login'))
        page.add(Column([username_input, password_input, register_button, back_button]))

    def show_main_screen():
        profile_input = TextField(label="Enter Investor Profile", multiline=True)
        result_text = Text()
        explanation_text = Text()
        
        industry_weight = Slider(min=0, max=1, value=settings['criteria_weights']['industry_match'], label="Industry Weight")
        technology_weight = Slider(min=0, max=1, value=settings['criteria_weights']['technology_fit'], label="Technology Weight")
        impact_weight = Slider(min=0, max=1, value=settings['criteria_weights']['impact_area'], label="Impact Area Weight")
        min_score_slider = Slider(min=0, max=1, value=settings['minimum_match_score'], label="Minimum Match Score")
        
        def update_weights(e):
            settings['criteria_weights']['industry_match'] = industry_weight.value
            settings['criteria_weights']['technology_fit'] = technology_weight.value
            settings['criteria_weights']['impact_area'] = impact_weight.value
            settings['minimum_match_score'] = min_score_slider.value
            save_settings()
        
        update_button = ElevatedButton("Update Weights", on_click=update_weights)
        
        data_type_dropdown = Dropdown(
            label="Select data type to update",
            options=[
                ft.dropdown.Option("investor"),
                ft.dropdown.Option("entrepreneur"),
                ft.dropdown.Option("project")
            ],
        )
        update_data_input = TextField(label="Enter new data (CSV format)")
        
        def on_data_update(e):
            new_data = pd.read_csv(StringIO(update_data_input.value))
            update_data(data_type_dropdown.value, new_data)
            page.add(Text(f"{data_type_dropdown.value.capitalize()} data updated successfully."))
            page.update()
        
        update_data_button = ElevatedButton("Update Data", on_click=on_data_update)
        
        def on_match(e):
            matches = match_profiles(profile_input.value)
            result_text.value = "\n".join([f"Match: {match[0]['name']}, Score: {match[1]:.2f}" for match in matches])
            explanations = generate_explanations(matches)
            explanation_text.value = "\n\n".join(explanations)
            page.update()

        match_button = ElevatedButton("Match", on_click=on_match)
        
        feedback_slider = Slider(min=1, max=5, divisions=4, label="Rate this match")
        
        def on_feedback(e):
            profile_id = "example_profile_id"  # Replace with actual profile ID
            save_feedback(profile_id, int(feedback_slider.value))
            page.add(Text(f"Feedback for profile {profile_id} saved with rating {int(feedback_slider.value)}."))
            page.update()
        
        feedback_button = ElevatedButton("Submit Feedback", on_click=on_feedback)

        logout_button = ElevatedButton("Logout", on_click=lambda _: show_screen('login'))

        page.add(Column([
            profile_input, match_button, result_text, explanation_text,
            Row([industry_weight, technology_weight, impact_weight]),
            min_score_slider, update_button,
            data_type_dropdown, update_data_input, update_data_button,
            feedback_slider, feedback_button, logout_button
        ]))

    show_screen('login')

ft.app(target=main)
