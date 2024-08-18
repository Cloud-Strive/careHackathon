import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from scipy.sparse import hstack, vstack
import joblib
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField, FloatField, IntegerField, TextAreaField
from wtforms.validators import DataRequired, Email, EqualTo
from flask_mail import Mail, Message
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer
import os
import random
import logging

# %%
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# %%
# Flask application setup
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAIL_SERVER'] = 'smtp.googlemail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your-email@gmail.com'  # Replace with your email
app.config['MAIL_PASSWORD'] = 'your-password'  # Replace with your email password


# %%
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
mail = Mail(app)

# %%
# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    user_type = db.Column(db.String(10), nullable=False)
    is_verified = db.Column(db.Boolean, default=False)

    def get_reset_token(self, expires_sec=1800):
        s = Serializer(app.config['SECRET_KEY'], expires_sec)
        return s.dumps({'user_id': self.id}).decode('utf-8')

    @staticmethod
    def verify_reset_token(token):
        s = Serializer(app.config['SECRET_KEY'])
        try:
            user_id = s.loads(token)['user_id']
        except:
            return None
        return User.query.get(user_id)





# %%
class StartupProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    company_name = db.Column(db.String(100), nullable=False)
    business_type = db.Column(db.String(100), nullable=False)
    funding_stage = db.Column(db.String(20), nullable=False)
    funding_needed = db.Column(db.Float, nullable=False)
    pitch_summary = db.Column(db.Text, nullable=False)
    team_size = db.Column(db.Integer, nullable=False)

# %%
class InvestorProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    full_name = db.Column(db.String(100), nullable=False)
    investment_interests = db.Column(db.String(200), nullable=False)
    investment_budget = db.Column(db.String(20), nullable=False)
    bio = db.Column(db.Text, nullable=False)

# %%
# Forms
class LoginForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

# %%
class RegistrationForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    user_type = StringField('User Type', validators=[DataRequired()])
    submit = SubmitField('Register')

# %%
class RequestResetForm(FlaskForm):
    email = StringField('Email', validators=[DataRequired(), Email()])
    submit = SubmitField('Request Password Reset')

# %%
class ResetPasswordForm(FlaskForm):
    password = PasswordField('New Password', validators=[DataRequired()])
    confirm_password = PasswordField('Confirm New Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Reset Password')

# %%
class StartupProfileForm(FlaskForm):
    company_name = StringField('Company Name', validators=[DataRequired()])
    business_type = StringField('Business Type', validators=[DataRequired()])
    funding_stage = StringField('Funding Stage', validators=[DataRequired()])
    funding_needed = FloatField('Funding Needed', validators=[DataRequired()])
    pitch_summary = TextAreaField('Pitch Summary', validators=[DataRequired()])
    team_size = IntegerField('Team Size', validators=[DataRequired()])
    submit = SubmitField('Submit')

# %%
class InvestorProfileForm(FlaskForm):
    full_name = StringField('Full Name', validators=[DataRequired()])
    investment_interests = StringField('Investment Interests', validators=[DataRequired()])
    investment_budget = StringField('Investment Budget', validators=[DataRequired()])
    bio = TextAreaField('Bio', validators=[DataRequired()])
    submit = SubmitField('Submit')

# %%
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# %%
def send_email(to, subject, template):
    msg = Message(
        subject,
        recipients=[to],
        html=template,
        sender=app.config['MAIL_USERNAME']
    )
    mail.send(msg)

# %%
# Function to load CSV files
def load_csv(file_path, encoding='utf-8', separator=','):
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path, encoding=encoding, sep=separator, on_bad_lines='skip')
        logging.info(f"File loaded successfully: {file_path}")
        logging.info(f"Columns: {df.columns.tolist()}")
        logging.info(f"Number of rows: {len(df)}")
        return df
    except Exception as e:
        logging.error(f"Error loading file {file_path}: {e}")
        return None


# %%
# Extract relevant features
def extract_combined_features(df, feature_columns, new_column_name='combined_features'):
    existing_columns = [col for col in feature_columns if col in df.columns]
    if not existing_columns:
        logging.warning(f"None of the specified feature columns exist in the dataframe for {new_column_name}")
        return

    df[new_column_name] = df[existing_columns].astype(str).agg(' '.join, axis=1)
    logging.info(f"Extracted combined features for {new_column_name}. Sample data:")
    logging.info(df[new_column_name].head())
    logging.info(f"Number of non-empty entries: {df[new_column_name].str.strip().ne('').sum()}")


# %%

# Check if combined features are not empty
def check_combined_features(df, name):
    if df['combined_features'].str.strip().eq('').all():
        logging.error(f"{name} 'combined_features' column is empty.")
        return False
    return True

# %%
# Function to handle missing values
def handle_missing_values(df):
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    df[non_numeric_columns] = df[non_numeric_columns].fillna('')
    return df

# %%
# Function to create and shuffle samples
def create_samples(investor_tfidf, research_tfidf):
    num_investors = investor_tfidf.shape[0]
    num_projects = research_tfidf.shape[0]
    positive_samples = []
    negative_samples = []

    for i in range(num_investors):
        for j in range(num_projects):
            combined_vector = hstack([investor_tfidf[i], research_tfidf[j]])
            positive_samples.append((combined_vector, 1))
            random_project = np.random.randint(0, num_projects)
            combined_vector_neg = hstack([investor_tfidf[i], research_tfidf[random_project]])
            negative_samples.append((combined_vector_neg, 0))

    logging.info(f"Number of positive samples: {len(positive_samples)}")
    logging.info(f"Number of negative samples: {len(negative_samples)}")

    samples = positive_samples + negative_samples
    random.shuffle(samples)
    return samples


# %%
# Train and evaluate models
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    cv_scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, cv=5)
    logging.info(f"{model_name} Cross-validation scores: {cv_scores}")
    logging.info(f"{model_name} Mean CV score: {cv_scores.mean()}")

    y_pred = grid_search.predict(X_test)
    logging.info(f"{model_name} Accuracy: {accuracy_score(y_test, y_pred)}")
    logging.info(f"{model_name} Classification Report:\n{classification_report(y_test, y_pred)}")

    return grid_search

# %%
# Main data processing and model training function
def process_data_and_train_model():
    entrepreneur_profiles = load_csv('togetherNow/Entrepreneur profiles.csv', separator=';')
    research_project_data = load_csv('togetherNow/Research project data (1).csv', separator=';')
    investor_profiles = load_csv('togetherNow/Invester_alumni profiles.csv')

    if entrepreneur_profiles is None or research_project_data is None or investor_profiles is None:
        logging.error("One or more required datasets failed to load. Exiting.")
        return None, None, None

    entrepreneur_profiles = handle_missing_values(entrepreneur_profiles)
    research_project_data = handle_missing_values(research_project_data)
    investor_profiles = handle_missing_values(investor_profiles)

    entrepreneur_feature_columns = ['Fields of knowledge', 'Which topics and industries are you interested in?', 'Impact areas of interest?']
    research_feature_columns = ['Industry', 'Overview of innovation', 'Technology benefits']
    investor_feature_columns = ['Fields of expertise', 'Type of support offering', 'Impact areas of interest']

    extract_combined_features(entrepreneur_profiles, entrepreneur_feature_columns)
    extract_combined_features(research_project_data, research_feature_columns)
    extract_combined_features(investor_profiles, investor_feature_columns)

    if not all(check_combined_features(df, name) for df, name in [
        (entrepreneur_profiles, "Entrepreneur profiles"),
        (research_project_data, "Research project data"),
        (investor_profiles, "Investor profiles")
    ]):
        logging.error("Exiting due to empty combined features.")
        return None, None, None

    vectorizer = TfidfVectorizer(max_features=1000)
    entrepreneur_tfidf = vectorizer.fit_transform(entrepreneur_profiles['combined_features'])
    research_tfidf = vectorizer.transform(research_project_data['combined_features'])
    investor_tfidf = vectorizer.transform(investor_profiles['combined_features'])

    samples = create_samples(investor_tfidf, research_tfidf)
    max_samples = 10000
    if len(samples) > max_samples:
        samples = random.sample(samples, max_samples)

    X = vstack([x[0] for x in samples])
    y = np.array([x[1] for x in samples])

    logging.info(f"Number of samples: {len(y)}")
    logging.info(f"Shape of X before feature selection: {X.shape}")

    selector = VarianceThreshold()
    X_var = selector.fit_transform(X.toarray())
    logging.info(f"Shape of X after removing constant features: {X_var.shape}")

    k_best = min(500, X_var.shape[1])
    feature_selector = SelectKBest(f_classif, k=k_best)
    X_new = feature_selector.fit_transform(X_var, y)
    logging.info(f"Shape of X after SelectKBest: {X_new.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.2, random_state=42)

    gb_model = train_and_evaluate_model(GradientBoostingClassifier(), X_train, X_test, y_train, y_test, "Gradient Boosting")
    rf_model = train_and_evaluate_model(RandomForestClassifier(), X_train, X_test, y_train, y_test, "Random Forest")

    best_model = gb_model if gb_model.best_score_ > rf_model.best_score_ else rf_model
    logging.info(f"Best model: {type(best_model.best_estimator_).__name__}")

    return best_model, vectorizer, feature_selector

# %%
with app.app_context():
    db.create_all()

# %%
model, vectorizer, feature_selector = process_data_and_train_model()

# %%
if model is None or vectorizer is None or feature_selector is None:
    logging.error("Failed to load model, vectorizer, or feature selector. Exiting.")
    exit()

# %%
joblib.dump(model, 'investor_matching_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(feature_selector, 'feature_selector.pkl')

# %%
# Routes
@app.route('/')
def home():
    return render_template('index.html')

# %%
@app.route('/about')
def about():
    return render_template('about.html')

# %%

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and check_password_hash(user.password_hash, form.password.data):
            if user.is_verified:
                login_user(user)
                if user.user_type == 'investor':
                    return redirect(url_for('investor_dashboard'))
                else:
                    return redirect(url_for('startup_dashboard'))
            else:
                flash('Please verify your email before logging in.', 'warning')
        else:
            flash('Invalid email or password', 'danger')
    return render_template('login.html', form=form)


# %%
@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data)
        user = User(email=form.email.data, password_hash=hashed_password, user_type=form.user_type.data)
        db.session.add(user)
        db.session.commit()
        
        token = user.get_reset_token()
        verify_url = url_for('verify_email', token=token, _external=True)
        html = render_template('verify_email.html', verify_url=verify_url)
        subject = "Verify Your Email"
        send_email(user.email, subject, html)
        
        flash('A verification email has been sent. Please check your inbox.', 'info')
        return redirect(url_for('login'))
    return render_template('login.html', form=form)

# %%
@app.route('/verify_email/<token>')
def verify_email(token):
    user = User.verify_reset_token(token)
    if user is None:
        flash('That is an invalid or expired token', 'warning')
        return redirect(url_for('login'))
    user.is_verified = True
    db.session.commit()
    flash('Your email has been verified! You can now log in.', 'success')
    return redirect(url_for('login'))

# %%
@app.route('/reset_password', methods=['GET', 'POST'])
def reset_request():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    form = RequestResetForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            token = user.get_reset_token()
            reset_url = url_for('reset_token', token=token, _external=True)
            html = render_template('reset_email.html', reset_url=reset_url)
            subject = "Password Reset Request"
            send_email(user.email, subject, html)
        flash('An email has been sent with instructions to reset your password.', 'info')
        return redirect(url_for('login'))
    return render_template('reset_request.html', form=form)

# %%
@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_token(token):
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    user = User.verify_reset_token(token)
    if user is None:
        flash('That is an invalid or expired token', 'warning')
        return redirect(url_for('reset_request'))
    form = ResetPasswordForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data)
        user.password_hash = hashed_password
        db.session.commit()
        flash('Your password has been updated! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('reset_token.html', form=form)


# %%
@app.route('/startup_profile', methods=['GET', 'POST'])
@login_required
def startup_profile():
    if current_user.user_type != 'startup':
        return redirect(url_for('home'))
    form = StartupProfileForm()
    if form.validate_on_submit():
        profile = StartupProfile(
            user_id=current_user.id,
            company_name=form.company_name.data,
            business_type=form.business_type.data,
            funding_stage=form.funding_stage.data,
            funding_needed=form.funding_needed.data,
            pitch_summary=form.pitch_summary.data,
            team_size=form.team_size.data
        )
        db.session.add(profile)
        db.session.commit()
        return redirect(url_for('startup_dashboard'))
    return render_template('startupProfile.html', form=form)

# %%
@app.route('/investor_profile', methods=['GET', 'POST'])
@login_required
def investor_profile():
    if current_user.user_type != 'investor':
        return redirect(url_for('home'))
    form = InvestorProfileForm()
    if form.validate_on_submit():
        profile = InvestorProfile(
            user_id=current_user.id,
            full_name=form.full_name.data,
            investment_interests=form.investment_interests.data,
            investment_budget=form.investment_budget.data,
            bio=form.bio.data
        )
        db.session.add(profile)
        db.session.commit()
        return redirect(url_for('investor_dashboard'))
    return render_template('investorProfile.html', form=form)

# %%
@app.route('/startup_dashboard')
@login_required
def startup_dashboard():
    if current_user.user_type != 'startup':
        return redirect(url_for('home'))
    profile = StartupProfile.query.filter_by(user_id=current_user.id).first()
    return render_template('startupDashboard.html', profile=profile)

# %%
@app.route('/investor_dashboard')
@login_required
def investor_dashboard():
    if current_user.user_type != 'investor':
        return redirect(url_for('home'))
    profile = InvestorProfile.query.filter_by(user_id=current_user.id).first()
    return render_template('investorDashboard.html', profile=profile)

# %%
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('home'))

# %%
@app.route('/match', methods=['POST'])
def match():
    data = request.json
    investor_profile = data['investor_profile']
    target_profiles = data['target_profiles']

    investor_vector = vectorizer.transform([investor_profile])
    target_vectors = vectorizer.transform(target_profiles)

    investor_vector_selected = feature_selector.transform(investor_vector.toarray())
    target_vectors_selected = feature_selector.transform(target_vectors.toarray())

    matches = []
    for i, target_vector in enumerate(target_vectors_selected):
        combined_vector = np.hstack([investor_vector_selected, target_vector.reshape(1, -1)])
        match_score = model.predict_proba(combined_vector)[0][1]
        matches.append({'id': i, 'score': float(match_score)})

    matches.sort(key=lambda x: x['score'], reverse=True)

    return jsonify(matches)

# %%
@app.route('/add_profile', methods=['POST'])
def add_profile():
    data = request.json
    profile_type = data['profile_type']
    profile_data = pd.DataFrame([data['profile_data']])

    if profile_type == 'entrepreneur':
        file_path = 'togetherNow/Entrepreneur profiles.csv'
    elif profile_type == 'researcher':
        file_path = 'togetherNow/Research project data (1).csv'
    elif profile_type == 'investor':
        file_path = 'togetherNow/data/Invester_alumni profiles.csv'
    else:
        return jsonify({'error': 'Invalid profile type'}), 400

    try:
        df = pd.read_csv(file_path, sep=';' if profile_type in ['entrepreneur', 'researcher'] else ',')
        df = df.append(profile_data, ignore_index=True)
        df.to_csv(file_path, index=False, sep=';' if profile_type in ['entrepreneur', 'researcher'] else ',')
        return jsonify({'message': 'Profile added successfully'}), 200
    except Exception as e:
        logging.error(f"Failed to add profile: {str(e)}")
        return jsonify({'error': f'Failed to add profile: {str(e)}'}), 500

# %%
if __name__ == '__main__':
    app.run(debug=True)


