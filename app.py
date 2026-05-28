#importing the libraries 
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import os



import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt



from openai import OpenAI

from flask_mail import Mail, Message


app = Flask(__name__)


# mail configuring 


app.config['MAIL_SERVER'] = 'smtp.gmail.com'

app.config['MAIL_PORT'] = 587

app.config['MAIL_USE_TLS'] = True

app.config['MAIL_USERNAME'] = 'admin@gmail.com'

app.config['MAIL_PASSWORD'] = 'password-email'

mail = Mail(app)


# loading the model 

model = pickle.load(
    open('performance_model.pkl', 'rb')
)

# GRgroq api 


client = OpenAI(

    api_key="your groq api key",

    base_url="https://api.groq.com/openai/v1"
)


# global storage 


student_data = {}

conversation_history = []

# this is for calling the home page 


@app.route('/')

def home():

    return render_template('index.html')

# AI FUNCTION


def generate_ai(prompt):

    try:

        response = client.chat.completions.create(

            model="llama-3.1-8b-instant",

            messages=[

                {
                    "role": "system",

                    "content":
                    "You are NeuroGrade AI, an intelligent student mentor."
                },

                {
                    "role": "user",

                    "content": prompt
                }

            ],

            temperature=0.7,

            max_tokens=200
        )

        return (
            response
            .choices[0]
            .message
            .content
        )

    except Exception as e:

        return f"AI Error: {str(e)}"


# GENERATE CHARTS


def generate_charts(

    attendance,
    stress_level

):

    os.makedirs(
        "static/charts",
        exist_ok=True
    )

    # ATTENDANCE CHART

    plt.figure(figsize=(5,3))

    plt.bar(
        ["Attendance"],
        [attendance]
    )

    plt.title("Attendance Analysis")

    plt.savefig(
        "static/charts/attendance.png"
    )

    plt.close()

    # STRESS CHART
    

    plt.figure(figsize=(5,3))

    plt.bar(
        ["Stress Level"],
        [stress_level]
    )

    plt.title("Stress Analysis")

    plt.savefig(
        "static/charts/stress.png"
    )

    plt.close()

# PREDICTION ROUTE

@app.route('/predict', methods=['POST'])

def predict():

    global student_data

    # GET INPUTS

    study_hours = float(
        request.form['StudyHours']
    )

    attendance = float(
        request.form['Attendance']
    )

    resources = float(
        request.form['Resources']
    )

    extracurricular = float(
        request.form['Extracurricular']
    )

    motivation = float(
        request.form['Motivation']
    )

    internet = float(
        request.form['Internet']
    )

    gender = float(
        request.form['Gender']
    )

    age = float(
        request.form['Age']
    )

    learning_style = float(
        request.form['LearningStyle']
    )

    online_courses = float(
        request.form['OnlineCourses']
    )

    discussions = float(
        request.form['Discussions']
    )

    assignment_completion = float(
        request.form['AssignmentCompletion']
    )

    edutech = float(
        request.form['EduTech']
    )

    stress_level = float(
        request.form['StressLevel']
    )

    
    # CREATE DATAFRAME

    features = pd.DataFrame([{

        "StudyHours": study_hours,
        "Attendance": attendance,
        "Resources": resources,
        "Extracurricular": extracurricular,
        "Motivation": motivation,
        "Internet": internet,
        "Gender": gender,
        "Age": age,
        "LearningStyle": learning_style,
        "OnlineCourses": online_courses,
        "Discussions": discussions,
        "AssignmentCompletion": assignment_completion,
        "EduTech": edutech,
        "StressLevel": stress_level

    }])

    # ==========================================
    # MODEL PREDICTION
    # ==========================================

    prediction = model.predict(features)[0]

    grade_map = {

        0: "Fail",
        1: "C Grade",
        2: "B Grade",
        3: "A Grade"

    }

    result = grade_map[prediction]

    # ==========================================
    # RECOMMENDATION
    # ==========================================

    if stress_level >= 3:

        recommendation = (
            "High stress detected. "
            "Take proper rest and reduce pressure."
        )

    elif attendance < 60:

        recommendation = (
            "Low attendance detected. "
            "Attend classes regularly."
        )

    elif study_hours < 2:

        recommendation = (
            "Increase study hours for better performance."
        )

    else:

        recommendation = (
            "Performance looks stable. "
            "Keep working consistently."
        )

    # ==========================================
    # GENERATE CHARTS
    # ==========================================

    generate_charts(
        attendance,
        stress_level
    )

    # ==========================================
    # AI INSIGHTS
    # ==========================================

    insights_prompt = f"""

    Generate AI insights for this student.

    Grade:
    {result}

    Attendance:
    {attendance}

    Stress:
    {stress_level}

    Study Hours:
    {study_hours}

    """

    ai_insights = generate_ai(
        insights_prompt
    )

    # ==========================================
    # ROADMAP
    # ==========================================

    roadmap_prompt = f"""

    Create a personalized roadmap
    to improve academic performance.

    Grade:
    {result}

    Attendance:
    {attendance}

    Stress:
    {stress_level}

    Study Hours:
    {study_hours}

    """

    roadmap = generate_ai(
        roadmap_prompt
    )

    # ==========================================
    # RISK ALERTS
    # ==========================================

    risk_alerts = []

    if stress_level >= 3:

        risk_alerts.append(
            "High Stress Risk"
        )

    if attendance < 50:

        risk_alerts.append(
            "Low Attendance Risk"
        )

    if study_hours < 1:

        risk_alerts.append(
            "Academic Performance Risk"
        )

    # ==========================================
    # STORE DATA
    # ==========================================

    student_data = {

        "prediction": result,

        "attendance": attendance,

        "study_hours": study_hours,

        "stress_level": stress_level,

        "recommendation": recommendation,

        "ai_insights": ai_insights,

        "roadmap": roadmap,

        "risk_alerts": risk_alerts
    }

    # ==========================================
    # RETURN DASHBOARD
    # ==========================================

    return render_template(

        'dashboard.html',

        **student_data
    )

# ==========================================
# DASHBOARD
# ==========================================

@app.route('/dashboard')

def dashboard():

    return render_template(

        'dashboard.html',

        **student_data
    )

# ANALYTICS

@app.route('/analytics')

def analytics():

    return render_template(

        'analytics.html',

        **student_data
    )

# ==========================================
# AI INSIGHTS
# ==========================================

@app.route('/insights')

def insights():

    return render_template(

        'insights.html',

        **student_data
    )

# REPORTS

@app.route('/reports')

def reports():

    return render_template(

        'reports.html',

        **student_data
    )

# ==========================================
# CHATBOT
# ==========================================

@app.route('/chatbot', methods=['POST'])

def chatbot():

    question = request.form['question']

    prompt = f"""

    You are NeuroGrade AI.

    Student Details:

    Grade:
    {student_data.get('prediction')}

    Attendance:
    {student_data.get('attendance')}

    Stress:
    {student_data.get('stress_level')}

    Study Hours:
    {student_data.get('study_hours')}

    Recommendation:
    {student_data.get('recommendation')}

    User Question:
    {question}

    Give personalized guidance.

    """

    ai_response = generate_ai(prompt)

    return ai_response

# ==========================================
# SEND REPORT EMAIL
# ==========================================

# ==========================================
# TEXTBLOB IMPORT
# ==========================================

from textblob import TextBlob

# ==========================================
# SENTIMENT ANALYSIS FUNCTION
# ==========================================

def analyze_sentiment(text):

    analysis = TextBlob(text)

    polarity = analysis.sentiment.polarity

    if polarity > 0:

        sentiment = "Positive 😊"

    elif polarity < 0:

        sentiment = "Negative 😔"

    else:

        sentiment = "Neutral 😐"

    return sentiment, polarity

# ==========================================
# SEND REPORT EMAIL
# ==========================================

@app.route('/send_report', methods=['POST'])

def send_report():

    name = request.form['name']

    email = request.form['email']

    feedback = request.form['feedback']

    # ==========================================
    # SENTIMENT ANALYSIS
    # ==========================================

    sentiment, polarity = analyze_sentiment(
        feedback
    )

    # ==========================================
    # AI EMOTION ALERT
    # ==========================================

    if polarity < -0.5:

        emotional_alert = (
            "⚠ High Negative Emotion Detected"
        )

    elif polarity > 0.5:

        emotional_alert = (
            "✅ Positive Emotion Detected"
        )

    else:

        emotional_alert = (
            "ℹ Neutral Emotional State"
        )

    # ==========================================
    # EMAIL BODY
    # ==========================================

    body = f"""

    NeuroGrade AI Student Report

    ==========================================
    STUDENT DETAILS
    ==========================================

    Student Name:
    {name}

    Predicted Grade:
    {student_data.get('prediction')}

    Attendance:
    {student_data.get('attendance')}

    Study Hours:
    {student_data.get('study_hours')}

    Stress Level:
    {student_data.get('stress_level')}

    ==========================================
    STUDENT FEEDBACK
    ==========================================

    {feedback}

    ==========================================
    SENTIMENT ANALYSIS
    ==========================================

    Sentiment:
    {sentiment}

    Emotion Score:
    {round(polarity, 2)}

    AI Emotional Alert:
    {emotional_alert}

    ==========================================
    RECOMMENDATION
    ==========================================

    {student_data.get('recommendation')}

    ==========================================
    AI INSIGHTS
    ==========================================

    {student_data.get('ai_insights')}

    ==========================================
    ROADMAP
    ==========================================

    {student_data.get('roadmap')}

    ==========================================
    RISK ALERTS
    ==========================================

    {student_data.get('risk_alerts')}

    """

    try:

        msg = Message(

            "NeuroGrade AI Student Report",

            sender=app.config['MAIL_USERNAME'],

            recipients=[email]
        )

        msg.body = body

        mail.send(msg)

        return f"""

        Report Sent Successfully

        Sentiment:
        {sentiment}

        Score:
        {round(polarity, 2)}

        """

    except Exception as e:

        return f"Mail Error: {str(e)}"
# ==========================================
# RUN APP
# ==========================================

if __name__ == "__main__":

    app.run(debug=True)
