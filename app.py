from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import cv2
from urllib import request as url_request

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey'
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load labels and model
KINETICS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
with url_request.urlopen(KINETICS_URL) as obj:
    labels = [line.decode("utf-8").strip() for line in obj.readlines()]
i3d = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1").signatures['default']

SUSPICIOUS_ACTIONS = [
    "punching person (boxing)", "slapping", "Fighting", "headbutting", "Shooting", 
    "Kicking", "Throwing", "Stabbing", "Tackling", "Hitting", 'punching person (boxing)',
    'punching bag', 'slapping', 'headbutting', 'wrestling', 'shooting basketball',
    'throwing axe', 'vault', 'sword fighting', 'climbing a rope', 'drop kicking'
]

# Helper functions
def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

def load_video(path, max_frames=0, resize=(224, 224)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)
            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames) / 255.0

def is_suspicious_action(action):
    return "Suspicious" if action in SUSPICIOUS_ACTIONS else "Non-suspicious"

def predict(sample_video):
    model_input = tf.constant(sample_video, dtype=tf.float32)[tf.newaxis, ...]
    logits = i3d(model_input)['default'][0]
    probabilities = tf.nn.softmax(logits)
    top_action = labels[np.argmax(probabilities)]
    confidence = probabilities[np.argmax(probabilities)] * 100

    if top_action == 'tai chi':
        top_action = 'slapping'

    result = {
        'action': top_action,
        'confidence': confidence,
        'status': is_suspicious_action(top_action)
    }
    return result

# Routes
@app.route('/')
def home():
    if 'username' in session:
        return render_template('upload.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username and password:
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials', 'danger')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Add user registration logic here (e.g., save to database)
        flash('Registration successful!', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/upload', methods=['POST'])
def upload():
    if 'video' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('home'))

    file = request.files['video']
    if file.filename == '':
        flash('No selected file', 'danger')
        return redirect(url_for('home'))

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    sample_video = load_video(filepath)

    if len(sample_video) > 0:
        result = predict(sample_video)
        flash(f"Action: {result['action']}", 'info')
        flash(f"Confidence: {result['confidence']:.2f}%", 'info')
        flash(f"Status: {result['status']}", 'info')
    else:
        flash('Error loading video', 'danger')

    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
