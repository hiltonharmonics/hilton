from flask import Flask, render_template, request, send_file, redirect, session, url_for, jsonify
import os
import librosa
import numpy as np
import plotly.graph_objs as go
import plotly.offline as pyo
from pydub import AudioSegment
import requests
from functools import wraps
import stripe
import warnings
from datetime import datetime, timedelta
import random, time, smtplib

# Firebase
import firebase_admin
from firebase_admin import credentials, firestore

app = Flask(__name__)
app.secret_key = 'your-secret-key'

# ────── Config ──────
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

FIREBASE_API_KEY = 'AIzaSyD_A4wsScC5wiI0VhjuaGX_IMXTIQjpX-U'
PAYPAL_CLIENT_ID = 'AXAFs6YTAX8boxdRdnw6lWNzUQVI7Th8SxaV9RRgzj0zuW4ktr2fB44r4RLgOUnAmkuQTXxcqA5XTXJa'
PAYPAL_PLAN_ID = 'P-20P97366KR465582CM7VYV4Q'
STRIPE_SECRET_KEY = 'sk_test_51R8yql2ez7n2f8H0KU9yczmO6WOhPWElBNJwSLnWOqFc1CBSWYG7AYtJs1fDwaipbD11XvOnTJzQ9hkwVLnk1roM00j5KUpXsB'
STRIPE_PRICE_ID = 'price_1R90972ez7n2f8H0A2ZJoMd6'
stripe.api_key = STRIPE_SECRET_KEY

UPLOAD_FOLDER = 'uploads'
RETUNE_FOLDER = 'retuned'
for folder in [UPLOAD_FOLDER, RETUNE_FOLDER]:
    os.makedirs(folder, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# In-memory OTP cache
otp_cache = {}

# Email sender function
EMAIL_ADDRESS = 'hiltonharmonicsapp@gmail.com'
EMAIL_PASSWORD = 'diwc lquy dfno ieht'

def send_otp_email(recipient, otp):
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        message = f"Subject: Your OTP Code\n\nYour Hilton Analyzer login code is: {otp}"
        server.sendmail(EMAIL_ADDRESS, recipient, message)

def generate_otp():
    return str(random.randint(100000, 999999))

# ────── Auth Decorators ──────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated

def subscription_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get('subscribed'):
            return redirect('/subscribe')
        return f(*args, **kwargs)
    return decorated

# ────── Firestore Logic ──────
def save_subscription(email):
    expires_at = datetime.utcnow() + timedelta(days=30)
    db.collection('subscriptions').document(email).set({
        'subscribed': True,
        'start_date': datetime.utcnow().isoformat(),
        'expires_at': expires_at.isoformat()
    })

def is_subscription_active(email):
    doc = db.collection('subscriptions').document(email).get()
    if doc.exists:
        data = doc.to_dict()
        expires_at = data.get('expires_at')
        if expires_at and datetime.utcnow() < datetime.fromisoformat(expires_at):
            return True
    return False

# ────── OTP Routes ──────
@app.route('/verify-otp', methods=['POST'])
def verify_otp():
    data = request.get_json()
    user_otp = data.get('otp')
    email = session.get('pending_user')
    if not email:
        return jsonify({'success': False, 'error': 'Session expired'})

    stored_otp, timestamp = otp_cache.get(email, (None, 0))
    if time.time() - timestamp > 300:
        return jsonify({'success': False, 'error': 'OTP expired'})

    if user_otp == stored_otp:
        session['user'] = email
        session['subscribed'] = is_subscription_active(email)
        otp_cache.pop(email, None)
        session.pop('pending_user', None)
        return jsonify({'success': True, 'redirect': '/' if session['subscribed'] else '/subscribe'})
    else:
        return jsonify({'success': False, 'error': 'Incorrect OTP'})

# ────── Add login route with OTP logic ──────
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        payload = {
            'email': request.form['email'],
            'password': request.form['password'],
            'returnSecureToken': True
        }
        url = f'https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_API_KEY}'
        res = requests.post(url, json=payload).json()
        if 'idToken' in res:
            email = res['email']
            otp = generate_otp()
            otp_cache[email] = (otp, time.time())
            send_otp_email(email, otp)
            session['pending_user'] = email
            return render_template('login.html', otp_prompt=True)
        return render_template('login.html', error='Invalid email or password.')
    return render_template('login.html')

# ────── Index route ──────
@app.route('/', methods=['GET', 'POST'])
@login_required
@subscription_required
def index():
    all_results = []
    if request.method == 'POST':
        files = request.files.getlist("file")
        for file in files:
            filename = file.filename
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            
            # Analyze audio
            y, sr = librosa.load(file_path, sr=44100)
            y = y[:sr * 5]
            fft = np.fft.fft(y)
            freqs = np.fft.fftfreq(len(y), 1/sr)
            magnitude = np.abs(fft)
            freqs, magnitude = freqs[freqs > 0], magnitude[freqs > 0]

            a3_range = (freqs >= 200) & (freqs <= 240)
            a3_freq = freqs[a3_range][np.argmax(magnitude[a3_range])] if np.any(a3_range) else None
            a4_est = (a3_freq / 220.0) * 440.0 if a3_freq else None

            top_idx = np.argsort(magnitude)[-5:]
            top_freqs = freqs[top_idx]
            top_notes = [freq_to_note(f) for f in top_freqs]

            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            key_idx = np.argmax(np.sum(chroma, axis=1))
            key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][key_idx]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                tempo = librosa.beat.tempo(y=y, sr=sr)

            tempo_val = round(float(tempo[0]), 1) if tempo is not None and len(tempo) > 0 else None

            solfeggio = [hz for hz in [174, 285, 396, 417, 528, 639, 741, 852, 963]
                         if magnitude[np.argmin(np.abs(freqs - hz))] > np.mean(magnitude) * 1.5]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=freqs, y=magnitude, mode='lines'))
            fig.update_layout(title='Frequency Spectrum', xaxis_title='Hz', yaxis_title='Magnitude', xaxis_range=[0, 2000])
            graph_html = pyo.plot(fig, output_type='div')

            # Retune
            retuned_path = os.path.join(RETUNE_FOLDER, 'retuned_' + os.path.splitext(filename)[0] + '.wav')
            audio = AudioSegment.from_file(file_path)
            shift_ratio = 432.0 / 440.0
            retuned = audio._spawn(audio.raw_data, overrides={"frame_rate": int(audio.frame_rate * shift_ratio)}).set_frame_rate(audio.frame_rate)
            retuned.export(retuned_path, format="wav")

            all_results.append({
                'filename': filename,
                'results': {
                    'a3_freq': round(a3_freq, 2) if a3_freq else None,
                    'a4_est': round(a4_est, 2) if a4_est else None,
                    'top_freqs': [round(float(f), 1) for f in top_freqs],
                    'top_notes': top_notes,
                    'key': key,
                    'tempo': tempo_val,
                },
                'solfeggio': solfeggio,
                'graph_html': graph_html,
                'retuned_file': retuned_path
            })
    return render_template('index.html', all_results=all_results, user=session.get('user'))
def freq_to_note(freq):
    A4 = 440.0
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    if freq <= 0:
        return "N/A"
    n = round(12 * np.log2(freq / A4))
    note_index = (n + 9) % 12
    octave = 4 + ((n + 9) // 12)
    return f"{notes[note_index]}{octave}"

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/login')

@app.route('/download/<path:filename>')
@login_required
def download(filename):
    file_path = os.path.join(RETUNE_FOLDER, os.path.basename(filename))
    return send_file(file_path, as_attachment=True)

# Remaining routes (signup, logout, subscribe, stripe) remain unchanged.

if __name__ == '__main__':
    app.run(debug=True)
