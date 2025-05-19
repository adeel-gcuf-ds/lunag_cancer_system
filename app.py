from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import numpy as np
from datetime import datetime
import io
import base64
from PIL import Image
import tensorflow as tf  # You may need to adjust this import based on your model framework

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///lungcancer.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Setup Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load the lung cancer detection model
# Adjust the path to your model file
MODEL_PATH = 'my_model_lung_cancer.keras'
model = None

def load_model():
    global model
    try:
        # Load your model - adjust this based on your model framework
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

# Image preprocessing function - adjust based on your model's requirements
def preprocess_image(image_path, target_size=(150, 150)):
    try:
        img = Image.open(image_path)
        img = img.resize(target_size)
        img = img.convert('RGB')  # Ensure 3 channels
        
        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# Function to get prediction from model
def get_prediction(image_path):
    global model
    
    # Preprocess the image
    processed_image = preprocess_image(image_path)
    
    if processed_image is None:
        return None, 0
    
    # Make prediction
    try:
        predictions = model.predict(processed_image)
        
        # Get the class with highest probability
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        
        # Map class index to class name
        classes = ['lung_aca', 'lung_n', 'lung_scc']  # Adjust based on your model's classes
        predicted_class = classes[class_idx]
        
        return predicted_class, confidence
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, 0

# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    predictions = db.relationship('Prediction', backref='user', lazy=True)

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(200), nullable=False)
    prediction_result = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # Find user by username
        user = User.query.filter_by(username=username).first()
        
        # Check if user exists and password is correct
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Check if username or email already exists
        existing_user = User.query.filter_by(username=username).first()
        existing_email = User.query.filter_by(email=email).first()
        
        if existing_user:
            flash('Username already exists')
            return render_template('signup.html')
        
        if existing_email:
            flash('Email already exists')
            return render_template('signup.html')
        
        # Create new user
        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email, password=hashed_password)
        
        # Add to database
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully! Please log in.')
        return redirect(url_for('login'))
    
    return render_template('signup.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/check_auth')
def check_auth():
    if current_user.is_authenticated:
        return jsonify({'authenticated': True})
    else:
        return jsonify({'authenticated': False})

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file and model is not None:
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get prediction from model
        result, confidence = get_prediction(filepath)
        
        if result is None:
            return jsonify({'error': 'Error processing image or making prediction'})
        
        # Save prediction to database
        prediction = Prediction(
            user_id=current_user.id,
            image_path=f'uploads/{filename}',
            prediction_result=result,
            confidence=confidence
        )
        db.session.add(prediction)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'prediction': result,
            'confidence': confidence,
            'image_path': url_for('static', filename=f'uploads/{filename}')
        })
    else:
        return jsonify({'error': 'Model not loaded or file error'})

@app.route('/history')
@login_required
def history():
    # Get user's prediction history
    predictions = Prediction.query.filter_by(user_id=current_user.id).order_by(Prediction.created_at.desc()).all()
    return render_template('history.html', predictions=predictions)

@app.route('/model_status')
def model_status():
    return jsonify({'model_loaded': model is not None})

if __name__ == '__main__':
    # Create database tables if they don't exist
    with app.app_context():
        db.create_all()
    
    # Load the model
    model_loaded = load_model()
    if not model_loaded:
        print("WARNING: Model could not be loaded. Application will run but predictions will fail.")
    
    app.run(debug=True)