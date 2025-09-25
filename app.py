from flask import Flask, request, jsonify
from flask_cors import CORS
from model import FakeNewsDetector
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configure CORS
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:*", "http://127.0.0.1:*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Initialize the fake news detector
detector = FakeNewsDetector()
app_start_time = datetime.now()

# Try to load existing model on startup
try:
    if detector.load_model():
        logger.info("✅ Pre-trained model loaded successfully!")
    else:
        logger.warning("⚠️ No pre-trained model found. Train a new model using /api/train endpoint.")
except Exception as e:
    logger.error(f"❌ Error during model initialization: {str(e)}")

# API Routes

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'message': 'Fake News Detection API is running!',
        'uptime': str(datetime.now() - app_start_time),
        'model_loaded': detector.is_trained,
        'confidence_threshold': '70% (below this = FAKE)',
        'features': 'Uses Title + Content for better detection',
        'dataset_format': 'Supports: id, Title, Content, label columns',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST', 'OPTIONS'])
def predict_news():
    """Main prediction endpoint with Title + Content support"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        # Get JSON data
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        
        # Validate input
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing required field: text'}), 400
        
        text = data.get('text', '').strip()
        title = data.get('title', '').strip()  # Optional title field
        
        if not text:
            return jsonify({'error': 'Text field cannot be empty'}), 400
        
        if len(text) < 5:
            return jsonify({'error': 'Text must be at least 5 characters long'}), 400
        
        # Check if model is trained
        if not detector.is_trained:
            return jsonify({
                'error': 'Model not trained yet. Please train the model first using /api/train endpoint.'
            }), 503
        
        # Make prediction using title + content if title provided
        logger.info(f"Making prediction for text length: {len(text)}, title provided: {bool(title)}")
        
        if title:
            result = detector.predict_with_title(title, text)
        else:
            result = detector.predict(text)
        
        # Log prediction with threshold info
        confidence_flag = "⚠️ LOW CONFIDENCE - MARKED AS FAKE" if result.get('confidence_threshold_applied') else "✅"
        logger.info(f"{confidence_flag} Prediction: {result['prediction']} (Confidence: {result['confidence']:.3f})")
        
        return jsonify({
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'confidence_percentage': f"{result['confidence']*100:.2f}%",
            'probabilities': result['probabilities'],
            'analysis_date': result['timestamp'],
            'confidence_threshold_applied': result['confidence_threshold_applied'],
            'title_used': result.get('title_used', False),
            'text_stats': {
                'original_length': result['text_length'],
                'combined_text_length': result.get('combined_text_length', result['text_length'])
            },
            'note': 'Uses Title + Content for enhanced detection. Confidence below 70% = FAKE'
        })
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({'error': str(e)}), 400
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error during prediction'}), 500

@app.route('/api/predict-with-title', methods=['POST'])
def predict_with_title():
    """Dedicated endpoint for predictions with title and content"""
    try:
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        data = request.get_json()
        
        # Validate input
        if not data or 'title' not in data or 'content' not in data:
            return jsonify({'error': 'Missing required fields: title and content'}), 400
        
        title = data.get('title', '').strip()
        content = data.get('content', '').strip()
        
        if not title and not content:
            return jsonify({'error': 'Either title or content must be provided'}), 400
        
        # Check if model is trained
        if not detector.is_trained:
            return jsonify({'error': 'Model not trained yet'}), 503
        
        # Make prediction
        logger.info(f"Making prediction with title: '{title[:50]}...' and content length: {len(content)}")
        result = detector.predict_with_title(title, content)
        
        # Log result
        confidence_flag = "⚠️ LOW CONFIDENCE - MARKED AS FAKE" if result.get('confidence_threshold_applied') else "✅"
        logger.info(f"{confidence_flag} Prediction: {result['prediction']} (Confidence: {result['confidence']:.3f})")
        
        return jsonify({
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'confidence_percentage': f"{result['confidence']*100:.2f}%",
            'probabilities': result['probabilities'],
            'analysis_date': result['timestamp'],
            'confidence_threshold_applied': result['confidence_threshold_applied'],
            'input_data': {
                'title': title,
                'content': content[:100] + '...' if len(content) > 100 else content
            },
            'text_stats': {
                'title_length': len(title),
                'content_length': len(content),
                'combined_text_length': result.get('combined_text_length', 0)
            },
            'note': 'Prediction made using combined Title + Content'
        })
        
    except Exception as e:
        logger.error(f"Prediction with title error: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    """Train model using Title + Content from dataset"""
    try:
        # YOUR EXACT DATASET PATH
        dataset_path = 'dataset2.xlsx'
        
        # Check if dataset exists in multiple possible locations
        possible_paths = [
            dataset_path,                    # Current directory
            f'../data/{dataset_path}',       # Data folder (parent)
            f'data/{dataset_path}',          # Data folder (current)
            f'../{dataset_path}',            # Parent directory
        ]
        
        found_dataset = None
        for path in possible_paths:
            if os.path.exists(path):
                found_dataset = path
                break
        
        if not found_dataset:
            return jsonify({
                'error': f'Dataset "dataset1.xlsx" not found. Please place it in one of these locations: {possible_paths}',
                'note': 'Your dataset should have columns: id, Title, Content, label'
            }), 400
        
        logger.info(f"Starting model training with Title + Content using: {found_dataset}")
        
        # Train the model using YOUR EXACT CODE with Title + Content
        training_results = detector.train_model(found_dataset)
        
        # Save the trained model
        save_info = detector.save_model()
        
        logger.info("✅ Model training completed successfully using Title + Content!")
        
        return jsonify({
            'message': 'Model trained successfully using Title + Content!',
            'dataset_path': found_dataset,
            'training_results': {
                'accuracy': training_results['accuracy'],
                'accuracy_percentage': f"{training_results['accuracy']*100:.2f}%",
                'training_date': detector.model_info['training_date'],
                'training_samples': detector.model_info['training_samples'],
                'test_samples': detector.model_info['test_samples'],
                'total_records': detector.model_info['total_records'],
                'dataset_columns': detector.model_info['dataset_columns'],
                'uses_title_and_content': detector.model_info.get('uses_title_and_content', True)
            },
            'model_saved': True,
            'confidence_threshold': '70% (below this = FAKE)',
            'save_paths': save_info,
            'feature_engineering': 'Combines Title and Content for better detection',
            'note': 'Training uses your exact ML code: TfidfVectorizer(max_features=5000) + LogisticRegression()'
        })
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        return jsonify({'error': f'Training failed: {str(e)}'}), 500

@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Get detailed information about the current model"""
    try:
        info = detector.get_model_info()
        return jsonify(info)
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({'error': 'Failed to get model information'}), 500

@app.route('/api/test-title-content', methods=['POST'])
def test_title_content():
    """Test with various title + content combinations"""
    try:
        if not detector.is_trained:
            return jsonify({'error': 'Model not trained yet'}), 503
        
        # Test cases with title + content
        test_cases = [
            {
                "title": "Breaking: Alien Contact Confirmed",
                "content": "Government sources reveal aliens have been in contact for decades according to leaked documents"
            },
            {
                "title": "Scientists Publish Cancer Research",
                "content": "New breakthrough treatment shows promising results in clinical trials published in Nature Medicine journal"
            },
            {
                "title": "SHOCKING: Secret Government Plot",
                "content": "Exclusive insider reveals hidden conspiracy to control population through mind control technology"
            },
            {
                "title": "Economic Report Shows Growth",
                "content": "Federal Reserve announces positive economic indicators with unemployment at historic lows"
            },
            {
                "title": "Health Study Results Released",
                "content": "Harvard researchers find significant correlation between diet and longevity in 20-year study"
            }
        ]
        
        results = []
        print("\n" + "="*70)
        print("TESTING WITH TITLE + CONTENT COMBINATIONS:")
        print("="*70)
        
        for i, case in enumerate(test_cases, 1):
            result = detector.predict_with_title(case["title"], case["content"])
            threshold_applied = result['confidence_threshold_applied']
            threshold_msg = " [MARKED AS FAKE - LOW CONFIDENCE]" if threshold_applied else ""
            
            print(f"\nTest {i}:")
            print(f"Title: {case['title']}")
            print(f"Content: {case['content']}")
            print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']*100:.2f}%){threshold_msg}")
            
            results.append({
                'test_number': i,
                'title': case['title'],
                'content': case['content'],
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'confidence_percentage': f"{result['confidence']*100:.2f}%",
                'confidence_threshold_applied': threshold_applied,
                'probabilities': result['probabilities']
            })
            
        return jsonify({
            'test_results': results,
            'total_tests': len(results),
            'feature_used': 'Title + Content combined',
            'confidence_threshold': '70%',
            'note': 'Tests demonstrate enhanced detection using both title and content'
        })
        
    except Exception as e:
        logger.error(f"Test title content error: {str(e)}")
        return jsonify({'error': 'Failed to test title content combinations'}), 500

@app.route('/api/dataset-info', methods=['GET'])
def get_dataset_info():
    """Get information about the expected dataset format with Title support"""
    return jsonify({
        'expected_filename': 'dataset1.xlsx',
        'required_columns': ['id', 'Title', 'Content', 'label'],
        'column_descriptions': {
            'id': 'Unique identifier for each record',
            'Title': 'News article title (combined with content for training)',
            'Content': 'Full text content of the news article',
            'label': 'Classification label - must be either "FAKE" or "REAL"'
        },
        'feature_engineering': 'Title and Content are combined with title getting extra weight',
        'example_structure': {
            'id': 1,
            'Title': 'Government Announces New Economic Policy',
            'Content': 'The federal government today announced a comprehensive new economic policy designed to boost growth and reduce unemployment...',
            'label': 'REAL'
        },
        'training_process': 'Combines Title + Content → TfidfVectorizer(max_features=5000) → LogisticRegression()',
        'confidence_threshold': '70% (predictions below this are marked as FAKE)',
        'prediction_options': [
            'Send only content text for basic prediction',
            'Send both title and content for enhanced prediction'
        ]
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({'error': 'Method not allowed'}), 405

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Fake News Detection API Server...")
    print("📊 Enhanced with Title + Content feature engineering:")
    print("   ✅ Dataset: dataset2.xlsx with columns [id, Title, Content, label]")
    print("   ✅ Feature Engineering: Combines Title + Content for better detection")
    print("   ✅ ML Logic: TfidfVectorizer(max_features=5000) + LogisticRegression()")
    print("   ✅ Split: train_test_split(test_size=0.2, random_state=42)")
    print("   ✅ Confidence threshold: 70% (below = FAKE)")
    print("   ✅ Title weight: Title appears twice in combined text for emphasis")
    print("\n📊 API Endpoints:")
    print("   GET  /api/health - Health check")
    print("   POST /api/predict - Predict with text (optional title)")
    print("   POST /api/predict-with-title - Predict with title + content")
    print("   POST /api/train - Train model with Title + Content")
    print("   POST /api/test-title-content - Test various title+content combinations")
    print("   GET  /api/model-info - Get model information")
    print("   GET  /api/dataset-info - Get dataset format info")
    print("🌐 Frontend should run on: http://localhost:5501")
    print("🔗 API running on: http://localhost:5001")
    print("=" * 70)
    
    # Run the app
    app.run(
        host='127.0.0.1',
        port=5001,
        debug=True,
        threaded=True
    )