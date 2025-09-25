# Step 2: Import libraries (Step 1 pip install is done via requirements.txt)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import re
from datetime import datetime

class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.model_info = {}
        self.is_trained = False
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def combine_title_content(self, title, content):
        """Combine title and content for better feature extraction"""
        # Handle missing values
        title = str(title) if pd.notna(title) else ""
        content = str(content) if pd.notna(content) else ""
        
        # Clean both title and content
        clean_title = self.preprocess_text(title)
        clean_content = self.preprocess_text(content)
        
        # Combine with separator - title gets more weight by appearing twice
        if clean_title and clean_content:
            combined = f"{clean_title}. {clean_title}. {clean_content}"
        elif clean_title:
            combined = clean_title
        elif clean_content:
            combined = clean_content
        else:
            combined = ""
            
        return combined
    
    def train_model(self, dataset_path):
        """Train model using your EXACT code logic with Title + Content"""
        try:
            print(f"Loading dataset from: {dataset_path}")
            
            # Step 3: Load dataset - YOUR EXACT CODE
            df = pd.read_excel(dataset_path)
            print("Dataset shape:", df.shape)
            print(df.head())

            # Check required columns
            required_columns = ['Title', 'Content', 'label']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Dataset must contain columns: {required_columns}. Missing: {missing_columns}")

            # Step 4: Features and labels - MODIFIED TO INCLUDE TITLE
            print("Combining Title and Content for training...")
            
            # Combine Title and Content for better detection
            df['combined_text'] = df.apply(
                lambda row: self.combine_title_content(row['Title'], row['Content']), 
                axis=1
            )
            
            # Remove rows with empty combined text
            df = df[df['combined_text'].str.len() > 10]
            
            X = df["combined_text"]  # Using combined title + content
            y = df["label"]          # FAKE or REAL
            
            print(f"After combining Title + Content:")
            print(f"Final dataset shape: {df.shape}")
            print(f"Label distribution:")
            print(y.value_counts())

            # Step 5: Train/Test split - YOUR EXACT CODE
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Step 6: Vectorization - YOUR EXACT CODE
            vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
            X_train_tfidf = vectorizer.fit_transform(X_train)
            X_test_tfidf = vectorizer.transform(X_test)

            # Step 7: Train model - YOUR EXACT CODE
            model = LogisticRegression()
            model.fit(X_train_tfidf, y_train)

            # Step 8: Evaluate model - YOUR EXACT CODE
            y_pred = model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            print("✅ Accuracy:", accuracy)
            print(f"✅ Accuracy Percentage: {accuracy*100:.2f}%")
            print("\nClassification Report:\n", classification_report(y_test, y_pred))
            
            # Store the trained components
            self.vectorizer = vectorizer
            self.model = model
            
            # Store model information
            self.model_info = {
                'accuracy': accuracy,
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'features': X_train_tfidf.shape[1],
                'dataset_columns': list(df.columns),
                'total_records': len(df),
                'uses_title_and_content': True
            }
            
            self.is_trained = True
            
            return {
                'accuracy': accuracy,
                'classification_report': classification_report(y_test, y_pred, output_dict=True)
            }
            
        except Exception as e:
            print(f"❌ Error training model: {str(e)}")
            raise
    
    def predict(self, text, title=None):
        """Predict using your EXACT code logic with optional title"""
        if not self.is_trained or not self.model or not self.vectorizer:
            raise ValueError("Model not trained yet! Please train the model first.")
        
        try:
            # If title is provided, combine it with content like in training
            if title:
                combined_text = self.combine_title_content(title, text)
                print(f"Using combined title + content for prediction")
            else:
                # If no title provided, just use the text content
                combined_text = self.preprocess_text(text)
            
            if len(combined_text) < 5:
                raise ValueError("Text too short for reliable analysis. Please provide more content.")
            
            # Step 9: Custom Prediction - YOUR EXACT CODE (modified for combined text)
            sample_text = [combined_text]  # Your code format
            sample_vec = self.vectorizer.transform(sample_text)
            prediction = self.model.predict(sample_vec)
            prediction_prob = self.model.predict_proba(sample_vec)  # probability for each class

            # Get predicted class and confidence - YOUR EXACT CODE
            predicted_class = prediction[0]
            confidence = max(prediction_prob[0])
            
            # NEW FEATURE: If confidence below 70%, mark as FAKE
            confidence_threshold_applied = False
            if confidence < 0.70:
                predicted_class = "FAKE"
                confidence_threshold_applied = True
                print(f"⚠️ Low confidence ({confidence*100:.2f}%), marking as FAKE")
            
            print(f"\nCustom Prediction: {predicted_class}")
            print(f"Confidence: {confidence*100:.2f}%")
            
            # Get probabilities for both classes
            classes = self.model.classes_
            prob_dict = {classes[i]: float(prediction_prob[0][i]) for i in range(len(classes))}
            
            result = {
                'prediction': predicted_class,
                'confidence': float(confidence),
                'probabilities': prob_dict,
                'text_length': len(text),
                'title_used': title is not None,
                'combined_text_length': len(combined_text),
                'timestamp': datetime.now().isoformat(),
                'confidence_threshold_applied': confidence_threshold_applied
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error making prediction: {str(e)}")
            raise
    
    def predict_with_title(self, title, content):
        """Convenience method to predict with both title and content"""
        return self.predict(content, title=title)
    
    def save_model(self, model_dir='models'):
        """Save trained model and vectorizer"""
        try:
            if not self.is_trained:
                raise ValueError("No trained model to save")
            
            # Create models directory
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, 'fake_news_model.pkl')
            vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
            info_path = os.path.join(model_dir, 'model_info.pkl')
            
            # Save model components
            joblib.dump(self.model, model_path)
            joblib.dump(self.vectorizer, vectorizer_path)
            joblib.dump(self.model_info, info_path)
            
            print(f"✅ Model saved successfully!")
            print(f"Model: {model_path}")
            print(f"Vectorizer: {vectorizer_path}")
            
            return {
                'model_path': model_path,
                'vectorizer_path': vectorizer_path,
                'info_path': info_path
            }
            
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
            raise
    
    def load_model(self, model_dir='models'):
        """Load pre-trained model and vectorizer"""
        try:
            model_path = os.path.join(model_dir, 'fake_news_model.pkl')
            vectorizer_path = os.path.join(model_dir, 'vectorizer.pkl')
            info_path = os.path.join(model_dir, 'model_info.pkl')
            
            # Check if files exist
            if not all(os.path.exists(path) for path in [model_path, vectorizer_path]):
                return False
            
            # Load model components
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            
            # Load model info if available
            if os.path.exists(info_path):
                self.model_info = joblib.load(info_path)
            
            self.is_trained = True
            
            print(f"✅ Model loaded successfully!")
            if self.model_info:
                print(f"Training date: {self.model_info.get('training_date', 'Unknown')}")
                print(f"Accuracy: {self.model_info.get('accuracy', 'Unknown'):.4f}")
                print(f"Uses Title + Content: {self.model_info.get('uses_title_and_content', False)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def get_model_info(self):
        """Get information about the current model"""
        if not self.is_trained:
            return {"error": "No trained model available"}
        
        return {
            "is_trained": self.is_trained,
            "model_type": "Logistic Regression with TF-IDF (Title + Content)",
            "vectorizer_features": len(self.vectorizer.get_feature_names_out()) if self.vectorizer else 0,
            "confidence_threshold": "70% (below this = FAKE)",
            "feature_engineering": "Combines Title and Content text",
            **self.model_info
        }

# Test function - YOUR EXACT CODE STYLE
if __name__ == "__main__":
    # Initialize detector
    detector = FakeNewsDetector()
    
    try:
        # Try to load existing model
        if detector.load_model():
            print("Using existing model")
        else:
            print("Training new model with Title + Content...")
            # Step 3: Load dataset - YOUR EXACT PATH
            dataset_path = "dataset1.xlsx"  # Your exact path
            detector.train_model(dataset_path)
            detector.save_model()
        
        # Step 9: Test prediction - YOUR EXACT CODE STYLE
        print("\n" + "="*60)
        print("TESTING WITH TITLE + CONTENT:")
        print("="*60)
        
        # Test with just content
        sample_text = "Alien invasion tomorrow"
        result = detector.predict(sample_text)
        print(f"\nContent only test: {sample_text}")
        print(f"Result: {result}")
        
        # Test with title + content
        sample_title = "Breaking News Alert"
        sample_content = "Government announces major policy change affecting all citizens"
        result = detector.predict_with_title(sample_title, sample_content)
        print(f"\nTitle + Content test:")
        print(f"Title: {sample_title}")
        print(f"Content: {sample_content}")
        print(f"Result: {result}")
        
        # More test cases
        test_cases = [
            {
                "title": "Scientists Publish Research",
                "content": "New study reveals breakthrough in cancer treatment published in Nature journal"
            },
            {
                "title": "SHOCKING: Secret Revealed",
                "content": "Aliens have been living among us according to leaked government documents"
            },
            {
                "title": "Economic Update",
                "content": "Stock markets show positive trends following government policy announcement"
            }
        ]
        
        print("\n" + "="*60)
        print("ADDITIONAL TEST CASES:")
        print("="*60)
        
        for i, case in enumerate(test_cases, 1):
            result = detector.predict_with_title(case["title"], case["content"])
            threshold_msg = " [MARKED AS FAKE - LOW CONFIDENCE]" if result['confidence_threshold_applied'] else ""
            print(f"\nTest {i}:")
            print(f"Title: {case['title']}")
            print(f"Content: {case['content']}")
            print(f"Prediction: {result['prediction']} (Confidence: {result['confidence']*100:.2f}%){threshold_msg}")
            
    except Exception as e:
        print(f"Error: {e}")