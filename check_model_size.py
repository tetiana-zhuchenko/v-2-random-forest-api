import joblib
import sys
import os

def check_model_sizes():
    """Check file sizes and memory usage of model components"""
    
    files = {
        'random_forest_model.pkl': 'Model',
        'vectorizer.pkl': 'Vectorizer', 
        'scaler.pkl': 'Scaler',
        'aspect_keywords.json': 'Aspect Keywords',
        'feature_names.json': 'Feature Names'
    }
    
    print("=== FILE SIZES ===")
    total_size = 0
    for filename, description in files.items():
        if os.path.exists(filename):
            size_mb = os.path.getsize(filename) / (1024 * 1024)
            total_size += size_mb
            print(f"{description}: {size_mb:.2f} MB")
        else:
            print(f"{description}: File not found")
    
    print(f"\nTotal project size: {total_size:.2f} MB")
    
    print("\n=== MEMORY USAGE ===")
    try:
        # Load model and check memory
        model = joblib.load('random_forest_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        
        model_size = sys.getsizeof(model) / (1024 * 1024)
        vectorizer_size = sys.getsizeof(vectorizer) / (1024 * 1024)
        
        print(f"Model memory: {model_size:.2f} MB")
        print(f"Vectorizer memory: {vectorizer_size:.2f} MB")
        
        print("\n=== MODEL INFO ===")
        print(f"Model type: {type(model)}")
        print(f"N estimators: {getattr(model, 'n_estimators', 'N/A')}")
        print(f"Max depth: {getattr(model, 'max_depth', 'N/A')}")
        
        # Check if it fits Render limits
        print("\n=== RENDER COMPATIBILITY ===")
        if total_size > 500:
            print("⚠️ Total size exceeds Render's 500MB limit")
        elif total_size > 100:
            print("⚠️ Large size - may cause deployment issues")
        else:
            print("✅ Size looks good for Render")
            
    except Exception as e:
        print(f"Error loading models: {e}")

if __name__ == "__main__":
    check_model_sizes()