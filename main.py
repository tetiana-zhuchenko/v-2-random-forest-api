from fastapi import FastAPI, HTTPException
import joblib
import json
import re
import os
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

# Глобальні змінні для моделі та компонентів
model = None
vectorizer = None
scaler = None
aspect_keywords = None
feature_names = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, vectorizer, scaler, aspect_keywords, feature_names

    try:
        # Завантаження Random Forest моделі
        model = joblib.load('random_forest_model.pkl')
        print("Random Forest модель завантажена успішно!")
        
        # Завантаження vectorizer
        vectorizer = joblib.load('vectorizer.pkl')
        print("Vectorizer завантажено")
        
        # Завантаження scaler
        try:
            scaler = joblib.load('scaler.pkl')
            print("Scaler завантажено")
        except Exception as e:
            print(f"Помилка завантаження scaler: {e}")
            scaler = None
        
        # Завантаження aspect keywords
        with open('aspect_keywords.json', 'r', encoding='utf-8') as f:
            aspect_keywords = json.load(f)
        print("Словники аспектів завантажено")
        
        # Завантаження feature names для валідації
        with open('feature_names.json', 'r', encoding='utf-8') as f:
            feature_names = json.load(f)
        print(f"Feature names завантажено: {len(feature_names)} ознак")
        
        print("Всі компоненти завантажені успішно!")
        
    except Exception as e:
        print(f"Помилка завантаження компонентів: {e}")
        raise

    yield  # Application starts here and serves requests

    # Cleanup after shutdown
    model = None
    vectorizer = None
    scaler = None
    aspect_keywords = None
    feature_names = None
    print("Всі ресурси звільнено")

app = FastAPI(title="Restaurant Review Sentiment API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

def handle_negations(text):
    """Замінює 'не + позитивне_слово' на відповідне негативне слово."""
    if pd.isna(text) or not isinstance(text, str):
        return text
    text_lower = text.lower()
    negation_patterns = [
        # Їжа - смак і якість
        (r'не\s+(дуже\s+)?смачн\w*', 'несмачн'),
        (r'не\s+(дуже\s+)?свіж\w*', 'несвіж'),  
        (r'не\s+(дуже\s+)?гаряч\w*', 'холодн'),
        (r'не\s+(дуже\s+)?соковит\w*', 'сух'),
        (r'не\s+(дуже\s+)?ніжн\w*', 'тверд'),
        (r'не\s+(дуже\s+)?ароматн\w*', 'безсмаков'),
        (r'не\s+(дуже\s+)?якісн\w*', 'неякісн'),
        # Обслуговування
        (r'не\s+(дуже\s+)?швидк\w*', 'повільн'),
        (r'не\s+(дуже\s+)?ввічлив\w*', 'груб'),
        (r'не\s+(дуже\s+)?професійн\w*', 'непрофесійн'),
        (r'не\s+(дуже\s+)?уважн\w*', 'неуважн'),
        (r'не\s+(дуже\s+)?привітн\w*', 'неприв ітн'),
        (r'не\s+(дуже\s+)?дружн\w*', 'недружн'),
        (r'не\s+(дуже\s+)?компетентн\w*', 'некомпетентн'),
        # Атмосфера
        (r'не\s+(дуже\s+)?затишн\w*', 'незатишн'),
        (r'не\s+(дуже\s+)?комфортн\w*', 'некомфортн'),
        (r'не\s+(дуже\s+)?чист\w*', 'брудн'),
        (r'не\s+(дуже\s+)?красив\w*', 'некрасив'),
        (r'не\s+(дуже\s+)?спокійн\w*', 'шумн'),
        (r'не\s+(дуже\s+)?тепл\w*', 'холодн'),
        (r'не\s+(дуже\s+)?стільн\w*', 'нестільн'),
        # Загальні оцінки
        (r'не\s+(дуже\s+)?добр\w*', 'погані'),
        (r'не\s+(дуже\s+)?гарн\w*', 'поганий'),
        (r'не\s+(дуже\s+)?відмінн\w*', 'погані'),
        (r'не\s+(дуже\s+)?класн\w*', 'погані'),
        (r'не\s+(дуже\s+)?чудов\w*', 'жахлив'),
        (r'не\s+(дуже\s+)?прекрасн\w*', 'жахлив'),
        # Цінність
        (r'не\s+(дуже\s+)?дешев\w*', 'дорог'),
        (r'не\s+(дуже\s+)?доступн\w*', 'дорог'),
        (r'не\s+(дуже\s+)?вигідн\w*', 'невигідн'),
        (r'не\s+(дуже\s+)?економн\w*', 'дорог'),
        # Складні конструкції
        (r'зовсім\s+не\s+', 'абсолютно_не_'),
        (r'взагалі\s+не\s+', 'зовсім_не_'),
        (r'не\s+зовсім\s+', 'частково_'),
        (r'не\s+дуже\s+', 'трохи_'),
        # Емоційні конструкції
        (r'не\s+сподобал\w*', 'розчарував'),
        (r'не\s+рекоменд\w*', 'не_рекомендую'),
        (r'не\s+варт\w*', 'не_вартує'),
        (r'не\s+задовольн\w*', 'незадовольн'),
        (r'не\s+влаштов\w*', 'не_влаштовує')
    ]
    
    # Застосування патернів
    result_text = text_lower
    for pattern, replacement in negation_patterns:
        result_text = re.sub(pattern, replacement, result_text)
    return result_text

def preprocess_text(text):
    """Обробка тексту як при тренуванні моделі"""
    # СПОЧАТКУ обробити заперечення
    text = handle_negations(str(text))
    
    # Приведення до нижнього регістру
    text = str(text).lower()
    
    # Видалення спеціальних символів, залишаємо тільки букви та пробіли
    text = re.sub(r'[^a-zA-Zа-яА-ЯіїєґІЇЄҐ\s]', '', text)
    
    # Видалення зайвих пробілів
    text = ' '.join(text.split())
    
    return text

def analyze_aspects(text):
    """Аналіз аспектів тексту"""
    results = {}
    
    for aspect_name, aspect_dict in aspect_keywords.items():
        positive_count = sum(1 for word in aspect_dict['positive'] if word in text.lower())
        negative_count = sum(1 for word in aspect_dict['negative'] if word in text.lower())
        
        total_mentions = positive_count + negative_count
        if total_mentions > 0:
            sentiment_ratio = (positive_count - negative_count) / total_mentions
        else:
            sentiment_ratio = 0.0
            
        results[aspect_name] = {
            'positive_count': positive_count,
            'negative_count': negative_count,
            'net_sentiment': positive_count - negative_count,
            'sentiment_ratio': sentiment_ratio,
            'total_mentions': total_mentions
        }
    
    return results

@app.post("/predict")
async def predict(request: dict):
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не завантажена")
    
    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Текст не може бути порожнім")
    
    try:
        # 1. Обробка тексту як при тренуванні
        processed_text = preprocess_text(text)
        
        # 2. TF-IDF векторизація
        text_features = vectorizer.transform([processed_text])
        
        # 3. Базові текстові ознаки (на основі оригінального тексту)
        basic_features = np.array([
            len(text),  # text_length
            len(text.split()),  # word_count  
            text.count('.') + text.count('!') + text.count('?') + 1,  # sentence_count
            np.mean([len(word) for word in text.split()]) if text.split() else 0,  # avg_word_length
            text.count('!'),  # exclamation_count
            text.count('?'),  # question_count
            sum(1 for c in text if c.isupper())  # caps_words
        ])
        
        # 4. Аспектні ознаки (на основі обробленого тексту)
        aspects = analyze_aspects(processed_text)
        aspect_features = []
        
        # Для кожного аспекту додаємо 4 ознаки
        for aspect in ['food', 'service', 'atmosphere', 'value']:
            aspect_data = aspects.get(aspect, {
                'positive_count': 0, 'negative_count': 0, 
                'net_sentiment': 0, 'sentiment_ratio': 0
            })
            aspect_features.extend([
                aspect_data['positive_count'],
                aspect_data['negative_count'], 
                aspect_data['net_sentiment'],
                aspect_data['sentiment_ratio']
            ])
        
        # 5. Загальні аспектні ознаки  
        total_mentions = sum(aspects[asp]['total_mentions'] for asp in aspects)
        total_positive = sum(aspects[asp]['positive_count'] for asp in aspects)
        total_negative = sum(aspects[asp]['negative_count'] for asp in aspects)
        
        general_aspect_features = [
            total_mentions,  # total_aspect_mentions
            total_positive,  # total_positive_aspects
            total_negative,  # total_negative_aspects
            len([asp for asp in aspects if aspects[asp]['total_mentions'] > 0]),  # aspect_balance
            0,  # aspect_ambivalence (складно обчислити без повного контексту)
            (total_positive - total_negative) / max(total_positive + total_negative, 1)  # overall_aspect_sentiment
        ]
        
        # 6. Об'єднання всіх числових ознак
        all_numeric_features = np.concatenate([basic_features, aspect_features, general_aspect_features])

        # Масштабування числових ознак якщо є scaler
        if scaler is not None:
            all_numeric_features = scaler.transform([all_numeric_features])[0]

        # Об'єднання TF-IDF та числових ознак
        combined_features = hstack([text_features, csr_matrix([all_numeric_features])])

        # Валідація розміру ознак
        if feature_names and combined_features.shape[1] != len(feature_names):
            raise HTTPException(
                status_code=500, 
                detail=f"Невідповідність ознак: очікується {len(feature_names)}, отримано {combined_features.shape[1]}"
            )

        # Random Forest працює зі sparse матрицями, не потребує конвертації
        
        print(f"Оригінальний текст: {text}")
        print(f"Оброблений текст: {processed_text}")
        print(f"Розмір ознак: {combined_features.shape}")
        
        # 7. Прогнозування
        prediction = model.predict(combined_features)[0]
        probabilities = model.predict_proba(combined_features)[0] 
        confidence = float(max(probabilities))
        
        # Перевіряємо чи є негативні аспекти
        has_negative_aspect = any(aspects[asp]['sentiment_ratio'] < 0 for asp in aspects)

        return {
            "sentiment": int(prediction),
            "sentiment_general": -1 if has_negative_aspect else 1,
            "confidence": confidence,
            "label": "Позитивний" if prediction == 1 else "Негативний",
            "final_label": "Негативний" if has_negative_aspect else "Позитивний",
            "aspects": {asp: aspects[asp]['sentiment_ratio'] for asp in aspects},
        }
        
    except Exception as e:
        print(f"Детальна помилка: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Помилка обробки: {str(e)}")

@app.post("/test_preprocessing")
async def test_preprocessing(request: dict):
    """Тестовий endpoint для перевірки обробки тексту"""
    text = request.get("text", "")
    
    if not text:
        raise HTTPException(status_code=400, detail="Текст не може бути порожнім")
    
    try:
        negation_handled = handle_negations(text)
        final_processed = preprocess_text(text)
        aspects = analyze_aspects(final_processed) if aspect_keywords else {}
        
        return {
            "original": text,
            "after_negation_handling": negation_handled,
            "final_processed": final_processed,
            "aspects_detected": {
                asp: {
                    "positive": data['positive_count'],
                    "negative": data['negative_count'],
                    "total": data['total_mentions']
                } for asp, data in aspects.items()
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Помилка тестування: {str(e)}")

@app.get("/")
def root():
    return {
        "message": "Restaurant Review Sentiment API with Random Forest", 
        "status": "OK",
        "model": "Random Forest",
        "version": "2.0"
    }

@app.get("/health")  
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_type": "Random Forest" if model else None,
        "vectorizer_loaded": vectorizer is not None,
        "scaler_loaded": scaler is not None,
        "aspects_loaded": aspect_keywords is not None,
        "expected_features": len(feature_names) if feature_names else "unknown",
        "components": {
            "model": "✓" if model else "✗",
            "vectorizer": "✓" if vectorizer else "✗", 
            "scaler": "✓" if scaler else "✗",
            "aspect_keywords": "✓" if aspect_keywords else "✗",
            "feature_names": "✓" if feature_names else "✗"
        }
    }

@app.get("/model_info")
def model_info():
    """Інформація про завантажену модель"""
    if not model:
        raise HTTPException(status_code=500, detail="Модель не завантажена")
    
    return {
        "model_type": str(type(model).__name__),
        "model_class": str(type(model)),
        "n_estimators": getattr(model, 'n_estimators', 'N/A'),
        "max_depth": getattr(model, 'max_depth', 'N/A'),
        "random_state": getattr(model, 'random_state', 'N/A'),
        "feature_count": len(feature_names) if feature_names else "unknown",
        "aspects": list(aspect_keywords.keys()) if aspect_keywords else []
    }