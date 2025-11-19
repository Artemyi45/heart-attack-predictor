# app.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import io

# Импортируем нашу библиотеку
from heart_predictor.data_loader import DataLoader
from heart_predictor.preprocessor import DataPreprocessor
from heart_predictor.model import HeartAttackModel

app = FastAPI(
    title="Heart Attack Predictor API",
    description="API для предсказания риска сердечного приступа",
    version="1.0.0"
)

# Создаем компоненты
data_loader = DataLoader()
preprocessor = DataPreprocessor()
model = HeartAttackModel()

print("🔄 Загружаем и обучаем модель...")

# Обучаем модель один раз при запуске
X_train, y_train = data_loader.load_train_data('heart_train.csv')
X_processed = preprocessor.fit_transform(X_train)
model.fit(X_processed, y_train)

print("✅ Модель готова к работе!")

@app.get("/")
def read_root():
    return {"message": "Heart Attack Predictor API работает!"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_ready": True}

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    """
    Предсказание риска сердечного приступа из CSV файла
    """
    try:
        # Проверяем что файл CSV
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Файл должен быть в формате CSV")
        
        print(f"📥 Получен файл: {file.filename}")
        
        # Читаем CSV файл
        contents = file.file.read()
        data = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        print(f"📊 Обрабатываем {len(data)} записей")
        print(f"📋 Колонки в файле: {list(data.columns)}")
        
        # ПЕРЕВОДИМ КОЛОНКИ НА РУССКИЙ (как в DataLoader)
        column_translation = {
            'Age': 'Возраст',
            'Cholesterol': 'Холестерин', 
            'Heart rate': 'Пульс',
            'Diabetes': 'Диабет',
            'Family History': 'Семейная история',
            'Smoking': 'Курение',
            'Obesity': 'Ожирение',
            'Alcohol Consumption': 'Алкоголь',
            'Exercise Hours Per Week': 'Тренировки в неделю (часы)',
            'Diet': 'Тип питания',
            'Previous Heart Problems': 'Проблемы с сердцем в прошлом',
            'Medication Use': 'Приём лекарств',
            'Stress Level': 'Уровень стресса',
            'Sedentary Hours Per Day': 'Сидячих часов в день',
            'Income': 'Доход',
            'BMI': 'ИМТ',
            'Triglycerides': 'Триглицериды',
            'Physical Activity Days Per Week': 'Активных дней в неделю',
            'Sleep Hours Per Week': 'Часов сна в день',
            'Sleep Hours Per Day': 'Часов сна в день',
            'Blood sugar': 'Уровень сахара в крови',
            'CK-MB': 'КФК-МБ',
            'Troponin': 'Тропонин',
            'Gender': 'Пол',
            'Systolic blood pressure': 'Систолическое давление',
            'Diastolic blood pressure': 'Диастолическое давление'
        }
        
        data = data.rename(columns=column_translation)
        
        # КОДИРУЕМ ПОЛ В ЧИСЛА
        if 'Пол' in data.columns:
            data['Пол'] = data['Пол'].map({'Female': 0, 'Male': 1, 'female': 0, 'male': 1})
            data['Пол'] = data['Пол'].fillna(-1)
        
        print(f"🎯 Колонки после перевода: {list(data.columns)}")
        
        # Обрабатываем данные и делаем предсказания
        data_processed = preprocessor.transform(data)
        predictions = model.predict(data_processed)
        
        # Формируем результаты
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "patient_id": i,
                "prediction": int(pred),
                "risk_level": "high" if pred == 1 else "low"
            })
        
        return JSONResponse({
            "status": "success",
            "total_patients": len(predictions),
            "patients_with_risk": int(sum(predictions)),
            "predictions": results
        })
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        raise HTTPException(status_code=400, detail=f"Ошибка обработки: {str(e)}")