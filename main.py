model="zheng_261301339_mymodel"
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import pandas as pd
engineDf =  pd.read_excel("https://www.dropbox.com/scl/fi/dcoz9yw3f8yywtzy82f4z/Engine.xlsx?rlkey=n53hfjjsrddywktksj156jra5&dl=1")

X = engineDf[['Miles', 'Load', 'Speed', 'Oil']]  
y = engineDf['Time']                            

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


pipeline = Pipeline([('scaler', StandardScaler()),('regressor', LinearRegression())])
pipeline.fit(X_train, y_train)

import joblib

joblib.dump(pipeline, "zheng_261301339_mymodel.pkl")
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# 1. 载入模型
model = joblib.load("zheng_261301339_mymodel.pkl")

# 2. 初始化 FastAPI 应用
app = FastAPI()

# 3. 定义输入数据结构
class InputData(BaseModel):
    Miles: float
    Load: float
    Speed: float
    Oil: float

@app.post("/dashboard")
def predict(input: InputData):
    data = [[input.Miles, input.Load, input.Speed, input.Oil]]
    prediction = model.predict(data)[0]
    return {"predicted_time": round(prediction, 2)}

@app.get("/")
def root():
    return {"message": "Hello, API is running!"}
