# -*- coding: utf-8 -*-
from flask import Flask, render_template, request
import numpy as np
import joblib

# 모델 가져오기
model = joblib.load("/Users/hongrolee/Documents/GitHub/edu_Machine_Learning/web_service/model/iris_classification_model.pkl")

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    if request.method == 'POST':
        
        # 파라미터를 전달 받습니다.
        sepal_length = float(request.form.get('sepal_length',False))
        sepal_width = float(request.form.get('sepal_width',False))
        petal_length = float(request.form.get('petal_length',False))
        petal_width = float(request.form.get('petal_width',False))     
           
        # 꽃의 품종을 분류합니다.        
        kind = model.predict([[sepal_length,sepal_width,petal_length,petal_width]])

        return render_template('index.html', result=kind)

if __name__ == '__main__':
   app.run(debug = True)