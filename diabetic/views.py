from django.shortcuts import render
from django.http import HttpResponseNotAllowed
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
import pickle

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    # Loading the dataset
    data = pd.read_csv(r"diabetes.csv")

    # Train test split
    x = data.drop("Outcome", axis=1)
    y = data['Outcome']
    # feature_names = x.columns.tolist()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    # Training and predicting
    model = LogisticRegression(max_iter=500)
    model.fit(x_train, y_train)
    
    if request.method == 'GET':
        try:
            val1 = float(request.GET['pregnancies'])
            val2 = float(request.GET['glucose'])
            val3 = float(request.GET['blood'])
            val4 = float(request.GET['skin'])
            val5 = float(request.GET['insulin'])
            val6 = float(request.GET['BMI'])
            val7 = float(request.GET['pedigree'])
            val8 = float(request.GET['age'])

            values = [val1, val2, val3, val4, val5, val6, val7, val8]
            
            pred = model.predict([values])
            
            if pred[0] == 1:
                result1 = "Positive"
            else:
                result1 = "Negative"
            
            return render(request, 'predict.html', {"result2": result1})
        
        except ValueError:
            # Handle the case where conversion to float fails
            error_message = "One or more input values are not valid numbers."
            return render(request, 'predict.html', {"error": error_message})
    else:
        return HttpResponseNotAllowed(['GET'])
