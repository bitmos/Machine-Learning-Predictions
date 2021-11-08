# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""

from django import template
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.urls import reverse
import numpy as np
import pandas as pd
import pickle
import json

def house_price_prediction_model():
    global __locations
    global __data_columns
    global __model
    with open("apps/static/assets/csv/house_price_predictions/columns.json", 'r')as f:
        __data_columns = json.load(f)['data_columns']
        __locations = __data_columns[3:]
    
    with open("apps/static/assets/csv/house_price_predictions/banglore_home_prices_model.pickle", 'rb') as f:
        __model = pickle.load(f)
    return __locations,__data_columns,__model
def house_price_predictions_one_hot_model():
    global data_columns
    global __model
    with open("apps/static/assets/csv/house_price_predictions_one_hot/columns.json", 'r') as f:
        data_columns = json.load(f)

    global __model
    with open("apps/static/assets/csv/house_price_predictions_one_hot/price.pickle", 'rb') as f:
        __model = pickle.load(f)
    return data_columns,__model

def iris_model():
    global target
    global model
    with open('apps/static/assets/csv/iris/columns.json','r') as f:
        target=json.load(f)["data_columns"]
    
    with open("apps/static/assets/csv/iris/iris.pickle",'rb') as f:
        model=pickle.load(f)
    return model,target

def salary_model():
    global data_columns
    global model
    with open("apps/static/assets/csv/salary_prediction/columns.json", 'r') as f:
        data_columns = json.load(f)
            

    global __model
    with open("apps/static/assets/csv/salary_prediction/salary_prediction.pickle", 'rb') as f:
        model = pickle.load(f)
        

    return data_columns,model
 
def titanic_model(x):
    try:
        df=pd.read_csv("apps/static/assets/csv/titanic/titanic.csv")
    except :
        print("failed")
    df.drop(['PassengerId','Name','SibSp','Parch','Ticket','Cabin','Embarked'],axis="columns",inplace=True)
    target=df.Survived
    inputs=df.drop(["Survived"],axis="columns")
    dummies=pd.get_dummies(inputs.Sex)
    inputs=pd.concat([inputs,dummies],axis='columns')
    inputs.drop('Sex',axis='columns',inplace=True)
    inputs.Age=inputs.Age.fillna(inputs.Age.mean())
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(inputs,target,test_size=0.2)
    from sklearn.naive_bayes import GaussianNB
    model=GaussianNB()
    
    model.fit(x_train,y_train)
    print(model.predict([x])[0])
    return model.predict([x])[0]


@login_required(login_url="/login/")
def index(request):
    context = {'segment': 'index'}

    html_template = loader.get_template('home/index.html')
    return HttpResponse(html_template.render(context, request))


@login_required(login_url="/login/")
def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:
        load_template = request.path.split('/')[-1]
        if load_template == 'admin':
            return HttpResponseRedirect(reverse('admin:index'))
        context['segment'] = load_template

        html_template = loader.get_template('home/' + load_template)
        
        if request.method == "GET" and load_template=='house_price_prediction.html':
            house_price_prediction_model()
            context={
            'locations':__locations
        }
               
            return HttpResponse(html_template.render({'context':context}, request))
        elif request.method == "POST" and load_template=='house_price_prediction.html':
            loc=request.POST.get('location')
            sqft=int(request.POST.get('area'))
            bhk=int(request.POST.get('bhk'))
            bath=int(request.POST.get('bath'))
            try:
                loc_index = __data_columns.index(loc.lower())
            except:
                loc_index = -1
            x = np.zeros(len(__data_columns))
            x[0] = sqft
            x[1] = bath
            x[2] = bhk
            if loc_index >= 0:
                x[loc_index] = 1
            price=round(__model.predict([x])[0], 2)
            context = {
              'locations':__locations,
              'selected' :loc,
              'bhk' : bhk,
              'bath' : bath,
              'area':sqft,
              'price':price
         }
            return HttpResponse(html_template.render({'context':context}, request))

        if request.method == "GET" and load_template=='titanic.html':
            context
            return HttpResponse(html_template.render({'context':context}, request))
        elif request.method == "POST" and load_template=='titanic.html':
            pclass=int(float(request.POST.get('class')))
            gender=request.POST.get('gender')
            age=request.POST.get('age')
            fare=request.POST.get('fare')
            x = np.zeros(5)
            x[0] = pclass
            x[1] = age
            x[2]=fare
            if gender=="Female":
                x[4] = 1
            else :
                x[3]=1
            print(x)
            prd=titanic_model(x)
            print("prd is ",prd)
            if(prd==0):
                text="Not Survived"
            else:
                text="Survived"
            context = {
                'selected_class':pclass,
                'selected_gender':gender,
                'age':age,
                'fare':fare,
                'price':text
                
            }
            return HttpResponse(html_template.render({'context':context}, request))
        if request.method == "GET" and load_template=='house_price_predictions_one_hot.html':
            house_price_predictions_one_hot_model()
            context={
            'locations':data_columns['towns']
        }
            return HttpResponse(html_template.render({'context':context}, request))
        elif request.method == "POST" and load_template=='house_price_predictions_one_hot.html':
            loc=request.POST.get('location')
            sqft=int(request.POST.get('area'))

            x = np.zeros(3)
            if (loc=="monroe township"):
                x[0]=1
        
            elif (loc=="robinsville"):
                x[1]=1
            
            x[2]=sqft
            price=round(__model.predict([x])[0], 2)
            context = {
              'locations':data_columns['towns'],
              'selected' :loc,
              'area':sqft,
              'price':price
         }
            return HttpResponse(html_template.render({'context':context}, request))
        if request.method=="GET" and load_template=='iris.html':
            iris_model()
            context
            return HttpResponse(html_template.render({'context':context}, request))
        elif request.method == "POST" and load_template=='iris.html':
            slength=float(request.POST.get('slength'))
            swidth=float(request.POST.get('swidth'))
            plength=float(request.POST.get('plength'))
            pwidth=float(request.POST.get('pwidth'))

            x = np.zeros(4)
            x[0] = slength
            x[1] = swidth
            x[2] = plength
            x[3]=pwidth
            price=target[model.predict([x])[0]]
            context = {
                'slength':slength,
                'swidth':swidth,
                'plength':plength,
                'pwidth':pwidth,
              'price':price
         }
            return HttpResponse(html_template.render({'context':context}, request))
        
        if request.method=="GET" and load_template=='salary.html':
            salary_model()
            context={
            'company':data_columns['company'],
            'job':data_columns['job'],
            'degree':data_columns['degree']
        }
            return HttpResponse(html_template.render({'context':context}, request))
        elif request.method == "POST" and load_template=='salary.html':
            loc=request.POST.get('location')
            company=request.POST.get('company')
            job=request.POST.get('job')
            degree=request.POST.get('degree')

            x = np.zeros(3)
            if (company=="google"):
                x[0]=2
            elif (company=="abc pharma"):
                x[0]=1
            if (job=="sales executive"):
                x[1]=2
            elif (job=="business manager"):
                x[1]=1
            if (degree=="masters"):
                x[2]=1
        
            price=round(model.predict([x])[0], 2)
            if price == 1 :
                t="More than 100k"
            else:
                t="Less than 100k"
            context = {
                'company':data_columns['company'],
            'job':data_columns['job'],
            'degree':data_columns['degree'],
            'selected_company':company,
            'selected_job':job,
            'selected_degree':degree,
            'price':price,
            'text':t
                
            }
            return HttpResponse(html_template.render({'context':context}, request))

    except template.TemplateDoesNotExist:

        html_template = loader.get_template('home/page-404.html')
        return HttpResponse(html_template.render(context, request))

    except:
        html_template = loader.get_template('home/page-500.html')
        return HttpResponse(html_template.render(context, request))
