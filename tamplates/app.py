
import flask
from flask import render_template
import pickle
import sklearn
import numpy as np # Линейная алгебра
import pandas as pd
import random as rd # получение случайных значений
import datetime # формат даты
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

app = flask.Flask(__name__, template_folder= 'templates')

@app.route('/', methods = ['POST', 'GET'])

@app.route('/index', methods = ['POST', 'GET'])

def main():
    if flask.request.method == 'GET':
        return render_template('main.html')
    
    if flask.request.method == 'POST':
       #with open('ARIMA1_model (1).pkl', 'rb') as f1:
           #loaded_model1 = pickle.laod(f1)
       #with open('ARIMA2_model (1).pkl', 'rb') as f2:
           #loaded_model2 = pickle.laod(f2)
       #with open('ARIMA3_model (1).pkl', 'rb') as f3:
           #loaded_model3 = pickle.laod(f3)
       #with open('ARIMA4_model.pkl', 'rb') as f4:
           #loaded_model4 = pickle.laod(f4)
       #with open('ARIMA5_model.pkl', 'rb') as f5:
           #loaded_model5 = pickle.laod(f5)
    
       kod_1 = float(flask.request.form['product_code'])
     
       data_train = pd.read_excel('T:/VKR/Train data.xlsx')
       data_train = data_train.rename(columns={"Товар код" : "product_code", "Вид товара" : "type_of_product", "Дата документа" : "date_of_sale", "Отдел" : "pharmacy_id", "Кол-во товара" : "quantity_sale"})
       data_train.date_of_sale = pd.to_datetime(data_train.date_of_sale)
       p_id_0 = 0
       dtr_0 = data_train[data_train['pharmacy_id'] == p_id_0]

       p_id_1 = 1
       dtr_1 = data_train[data_train['pharmacy_id'] == p_id_1]

       p_id_2 = 2
       dtr_2 = data_train[data_train['pharmacy_id'] == p_id_2]

       p_id_3 = 3
       dtr_3 = data_train[data_train['pharmacy_id'] == p_id_3]

       p_id_4 = 4
       dtr_4 = data_train[data_train['pharmacy_id'] == p_id_4]

       dtr_0_1 = dtr_0[dtr_0['product_code'] == kod_1]
       dtr_1_1 = dtr_1[dtr_1['product_code'] == kod_1]
       dtr_2_1 = dtr_2[dtr_2['product_code'] == kod_1]
       dtr_3_1 = dtr_3[dtr_3['product_code'] == kod_1]
       dtr_4_1 = dtr_4[dtr_4['product_code'] == kod_1]

       df0 = pd.DataFrame(dtr_0_1)
       df0.drop(['type_of_product', 'product_code','pharmacy_id'], axis=1, inplace=True )
       df0['date_of_sale'] = pd.to_datetime(df0['date_of_sale'])
       df0.set_index('date_of_sale', inplace = True)
       df0_r = df0.resample('M').sum()

       df1 = pd.DataFrame(dtr_1_1)
       df1.drop(['type_of_product', 'product_code','pharmacy_id'], axis=1, inplace=True )
       df1['date_of_sale'] = pd.to_datetime(df1['date_of_sale'])
       df1.set_index('date_of_sale', inplace = True)
       df1_r = df1.resample('M').sum()

       df2 = pd.DataFrame(dtr_2_1)
       df2.drop(['type_of_product', 'product_code','pharmacy_id'], axis=1, inplace=True )
       df2['date_of_sale'] = pd.to_datetime(df2['date_of_sale'])
       df2.set_index('date_of_sale', inplace = True)
       df2_r = df2.resample('M').sum()

       df3 = pd.DataFrame(dtr_3_1)
       df3.drop(['type_of_product', 'product_code','pharmacy_id'], axis=1, inplace=True )
       df3['date_of_sale'] = pd.to_datetime(df3['date_of_sale'])
       df3.set_index('date_of_sale', inplace = True)
       df3_r = df3.resample('M').sum()

       df4 = pd.DataFrame(dtr_4_1)
       df4.drop(['type_of_product', 'product_code','pharmacy_id'], axis=1, inplace=True )
       df4['date_of_sale'] = pd.to_datetime(df4['date_of_sale'])
       df4.set_index('date_of_sale', inplace = True)
       df4_r = df4.resample('M').sum()

       p = 1  # Порядок авторегрессии (AR)
       d = 0  # Порядок разностей (d)
       q = 1  # Порядок скользящего среднего (MA)

       model_arima0 = sm.tsa.ARIMA(df0_r.quantity_sale, order=(p, d, q)).fit()
       forecast_horizon = 3
       forecast_arma0 = model_arima0.predict(start=0, end=len(df0_r.quantity_sale) + forecast_horizon)
       # Создание индекса для прогнозного периода
       forecast_index_arima0 = pd.date_range(start=df0_r.quantity_sale.index[22], periods=len(forecast_arma0))
       periods=len(forecast_arma0)
       result_arima0 = forecast_arma0[-3:].mean()*3

       model_arima1 = sm.tsa.ARIMA(df1_r.quantity_sale, order=(p, d, q)).fit()
       forecast_horizon = 3
       forecast_arma1 = model_arima1.predict(start=0, end=len(df1_r.quantity_sale) + forecast_horizon)
       forecast_index_arima1 = pd.date_range(start=df1_r.quantity_sale.index[22], periods=len(forecast_arma0))
       result_arima1 = forecast_arma1[-3:].mean()*3

       model_arima2 = sm.tsa.ARIMA(df2_r.quantity_sale, order=(p, d, q)).fit()
       forecast_horizon = 3
       forecast_arma2 = model_arima2.predict(start=0, end=len(df1_r.quantity_sale) + forecast_horizon)
       forecast_index_arima2 = pd.date_range(start=df2_r.quantity_sale.index[22], periods=len(forecast_arma0))
       result_arima2 = forecast_arma2[-3:].mean()*3

       model_arima3 = sm.tsa.ARIMA(df3_r.quantity_sale, order=(p, d, q)).fit()
       forecast_horizon = 3
       forecast_arma3 = model_arima3.predict(start=0, end=len(df1_r.quantity_sale) + forecast_horizon)
       forecast_index_arima3 = pd.date_range(start=df3_r.quantity_sale.index[22], periods=len(forecast_arma0))
       result_arima3 = forecast_arma3[-3:].mean()*3

       model_arima4 = sm.tsa.ARIMA(df4_r.quantity_sale, order=(p, d, q)).fit()
       forecast_horizon = 3
       forecast_arma4 = model_arima4.predict(start=0, end=len(df1_r.quantity_sale) + forecast_horizon)
       forecast_index_arima4 = pd.date_range(start=df4_r.quantity_sale.index[22], periods=len(forecast_arma0))
       result_arima4 = forecast_arma4[-3:].mean()*3

       ListAns = pd.DataFrame({'Филиал': ['№1', '№2', '№3','№4', '№5'], 'pharmacy_id':['0', '1', '2','3', '4'], 'forecast_sales':[result_arima0, result_arima1, result_arima2, result_arima3, result_arima4]})
       ListAnsp= ListAns[ListAns['forecast_sales'] == ListAns['forecast_sales'].max()]
       return render_template('main.html', result=ListAnsp) 
    
if __name__ == '__main__':
    app.run()

