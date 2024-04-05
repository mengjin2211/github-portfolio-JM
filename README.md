# Data Engineer Manager

## Work Experience
Data Quality Manager       Policy Reporter 						2022-now \
Manager of Data Analytics  LandSure Systems, Land Title& Survey Authority of BC 	2021-2022 \
Manager of Records         Simon Fraser University 				2020-2021 \
Senior Programs Specialist University of Calgary 					2017-2020 \
Program Coordinator        University of Calgary 					2013-2017 
 
## Education and Certifications
•	MA (U of Electronic Science & Tech) \
•	Business Intelligence & Analytics Certificate (University of Calgary) \
•	Database Administration Certificate (University of Calgary) \
•	IBM Data Science Professional Certificate \
•	IBM Data Engineering Certificate \
•	Microsoft Azure DP-900 \
•	Microsoft Azure AZ-900  \
•	Machine Learning Certificate \
•	IBM Data Warehouse Engineer Certificate 

## Projects
### Machine Learning Projects
![Time-Series Analysis of Influxes Using Prophet Machine Learning Model](/assets/Time Series Analysis ML Model.png)
```python
def create_prophet_model(df):
     
    changepoint_prior_scale_values = 20
    seasonality_prior_scale_values = 3 

    model = Prophet(
        changepoint_prior_scale=changepoint_prior_scale_values,
        seasonality_prior_scale=seasonality_prior_scale_values)
    model.add_seasonality(name='monthly', period=4, fourier_order=10)
    model.add_seasonality(name='quarterly', period=13, fourier_order=9)
         
    np.random.seed(0)
    model.fit(df)
    return model, df

def future_df(model, weeks):
    initial_df = model.history.copy()
    week = weeks+ 50
    future = model.make_future_dataframe(periods=week, freq='W')
    if 'week' in locals():
        print(f"Week value exists and is: {week}")
    return future

def predict_prophet(weeks): 
    global df2
    model, df2 = create_prophet_model(df2)  # Make sure df2 is defined
    test_data = future_df(model, weeks)
    prediction = model.predict(test_data)[['ds', 'yhat']]
    prediction['yhat'] = prediction['yhat'].round(0).astype(int)
 
    today_date = datetime.now().strftime('%Y-%m-%d')
    filter_prediction = prediction[prediction['ds'] > today_date]
    filter_prediction=filter_prediction.rename(columns={'ds': 'Week', 'yhat': 'Influx'}).to_string(index=False)
   
    return filter_prediction

def image(weeks):
    import matplotlib.pyplot as plt
    import base64
    from io import BytesIO
    fig, ax = plt.subplots(figsize=(6,3))
    pred=predict_prophet(weeks)
    ax.plot(df2['ds'], df2['y'], label='Training Data', color='blue')
    ax.plot(pred['ds'], pred['yhat'], label='Predict', color='red')
    plt.xlabel('Time')
    plt.ylabel('Influx')
    plt.title('Prophet Influx Forecast')
    plt.legend(loc='best')
    plt.show()
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    # img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')  
    return pred, img_base64

•	NLP Project
### Report Automation
•	Time-Series Analysis of Influxes Using Prophet Machine Learning Model
•	NLP Project
### SQL Queries
•	Time-Series Analysis of Influxes Using Prophet Machine Learning Model
•	NLP Project
