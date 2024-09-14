import pandas as pd
from prophet import Prophet

# Load the actual dataset
df = pd.read_csv(r'C:\Users\hp\Desktop\Thesis\1804010_Thunderstorm&Lightning Dataset.csv')
df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
df = df.set_index('Date').groupby('District').resample('D').mean()
df = df.reset_index()
df.fillna(0, inplace=True)

# Define columns to forecast (excluding 'Lightning Events')
columns_to_forecast = df.columns.difference(['Date', 'District', 'Lightning Events'])

# DataFrame to store the forecasted values
forecast_df = pd.DataFrame()

# Generate forecast for each District and each parameter
for District in df['District'].unique():
    district_data = df[df['District'] == District].copy()
    district_forecast = pd.DataFrame({'Date': pd.date_range(start=district_data['Date'].max(), periods=1095, freq='D')[1:]})
    
    for column in columns_to_forecast:
        district_data_rename = district_data.rename(columns={'Date': 'ds', column: 'y'})
        m = Prophet(interval_width=0.95)
        m.fit(district_data_rename[['ds', 'y']])
        
        # Make future dataframe for next three years (1095 days)
        future = m.make_future_dataframe(periods=1095)
        forecast = m.predict(future)
        
        # Merge forecast with district_forecast DataFrame
        district_forecast = district_forecast.merge(forecast[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': column + '_Forecast'}), on='Date', how='left')
    
    # Add District, Latitude, and Longitude information
    district_forecast['District'] = District
    district_forecast['Latitude'] = district_data['Latitude'].iloc[0]
    district_forecast['Longitude'] = district_data['Longitude'].iloc[0]
    
    # Append district forecast to forecast_df
    forecast_df = pd.concat([forecast_df, district_forecast], ignore_index=True)

# Save the forecasted data to a new CSV file
forecast_df.to_csv(r'C:\Users\hp\Desktop\Thesis\Forecasted_Data_All_Parameters1.csv', index=False)

print("Forecasting complete. Forecasted data saved to 'Forecasted_Data_All_Parameters.csv'.")
forecast_df.to_json(r'C:\Users\hp\Desktop\Thesis\Forecasted_Data_All_Parameters1.json', orient='records', date_format='iso')

print("Forecasting complete. Forecasted data saved to 'Forecasted_Data_All_Parameters.json'.")







