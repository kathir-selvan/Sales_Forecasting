# Import  packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



# load dataset
data = pd.read_csv("sales_data.csv")


#Display the dataset
print("Sales Data :\n", data)

#convert Month to a numeric index for forecasting

data['Month_No'] = np.arange(1,len(data) + 1)


#Basisc Sales Analysis
print("\nAverage Sales:", data['Sales'].mean())
print("Highest Sales:", data['Sales'].max())
print("Lowesst Sales:", data['Sales'].min())

#plot Sales trend
plt.figure(figsize=(8,5))
plt.plot(data['Month'],data['Sales'], marker='o', label = 'Actual Sales')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.grid(True)
plt.legend()
plt.show()

#Linear Regression 
from sklearn.linear_model import LinearRegression

x = data[['Month_No']]
y = data ['Sales']


model = LinearRegression()
model.fit(x,y)
#predicet nxt  motns

future_months = np.array([13],[14],[15])
predicted_sales  = model.predict(future_months)


print("\nForecast for next 3 months:")
for i , val in enumerate(predicted_sales,start=1):
    print(f"Month {12+i}: {val:.2f}")


#viaializze forecast

plt.figure(figsize=(8,5))
plt.plot(data['Month_No'], data['Sales'], 'bo-',label='Actual Sales')    
plt.plot(future_months, predicted_sales, 'r*-',label='Forecast')
plt.title("Sales Forecasting")
plt.xlabel("Month NUmber")
plt.ylabel("Sales")
plt.legend()
plt.grid(True)
plt.show()