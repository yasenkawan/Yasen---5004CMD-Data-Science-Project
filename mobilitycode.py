# ==============================
# 1. IMPORT LIBRARIES
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

from multiprocessing import Pool

# ==============================
# 2. LOAD DATA
# ==============================
df = pd.read_csv("Trips_by_Distance.csv")

# Optional second dataset
df_full = pd.read_csv("Trips_Full_Data.csv")

# ==============================
# 3. DATA PREPROCESSING
# ==============================
df.dropna(inplace=True)

# Convert date column (if exists)
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])

# ==============================
# 4. QUESTION (a)
# ==============================
# People staying at home per week
start = time.time()

home_weekly = df.groupby('Week')['Population Staying at Home'].sum()

end = time.time()
print("Sequential Time (Q1):", end - start)

# Plot
home_weekly.plot(kind='line', title="People Staying at Home per Week")
plt.xlabel("Week")
plt.ylabel("Population")
plt.show()

# Distance travelled by people not staying home
travel_cols = [col for col in df.columns if "Trips" in col]

distance_travel = df[travel_cols].sum()

distance_travel.plot(kind='bar', title="Distance Travel Distribution")
plt.xticks(rotation=45)
plt.show()

# ==============================
# 5. QUESTION (b)
# ==============================
# Filter data
df_10_25 = df[df['Trips 10-25'] > 10000000]
df_50_100 = df[df['Trips 50-100'] > 10000000]

# Scatter plot
plt.scatter(df_10_25.index, df_10_25['Trips 10-25'], label='10-25 Trips')
plt.scatter(df_50_100.index, df_50_100['Trips 50-100'], label='50-100 Trips')

plt.legend()
plt.title("Trip Comparison Scatter Plot")
plt.xlabel("Index")
plt.ylabel("Trips")
plt.show()

# ==============================
# 6. QUESTION (c) - MODELLING
# ==============================
# Example: use one distance column
X = df[['Trips 1-3']]   # independent variable
y = df['Trips 10-25']   # dependent variable

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print("RMSE:", rmse)
print("R2:", r2)

# Scatter plot with regression line
plt.scatter(X, y, label="Actual")
plt.plot(X, y_pred, color='red', label="Predicted")
plt.legend()
plt.title("Regression Model")
plt.xlabel("Distance")
plt.ylabel("Trips")
plt.show()

# ==============================
# 7. QUESTION (d)
# ==============================
# Total travellers by distance
distance_sum = df[travel_cols].sum()

distance_sum.plot(kind='bar', title="Travellers by Distance")
plt.xticks(rotation=45)
plt.show()

# ==============================
# 8. SEQUENTIAL PROCESSING
# ==============================
start = time.time()

seq_result = df.groupby('Week')['Population Staying at Home'].sum()

end = time.time()
print("Sequential Execution Time:", end - start)

# ==============================
# 9. PARALLEL PROCESSING FUNCTION
# ==============================
def process_chunk(chunk):
    return chunk.groupby('Week')['Population Staying at Home'].sum()

# ==============================
# 10. PARALLEL (10 CORES)
# ==============================
chunks = np.array_split(df, 10)

start = time.time()

with Pool(10) as p:
    results = p.map(process_chunk, chunks)

parallel_10 = pd.concat(results).groupby(level=0).sum()

end = time.time()
print("Parallel Time (10 cores):", end - start)

# ==============================
# 11. PARALLEL (20 CORES)
# ==============================
chunks = np.array_split(df, 20)

start = time.time()

with Pool(20) as p:
    results = p.map(process_chunk, chunks)

parallel_20 = pd.concat(results).groupby(level=0).sum()

end = time.time()
print("Parallel Time (20 cores):", end - start)

# ==============================
# 12. COMPARISON OUTPUT
# ==============================
print("\nComparison Summary:")
print("Sequential:", seq_result.head())
print("Parallel (10):", parallel_10.head())
print("Parallel (20):", parallel_20.head())