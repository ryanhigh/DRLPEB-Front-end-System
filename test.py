import pandas as pd
import numpy as np

df = pd.read_csv('static/rewards.csv')
df = df.iloc[:, 1:]
xaxis_data=df.iloc[:2,0].tolist()
y_axis=df.iloc[:2,2].tolist()

if __name__ == '__main__':
    print(xaxis_data, y_axis)