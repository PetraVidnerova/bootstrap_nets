import numpy as np
import pandas as pd



x = np.linspace(0, 1, 1000)
xx = 2*x-1
y = np.sin(abs(10*xx))*abs(10*xx)



df = pd.DataFrame()
df["X"] = x
df["Y"] = y

print(df)

df.to_csv("data_sin.csv", index=False)
