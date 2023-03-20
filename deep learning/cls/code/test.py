import pandas as pd

a = pd.DataFrame()
a = a.append({"a": 1, "b": 1e-1}, ignore_index=True)
print(a)