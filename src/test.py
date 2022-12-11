import pandas as pd
dir=r"data/train.xlsx"
data = pd.read_excel(io=dir, sheet_name=1,header=0)
pd.set_option('display.max_rows',None)
f = open("test.txt", 'w')
need_=data.iloc[(i for i in range(data.shape[0]) if (i % 24 == 2)),:]
print(need_, file= f)