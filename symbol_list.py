import pandas as pd
import os
# get by 'Name' and 'Symbol'
os.chdir('stock_recommendation/src')
df_nasdaq = pd.read_csv('nasdaq_screener_1751610474960.csv')
df_xams = pd.read_csv('Euronext_Equities_XAMS.csv',
    delimiter=';',
)
df_xpar= pd.read_csv('Euronext_Equities_XPAR.csv',
    delimiter=';',
)
df_xams['Symbol'] = df_xams['Symbol'].astype(str) + '.AS'
df_xpar['Symbol'] = df_xpar['Symbol'].astype(str) + '.PA'

df = pd.concat([df_nasdaq[['Symbol', 'Name']], df_xams[['Symbol', 'Name']], df_xpar[['Symbol', 'Name']]], ignore_index=True)
df.to_csv('symbol.csv', index=False)