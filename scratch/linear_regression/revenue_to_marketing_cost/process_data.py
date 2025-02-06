import pandas as pd # type: ignore
def data_columns():

    df=pd.read_csv('revenue-marketing.csv')
    
    return df['TV Ad Budget ($)'].to_numpy(),df['Sales ($)'].to_numpy()
