import lifelines
import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_gbsg2(df=False):
    data = lifelines.datasets.load_gbsg2()
    data_encoded = pd.get_dummies(data, columns=['horTh', 'menostat', 'tgrade'])
    cols_to_scale = [ 'age', 'tsize', 'pnodes', 'progrec','estrec']
    scaler = StandardScaler()
    scaler.fit(data_encoded[cols_to_scale])

    data_encoded[cols_to_scale] = scaler.transform(data_encoded[cols_to_scale])
    covariates = data_encoded.drop(['time', 'cens'],axis = 1).values
    event_times = data['time'].values
    event_indicators = data['cens'].values
    
    if df==True:
        return data_encoded
    else:
        return covariates, event_times, event_indicators


