import pandas as pd

def engineer(df):
    df = df.copy()

    df['priority_score'] = df['priority'].map({'Low':1,'Medium':2,'High':3}).fillna(2)

    df['time_used'] = df['estimated_time'] - df['time_remaining']
    df['time_ratio'] = df['time_remaining'] / (df['estimated_time'] + 1)
    df['completion_percentage'] = (df['time_used']/(df['estimated_time']+1))*100

    df['urgency_score'] = df['priority_score']*(1/(df['time_ratio']+0.01))
    df['risk_score'] = df['priority_score']*(1-df['time_ratio'])

    df['tight_deadline'] = (df['time_remaining'] < 20).astype(int)
    df['very_tight'] = (df['time_remaining'] < 10).astype(int)

    return df