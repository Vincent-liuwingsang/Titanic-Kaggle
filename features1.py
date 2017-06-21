import sklearn.preprocessing as pp
import numpy as np
import pandas as pd
import re

def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

def add_title(df):
    df['Title'] = df['Name'].apply(get_title)
    df['Title'] = df['Title'].replace(['Lady', 'Dona', 'the Countess', 'Sir', 'Don', 'Jonkheer'], 'Royalty')
    df['Title'] = df['Title'].replace(['Capt','Col','Major','Dr','Rev'], 'Officer')
    df['Title'] = df['Title'].replace(['Mlle','Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    return df

def fillAges(row, grouped_median):
    if row['Sex']=='female' and row['Pclass'] == 1:
        if row['Title'] == 'Miss':
            return grouped_median.loc['female', 1, 'Miss']['Age']
        elif row['Title'] == 'Mrs':
            return grouped_median.loc['female', 1, 'Mrs']['Age']
        elif row['Title'] == 'Officer':
            return grouped_median.loc['female', 1, 'Officer']['Age']
        elif row['Title'] == 'Royalty':
            return grouped_median.loc['female', 1, 'Royalty']['Age']

    elif row['Sex']=='female' and row['Pclass'] == 2:
        if row['Title'] == 'Miss':
            return grouped_median.loc['female', 2, 'Miss']['Age']
        elif row['Title'] == 'Mrs':
            return grouped_median.loc['female', 2, 'Mrs']['Age']

    elif row['Sex']=='female' and row['Pclass'] == 3:
        if row['Title'] == 'Miss':
            return grouped_median.loc['female', 3, 'Miss']['Age']
        elif row['Title'] == 'Mrs':
            return grouped_median.loc['female', 3, 'Mrs']['Age']

    elif row['Sex']=='male' and row['Pclass'] == 1:
        if row['Title'] == 'Master':
            return grouped_median.loc['male', 1, 'Master']['Age']
        elif row['Title'] == 'Mr':
            return grouped_median.loc['male', 1, 'Mr']['Age']
        elif row['Title'] == 'Officer':
            return grouped_median.loc['male', 1, 'Officer']['Age']
        elif row['Title'] == 'Royalty':
            return grouped_median.loc['male', 1, 'Royalty']['Age']

    elif row['Sex']=='male' and row['Pclass'] == 2:
        if row['Title'] == 'Master':
            return grouped_median.loc['male', 2, 'Master']['Age']
        elif row['Title'] == 'Mr':
            return grouped_median.loc['male', 2, 'Mr']['Age']
        elif row['Title'] == 'Officer':
            return grouped_median.loc['male', 2, 'Officer']['Age']

    elif row['Sex']=='male' and row['Pclass'] == 3:
        if row['Title'] == 'Master':
            return grouped_median.loc['male', 3, 'Master']['Age']
        elif row['Title'] == 'Mr':
            return grouped_median.loc['male', 3, 'Mr']['Age']

    
def fill_age(df):
	grouped_train = df.iloc[:891].groupby(['Sex','Pclass','Title'])
	grouped_test = df.iloc[891:].groupby(['Sex','Pclass','Title'])
	grouped_median_train = grouped_train.median()
	grouped_median_test = grouped_test.median()
	df.iloc[:891].Age = df.iloc[:891].apply(lambda r : fillAges(r, grouped_median_train) if np.isnan(r['Age']) else r['Age'], axis=1)
	df.iloc[891:].Age = df.iloc[891:].apply(lambda r : fillAges(r, grouped_median_test) if np.isnan(r['Age']) else r['Age'], axis=1)
	return df

def fill_fare(df):
	df.iloc[:891].Fare.fillna(df.iloc[:891].Fare.mean(), inplace=True)
	df.iloc[891:].Fare.fillna(df.iloc[891:].Fare.mean(), inplace=True)
	return df

def fill_embarked(df):
	df['Embarked'].fillna('S', inplace=True)
	return df

def fill_cabin(df):
	df.Cabin.fillna('U', inplace=True)
	df['Cabin'] = df['Cabin'].apply(lambda c : c[0])
	return df

def cleanTicket(ticket):
	ticket = ticket.replace('.','')
	ticket = ticket.replace('/','')
	ticket = ticket.split()
	ticket = map(lambda t : t.strip(), ticket)
	ticket = filter(lambda t : not t.isdigit(), ticket)
	if len(ticket) > 0:
		return ticket[0]
	else: 
		return 'XXX'

def fill_ticket(df):
	df['Ticket'] = df['Ticket'].map(cleanTicket) 
	return df



def add_family(df):
	df['FamilySize'] = df['Parch'] + df['SibSp'] + 1
	df['Singleton'] = df['FamilySize'].map(lambda s: 1 if s == 1 else 0)
	df['SmallFamily'] = df['FamilySize'].map(lambda s: 1 if 2<=s<=4 else 0)
	df['LargeFamily'] = df['FamilySize'].map(lambda s: 1 if 5<=s else 0)
	return df
    

