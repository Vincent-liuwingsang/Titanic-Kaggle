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

#buggy   
def fill_age(df):
	grouped_train = df.iloc[:891].groupby(['Sex','Pclass','Title'])
	grouped_test = df.iloc[891:].groupby(['Sex','Pclass','Title'])
	grouped_median_train = grouped_train.median()
	grouped_median_test = grouped_test.median()
	df.info()
	df.loc[:891,('Age')] = df.iloc[:891].apply(lambda r : r['Age'] if isinstance(r['Age'], float) else 999.0, axis=1)
	df.loc[891:,('Age')] = df.iloc[891:].apply(lambda r : fillAges(r, grouped_median_test) if np.isnan(r['Age']) else r['Age'], axis=1)
	df.info()
	return df

def fill_age_1(df):
	grouped_train = df.groupby(['Sex','Pclass','Title'])
	grouped_median_train = grouped_train.median()
	df['Age'] = df.apply(lambda r : fillAges(r, grouped_median_train) if np.isnan(r['Age']) else r['Age'],axis=1)
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
	df['Cabin'] = df['Cabin'].apply(lambda c : 1 if c[0]=='U' else 0)
	return df

def ticketAlpha(ticket):
	ticket = ticket.replace('.','')
	ticket = ticket.replace('/','')
	ticket = ticket.split()
	ticket = map(lambda t : t.strip(), ticket)
	ticket = filter(lambda t : not t.isdigit(), ticket)
	if len(ticket) > 0:
		return ticket[0]
	else: 
		return 'XXX'
def ticketNum(ticket):
	ticket = ticket.replace('.','')
	ticket = ticket.replace('/','')
	ticket = ticket.replace('[a-zA-Z]','')
	ticket = ticket.split()
	return str(ticket[len(ticket)-1])[0]


def fill_ticket(df):
	df['TicketAlpha'] = df['Ticket'].map(ticketAlpha)
	df['TicketNum'] = df['Ticket'].map(ticketNum)
	df['TicketNum'] = df['TicketNum'].convert_objects(convert_numeric=True)
	return df



def add_family(df):
	df['FamilySize'] = df['Parch'] + df['SibSp'] + 1
	df['Singleton'] = df['FamilySize'].map(lambda s: 1 if s == 1 else 0)
	df['SmallFamily'] = df['FamilySize'].map(lambda s: 1 if 2<=s<=4 else 0)
	df['LargeFamily'] = df['FamilySize'].map(lambda s: 1 if 5<=s else 0)
	return df

def simplify_ages_intuition(df):
	bins=(-1,0,5,12,18,25,35,60,120)
	tags = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
	df['Age'] = pd.cut(df['Age'],bins , labels=tags) 
	return df

def simplify_ages_bins(df,b):
	tags=range(b)
	df['Age'] = pd.cut(df['Age'],bins=b,labels=tags) 
	return df
def add_share(total):
	FareFreq=total.Fare.value_counts()
	TicketFreq=total.Ticket.value_counts()
	CabinFreq=total.Cabin.value_counts()
	total['ShareFare'] = total['Fare'].apply(lambda x: FareFreq[x] if FareFreq[x]>1 else 0)
	total['ShareTicket'] = total['Ticket'].apply(lambda x: TicketFreq[x] if TicketFreq[x]>1 else 0)
	total['ShareCabin'] = total['Cabin'].apply(lambda x: CabinFreq[x] if CabinFreq[x]>1 else 0)

