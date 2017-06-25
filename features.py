import sklearn.preprocessing as pp
import numpy as np
import pandas as pd
import re


def binary_features(df):
	df=pd.concat([df,pd.get_dummies(df['Deck']).rename(columns=lambda x: 'Deck_'+str(x))],axis=1)
	df=pd.concat([df, pd.get_dummies(df['Title']).rename(columns=lambda x: 'Title_' + str(x))], axis=1)	
	return df

def simplify_fare(df):
    bins=(-1,0,7.896,14.454,31.275,512.4)
    tags=['unknown', 'first','second','third','forth']
    df['Fare']=pd.cut(df['Fare'], bins, tags)
    return df	

def simplify_ages(df):
    bins=(-1,0,5,12,18,25,35,60,120)
    tags = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    df['Age'] = pd.cut(df['Age'],bins , labels=tags) 
    return df


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""


def fill_features(df):
    df['Embarked'].fillna('S',inplace=True)
    df['Fare'].fillna(df['Fare'].median(),inplace=True)
    mu=df['Age'].mean()
    delta=df['Age'].std()
    rand_list=np.random.randint(mu-delta,mu+delta,size=df['Age'].isnull().sum())
    df.loc[df['Age'].isnull(), 'Age']=rand_list
    df['Age']=df['Age'].astype(int)
	df['Cabin'].fillna('U0')
    return df

def add_features(df):
    df['FamilySize']=df['SibSp']+df['Parch']+1
    df['IsAlone']=1
    df.loc[df['FamilySize']>1, 'IsAlone']=0
    df['HasCabin']=df['Cabin'].apply(lambda x: 0 if type(x)==float else 1)
	df['Deck']=total['Cabin'].apply(lambda x: x[0])
	df=extract_ticket(df)
    return df

def simplify_features(df):
    df=simplify_fare(df)
    df=simplify_ages(df)
    df=simplify_title(df)
    return df

def drop_features(df):
    df.drop('PassengerId',axis=1,inplace=True)
    df.drop('Name',axis=1,inplace=True)
    #df.drop('Cabin',axis=1,inplace=True)
    #df.drop('SibSp',axis=1,inplace=True)
    #total.drop('Parch',axis=1,inplace=True)
    df.drop('Ticket',axis=1,inplace=True)
    return df

def encode_features(df):
    features = ['Fare', 'Age', 'Embarked', 'Sex', 'Title']
    for feature in features:
        le=pp.LabelEncoder()
        le=le.fit(df[feature])
        df[feature]=le.transform(df[feature])
    return df

def extract_ticket(df):
    
    # extract and massage the ticket prefix
    df['TicketPrefix'] = df['Ticket'].map( lambda x : getTicketPrefix(x.upper()))
    df['TicketPrefix'] = df['TicketPrefix'].map( lambda x: re.sub('[.?/?]', '', x) )
    df['TicketPrefix'] = df['TicketPrefix'].map( lambda x: re.sub('STON', 'SOTON', x) )
        
    # create binary features for each prefix
    #prefixes = pd.get_dummies(df['TicketPrefix']).rename(columns=lambda x: 'TicketPrefix_' + str(x))
    #df = pd.concat([df, prefixes], axis=1)
    
    # factorize the prefix to create a numerical categorical variable
    df['TicketPrefixId'] = pd.factorize(df['TicketPrefix'])[0]
    
    # extract the ticket number
    df['TicketNumber'] = df['Ticket'].map( lambda x: getTicketNumber(x) )
    
    # create a feature for the number of digits in the ticket number
    df['TicketNumberDigits'] = df['TicketNumber'].map( lambda x: len(x) ).astype(np.int)
    
    # create a feature for the starting number of the ticket number
    df['TicketNumberStart'] = df['TicketNumber'].map( lambda x: x[0:1] ).astype(np.int)
    
    # The prefix and (probably) number themselves aren't useful
    df.drop(['TicketPrefix', 'TicketNumber'], axis=1, inplace=True)
	return df
    
 
def getTicketPrefix(ticket):
    match = re.compile("([a-zA-Z./]+)").search(ticket)
    if match:
        return match.group()
    else:
        return 'U'
 
def getTicketNumber(ticket):
    match = re.compile("([d]+$)").search(ticket)
    if match:
        return match.group()
    else:
        return '0'

