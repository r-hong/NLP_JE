import pandas as pd
import numpy as np
import dill
import time
import re, collections
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer


#####  functions ####
def saveList2File(fname,data):
	f = open(fname, 'w')
	for x in data:
		print >>f, x
	f.close()

#get data list
def getListFromFile(fname):
    y = []
    f = open(fname, 'r')
    for line in f:
        y.append((line).rstrip())
    f.close()
    return y

def binarizeTokenList(Val):
    if TFR.count(Val)>0:
        return 1
    else:
        return 0

def binarizeReviews(cleanReviews,FV): 
    global TFR
    BRs=[]
    for i in cleanReviews:
        TFR=i
        BRs.append(map(binarizeTokenList,FV))
    return np.asarray(BRs) 

def words(text):
    return re.findall('[a-z]+', text.lower())

def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model
def edits1(word):
    s = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [a + b[1:] for a, b in s if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in s if len(b)>1]
    replaces   = [a + c + b[1:] for a, b in s for c in alphabet if b]
    inserts    = [a + c + b     for a, b in s for c in alphabet]
    return set(deletes + transposes + replaces + inserts)
def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

def known(words): 
    return set(w for w in words if w in NWORDS)

def correct(word):
    candidates = known([word]) or known(edits1(word)) or    known_edits2(word) or [word]
    return max(candidates, key=NWORDS.get)

def correct_top(word, n):
    candidates = known([word]) or known(edits1(word)) or    known_edits2(word) or [word]
    s = sorted(candidates, key=NWORDS.get, reverse=True)
    return s[0], s[:n]

# lowercase + tokenize + spellcheck + stopwords
def cleanList(textData):
    wordList=tokenizer.tokenize(textData.lower())
    for w in wordList:
        wordList[wordList.index(w)]=correct(w)
    cleaned = [w for w in wordList if not w in stopwords.words('english')]
    return cleaned	

def getReducedDataSet(df,colName,nRows):
# This function created a reduced data frame to be use in training/validation/testing
# input :
# df        --- the full dataset
# colNamed  --- the name of the column (e.g., 'delivery_time','quality','service')
# nRows     --- the number of rows per each star ([1-6]) on the output dataframe
# this function does the following:
# (1) creates a randomization by rows of df
# (2) creates a 'TMP' dataframe with the relevant columns (colName + 'commentary')
# (3) sorts 'TMP' is ascending order using 'colName'
# (4) keeps the rows in which the stars <> 0
# (5) created 6 smaller dataframes (d1...d6) comntaining 'nRows' rows for each star rating
# (6) concatenates (d1...d6) into an output final dataframe
# We can use the function to create dataframes for training with different size (larger)
# then the dataframes for validation/testing. 
# As we are randomizing the initial dataframe at the beginning of the function we can use
# this function to 'bootstrap' the initial dataframe and create a large number
# of training/validation/testing sets.
#-------------------------------------------------------------------------------
    if colName == "delivery_time":
        colNumber = 3
    if colName == "quality":
        colNumber = 4    
    if colName == "service":
        colNumber = 5    
    TMP = df.iloc[np.random.permutation(len(df))] #randomize rows in the dataframe
    TMP = TMP.iloc[:,[colNumber,6]] #get relevant columns
    TMP = TMP.sort([colName], ascending=[1]) #sort data frame 
    TMP = TMP.loc[TMP[colName] <> 0] # take only ratings <> 0

    #getting 'nRows' for each star (1,2,3,4,5,6)
    d1 = TMP.loc[TMP[colName] == 1].iloc[range(nRows),:]
    d2 = TMP.loc[TMP[colName] == 2].iloc[range(nRows),:]
    d3 = TMP.loc[TMP[colName] == 3].iloc[range(nRows),:]
    d4 = TMP.loc[TMP[colName] == 4].iloc[range(nRows),:]
    d5 = TMP.loc[TMP[colName] == 5].iloc[range(nRows),:]
    d6 = TMP.loc[TMP[colName] == 6].iloc[range(nRows),:]
    frames = [d1, d2, d3, d4, d5, d6]
    return pd.concat(frames)	

	
##### main program #####	
df=pd.read_pickle('full_data.db')

with open('TrainedWordsForSpellCheck.pkl', 'rb') as f:
    NWORDS=dill.load(f)
alphabet = 'abcdefghijklmnopqrstuvwxyz'
tokenizer = RegexpTokenizer(r'\w+')


trainingDF = getReducedDataSet(df,'quality',100000)
Y = trainingDF['quality'].tolist()
del df

start_time = time.time()
TMP           = trainingDF['commentary'].tolist()
TMP1          = [x.encode('UTF8') for x in TMP]
cleanReviews  = map(cleanList,TMP1)
print("--- %s seconds ... cleaning and encoding...---" % (time.time() - start_time))

#with open('cleanReviewsForBinarization01.pkl', 'rb') as f:
#    cleanReviews=dill.load(f)
#with open('binaryTrainingSetY02.pkl', 'rb') as f:
#    Y=dill.load(f)
print "Length of Y:", len(Y)
print "length of X:", len(cleanReviews)

with open('jeTokensWithVaderSentiments01.pkl', 'rb') as f2:
    tokens = dill.load(f2)
FV = [str(tokens[i][0]) for i in range(len(tokens))]


start_time = time.time()    
binaryTrainingSet = binarizeReviews(cleanReviews,FV)
print("--- %s seconds ... binarizing...---" % (time.time() - start_time))
print "length of the binarized X:", len(binaryTrainingSet)	
print "length of the binarized X[0]:", len(binaryTrainingSet[0])

del cleanReviews
sizeFeatureVector = len(FV)
del FV
print "free some memory: done!"

#saving
#with open('binaryTrainingSetX02.pkl', 'wb') as f1:
#    dill.dump(binaryTrainingSet, f1)
#with open('binaryTrainingSetY02.pkl', 'wb') as f2:
#    dill.dump(Y, f2)

saveList2File('binaryTrainingSetX02.flat', binaryTrainingSet)
saveList2File('binaryTrainingSetY02.flat', Y)
print "save some variables to disk: done!!"	

	
#training the NB classifier
from sklearn import svm
#from sklearn.naive_bayes import BernoulliNB

start_time = time.time()
clf = svm.SVC(decision_function_shape='ovr')
clf.fit(binaryTrainingSet, Y)
print("--- %s seconds ... model training...---" % (time.time() - start_time))

#save the classifier
with open('SVM_classifier01.pkl', 'wb') as f2:
    dill.dump(clf, f2)	

print "classifying a random vector"
zz=np.random.choice([0, 1], size=(len(binaryTrainingSet[0]),), p=[1./2, 1./2])
print(clf.predict(zz))



	
print "done!!"