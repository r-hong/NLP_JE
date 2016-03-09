# testing the trained BernouliNB model on fake (random) data
import numpy as np
import pandas as pd
import dill
import random
import time
import re, collections
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
plt.rc('xtick', labelsize=30)
plt.rc('ytick', labelsize=30)
axis_font = {'size':'30'}

###### functions #######
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

def convPosNeg(vector):
#convert a vector of ratings (1...6) to a vector of neg and pos entries
	back=[]
	for i in range(len(vector)):
		if vector[i]<=3:
			back.append('neg')
		elif vector[i]>=4:
			back.append('pos')
	return back
	
def convPosNegNeu(vector):
#convert a vector of ratings (1...6) to a vector of neg, neu, and pos entries
	back=[]
	for i in range(len(vector)):
		if vector[i]<=2:
			back.append('neg')
		elif vector[i]>=5:
			back.append('pos')
		else:
			back.append('neu')
	return back	
	
	
#######################################################
####                    MAIN                      #####
#######################################################
#get the full database	
df=pd.read_pickle('full_data.db')

#get a reduced version of the dataset
testDF = getReducedDataSet(df,'quality',100)
Yhat = testDF['quality'].tolist()
del df



#clean and binarize the reduced dataset
with open('TrainedWordsForSpellCheck.pkl', 'rb') as f:
    NWORDS=dill.load(f)
alphabet = 'abcdefghijklmnopqrstuvwxyz'
tokenizer = RegexpTokenizer(r'\w+')

start_time = time.time()
TMP           = testDF['commentary'].tolist()
TMP1          = [x.encode('UTF8') for x in TMP]
cleanReviews  = map(cleanList,TMP1)
print("--- %s seconds ... cleaning and encoding...---" % (time.time() - start_time))

#load my feature vector
with open('jeTokensWithVaderSentiments01.pkl', 'rb') as f2:
    tokens = dill.load(f2)
FV = [str(tokens[i][0]) for i in range(len(tokens))]

#binarize my test data using my feature vector
start_time = time.time()    
binaryTestSet = binarizeReviews(cleanReviews,FV)
print("--- %s seconds ... binarizing...---" % (time.time() - start_time))
print "length of the binarized X:", len(binaryTestSet)	
print "length of the binarized X[0]:", len(binaryTestSet[0])
	
	
#get the saved classifier	
with open('BernoulliNB_classifier02.pkl', 'rb') as f2:
    bernouNB = dill.load(f2)
Y = bernouNB.predict(binaryTestSet)

#coversion to vector with (pos,neg) and (pos,neu,neg)
Yhat_PosNeg = convPosNeg(Yhat)	
Y_PosNeg    = convPosNeg(Y)
Yhat_PosNegNeu = convPosNegNeu(Yhat)	
Y_PosNegNeu    = convPosNegNeu(Y)


#confusion matrix
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=36)
    plt.colorbar()
    tick_marks = np.arange(6)
    plt.xticks(tick_marks, ['1','2','3','4','5','6'], rotation=45)
    plt.yticks(tick_marks, ['1','2','3','4','5','6'])
    plt.tight_layout()
    plt.ylabel('True label',**axis_font)
    plt.xlabel('Predicted label',**axis_font)

cm = confusion_matrix(Yhat, Y)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure()
plot_confusion_matrix(cm_normalized)
plt.show()

###########################################################################
print "---------------------------------------------------------"	
print "Precision/Recall/F1 (good, bad)"
print (metrics.precision_score(Yhat_PosNeg, Y_PosNeg, average=None), np.mean(metrics.precision_score(Yhat_PosNeg, Y_PosNeg, average=None)) )
print "..."
print (metrics.recall_score(Yhat_PosNeg, Y_PosNeg, average=None), np.mean(metrics.recall_score(Yhat_PosNeg, Y_PosNeg, average=None)) )
print "..."
print (metrics.f1_score(Yhat_PosNeg, Y_PosNeg, average=None), np.mean(metrics.f1_score(Yhat_PosNeg, Y_PosNeg, average=None)) )
print "..."
print "---------------------------------------------------------"	
print "Precision/Recall/F1 (good, neutral, bad)"
print (metrics.precision_score(Yhat_PosNegNeu, Y_PosNegNeu, average=None), np.mean(metrics.precision_score(Yhat_PosNegNeu, Y_PosNegNeu, average=None)) )
print "..."
print (metrics.recall_score(Yhat_PosNegNeu, Y_PosNegNeu, average=None), np.mean(metrics.recall_score(Yhat_PosNegNeu, Y_PosNegNeu, average=None)) )
print "..."
print (metrics.f1_score(Yhat_PosNegNeu, Y_PosNegNeu, average=None), np.mean(metrics.f1_score(Yhat_PosNegNeu, Y_PosNegNeu, average=None)) )
print "..."
print "---------------------------------------------------------"	
print "Precision/Recall/F1 (1, 2, 3, 4, 5, 6)"
print (metrics.precision_score(Yhat, Y, average=None), np.mean(metrics.precision_score(Yhat, Y, average=None)) )
print "..."
print (metrics.recall_score(Yhat, Y, average=None), np.mean(metrics.recall_score(Yhat, Y, average=None)) )
print "..."
print (metrics.f1_score(Yhat, Y, average=None), np.mean(metrics.f1_score(Yhat, Y, average=None)) )
print "..."
print "---------------------------------------------------------"


