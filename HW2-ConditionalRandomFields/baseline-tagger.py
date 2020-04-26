###############
### IMPORTS ###
###############

from collections import namedtuple
import csv
import glob
import os
import sys
import pycrfsuite
import collections

############
### DEFS ###
############

#!/usr/bin/env python3

"""hw2_corpus_tools.py: CSCI544 Homework 2 Corpus Code

USC Computer Science 544: Applied Natural Language Processing

Provides three functions and two data containers:
get_utterances_from_file - loads utterances from an open csv file
get_utterances_from_filename - loads utterances from a filename
get_data - loads all the CSVs in a directory
DialogUtterance - A namedtuple with various utterance attributes
PosTag - A namedtuple breaking down a token/pos pair

Feel free to import, edit, copy, and/or rename to use in your assignment.
Do not distribute.

Written in 2015 by Christopher Wienberg.
Questions should go to your instructor/TAs.
"""


def get_utterances_from_file(dialog_csv_file):
    """Returns a list of DialogUtterances from an open file."""
    reader = csv.DictReader(dialog_csv_file)
    return [_dict_to_dialog_utterance(du_dict) for du_dict in reader]

def get_utterances_from_filename(dialog_csv_filename):
    """Returns a list of DialogUtterances from an unopened filename."""
    with open(dialog_csv_filename, "r") as dialog_csv_file:
        return get_utterances_from_file(dialog_csv_file)

def get_data(data_dir):
    """Generates lists of utterances from each dialog file.

    To get a list of all dialogs call list(get_data(data_dir)).
    data_dir - a dir with csv files containing dialogs"""
    dialog_filenames = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    for dialog_filename in dialog_filenames:
        yield get_utterances_from_filename(dialog_filename)

DialogUtterance = namedtuple(
    "DialogUtterance", ("act_tag", "speaker", "pos", "text"))

DialogUtterance.__doc__ = """\
An utterance in a dialog. Empty utterances are None.

act_tag - the dialog act associated with this utterance
speaker - which speaker made this utterance
pos - a list of PosTag objects (token and POS)
text - the text of the utterance with only a little bit of cleaning"""

PosTag = namedtuple("PosTag", ("token", "pos"))

PosTag.__doc__ = """\
A token and its part-of-speech tag.

token - the token
pos - the part-of-speech tag"""

def _dict_to_dialog_utterance(du_dict):
    """Private method for converting a dict to a DialogUtterance."""

    # Remove anything with 
    for k, v in du_dict.items():
        if len(v.strip()) == 0:
            du_dict[k] = None

    # Extract tokens and POS tags
    if du_dict["pos"]:
        du_dict["pos"] = [
            PosTag(*token_pos_pair.split("/"))
            for token_pos_pair in du_dict["pos"].split()]
    return DialogUtterance(**du_dict)


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
    ]
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')
                
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]  


##############################
########## TRAINING ##########
##############################



training_dir = sys.argv[1]
dev_dir = sys.argv[2]
result_dir = sys.argv[3]

du_list = list(get_data(training_dir)) # gets data from 75% of files

features = []
labels = []
    
for file in du_list:
    previousSpeaker = file[0].speaker
    firstUtterance = file[0].text
    
    for utterance in file:
        myFeatures = []
        if utterance == None:
            continue
        
        labels.append(utterance.act_tag)
        #print(utterance.text)
        
        # feature 1: check whether speaker has changed
        if utterance.speaker == previousSpeaker:
            speakerStatus = 'same-speaker'
        else:
            speakerStatus = 'new-speaker'
        
        previousSpeaker = utterance.speaker # update previousSpeaker tracker
        
        myFeatures.append(speakerStatus)
        
        # feature 2: check whether this is the first utterance
        if utterance.text == firstUtterance:
            firstStatus = 'first'
        else:
            firstStatus = 'not-first'

        myFeatures.append(firstStatus)
        
        # feature 3: all tokens in the utterance
        # feature 4: all POS tags in the utterance
        myTokens = []
        myPOSTags = []
        if utterance.pos == None:
            myFeatures.append('NO_WORD')
            myFeatures.append('NO_TOKEN')
        else:
            for word in utterance.pos:
                myFeatures.append(word[0])
                myFeatures.append(word.pos)
            
        features.append(myFeatures)
        

myTrainer = pycrfsuite.Trainer(verbose=False) # initialize new trainer

myTrainer.set_params({
    'c1': 1.0, #coefficient for L1 penalty
    'c2': 1e-3, #coefficient for L2 penalty
    'max_iterations': 50, # stop earlier
    
    'feature.possible_transitions': True # include transitions that are possible, but not observed
})
    

myTrainer.append(features, labels)
model_dir = os.getcwd() + '/model.txt'
myTrainer.train(model_dir)

#################################
#### FORMAT INPUT TO TAGGER #####
#################################


tagger = pycrfsuite.Tagger()

test_list = list(get_data(dev_dir)) # gets data from 75% of files

open(result_dir, 'w').close() # clear the file

f = open(result_dir,"w+")
tagger.open(model_dir)
numLinesPrinted = 0

for file in test_list:
    previousSpeaker = file[0].speaker
    firstUtterance = file[0].text
    file_features = []
    
    for utterance in file:
        myFeatures = []
        if utterance == None:
            #print('NO_WORD')
            #f.write('NO_WORD\n')
            continue

        
        # feature 1: check whether speaker has changed
        if utterance.speaker == previousSpeaker:
            speakerStatus = 'same-speaker'
        else:
            speakerStatus = 'new-speaker'
        
        previousSpeaker = utterance.speaker # update previousSpeaker tracker
        
        myFeatures.append(speakerStatus)
        
        # feature 2: check whether this is the first utterance
        if utterance.text == firstUtterance:
            firstStatus = 'first'
        else:
            firstStatus = 'not-first'

        myFeatures.append(firstStatus)
        
        # feature 3: all tokens in the utterance
        # feature 4: all POS tags in the utterance
        myTokens = []
        myPOSTags = []
        if utterance.pos == None:
            myFeatures.append('NO_WORD')
            myFeatures.append('NO_TOKEN')
        else:
            for word in utterance.pos:
                myFeatures.append(word[0])
                myFeatures.append(word.pos)
            
        file_features.append(myFeatures)
        
        
    taggerResults = tagger.tag(file_features) # results of the entire dialogue file
    for each in taggerResults:
        #print(each)
        f.write(each + '\n')
    if not file == test_list[-1]:
        #print(' ') # spacing between files
        f.write('\n')
f.close()


