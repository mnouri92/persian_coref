from __future__ import unicode_literals
from hazm import *
from collections import deque

#from importlib import reload
import os
import pandas as pd
import sys
import csv


columns = ['Document ID', 'Part number', 'Word number', 'Word', 'Part of speech', 'Parse bit', 'Predicate lemma',
           'Predicate Frameset ID', 'Word sence', 'Speaker', 'Named Entities', 'Predicate Arguments', 'Corefrence']
dataframe = pd.DataFrame(columns=columns)
filesid = []
partnumber = []
wordnumbers = []
nameEntity = []
words = []
PFI = []
parsbit = []
prearg = []
spk = []
wrdsns = []
corefs = []

def readall(directory):
    word = ""
    for root, dir, f in os.walk(directory):
        cfile = 0
        for file in f:
            files = (os.path.join((root.replace(directory, "")), file)).replace(" ","")
            with open(root + '/' + file, 'r') as fl:
                wordnum = 0
                filesid.append("#begin document " + '(nw/' + files + ')' + '; part ' + str(cfile))
                emptyline()
                for line in fl:
                    line = line.split('\t')
                    if line[0]:
                        word = line[0]
                    else:
                        continue
                    if not ('\n' in word):
                        words.append(word)
                        filesid.append("nw/"+"".join(files.split()))
                        partnumber.append(cfile)
                        wordnumbers.append(wordnum)
                        line[3] = line[3].replace("*)", ')')
                        coref = line[3].replace("*", "-")
                        if coref.isdigit():
                            corefs.append('(' + coref + ')')
                        elif "(" in coref:
                            coref = coref.replace("(-", "")
                            digit = coref
                            corefs.append('(' + coref)
                        elif ')' in coref:
                            coref = digit + ')'
                            corefs.append(coref)
                        else:
                            corefs.append(coref)

                        if wordnum == 0:
                            parsbit.append('(*')
                        else:
                            parsbit.append('*')

                        nameEntity.append(line[1])
                        spk.append('-')
                        PFI.append('-')
                        wrdsns.append('-')
                        prearg.append('*')
                        wordnum += 1

                    if (('\n' in word) or (word == '.')) and (words[-1] != ''):
                        wordnum = 0
                        emptyline()
                        filesid.append("")
                    if filesid[-1] == '' and not(')' in parsbit[-2]):
                        parsbit[-2] = parsbit[-2] + ')'

                if not ('\n' in word) and filesid[-1] != '':
                    emptyline()
                    filesid.append("")
                if filesid[-1] == "" and not (')' in parsbit[-2]):
                    parsbit[-2] = parsbit[-2] + ')'


                filesid.append("#end document")
                emptyline()

                cfile += 1

    dataframe['Document ID'] = filesid
    dataframe['Part number'] = partnumber
    dataframe['Word number'] = wordnumbers
    dataframe['Word'] = words
    dataframe['Named Entities'] = nameEntity
    NEcorrect()
    dataframe['Corefrence'] = corefs
    corefCorrect()
    dataframe['Parse bit'] = parsbit
    dataframe['Word sence'] = wrdsns
    dataframe['Speaker'] = spk
    dataframe['Predicate Arguments'] = prearg
    dataframe['Predicate Frameset ID'] = PFI


def emptyline():
    words.append("")
    partnumber.append('')
    wordnumbers.append('')
    corefs.append('')
    nameEntity.append('')
    spk.append('')
    parsbit.append('')
    wrdsns.append('')
    prearg.append('')
    PFI.append('')

def convToSentence():
    sent = []
    sentences = []
    for word in words:
        if word != "":
            sent.append(word)
        else:
            sentences.append(sent)
            sent = []
    return sentences


def partOfSpeech():
    sentences = convToSentence()
    i = -1
    tagger = POSTagger(model='resources/postagger.model')
    post = [None] * len(words)
    t = 0
    for sent in sentences:
        pos = tagger.tag(sent)
        i += 1
        for wordpos in pos:
            post[i] = wordpos[1]
            if words[i] is "":
                post[i] = ""
            i += 1

    dataframe['Part of speech'] = post


def predicateLemma():
    lemm = []
    lemmatizer = Lemmatizer()
    for word in words:
        lem = lemmatizer.lemmatize(word)
        if '#' in lem:
            lemm.append(lem)
        elif word == "":
            lemm.append("")
        else:
            lemm.append('-')
    dataframe['Predicate lemma'] = lemm


def corefCorrect():
    for i in range(0, len(corefs)):
        if '(' in corefs[i]:
            digit = corefs[i]
        elif corefs[i] == ")":
            corefs[i] = digit.replace('(', '') + ')'


lastNE = ""


def NEcorrect():
    for i in range(0, len(nameEntity)):
        if '(' in nameEntity[i]:
            nameEntity[i] = '(' + nameEntity[i].replace('(', '')
            lastNE = '('
        elif ')' in nameEntity[i]:
            if lastNE == ')':
                nameEntity[i] = '*'
            else:
                nameEntity[i] = '*)'
            lastNE = ')'
        elif nameEntity[i] == '-':
            nameEntity[i] = nameEntity[i].replace('-', '*')
        elif nameEntity[i] != '*' and nameEntity[i] != '':
            nameEntity[i] = '(' + nameEntity[i] + ')'
    dataframe['Named Entities'] = nameEntity


def parser():
    lemmatizer = Lemmatizer()
    tagger = POSTagger(model='resources/postagger.model')
    parsr = DependencyParser(tagger=tagger, lemmatizer=lemmatizer)
    parse = ['*'] * len(dataframe)
    for i in range(0, len(words)):
        if words[i] == "":
            parse[i] = ""
    dataframe['Parse bit'] = parse


if __name__ == '__main__':
    reload(sys)
    sys.setdefaultencoding('utf8')
    directory = sys.argv[1]
    readall(directory)
    partOfSpeech()
    predicateLemma()
    f = open(sys.argv[2] + ".persian.v4_gold_conll", "w")
    f.write(dataframe.to_csv(sep=str('\t'), header=False, index=False))
    f.close()
