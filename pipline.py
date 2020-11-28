#!/usr/bin/env python3

import argparse
import random
import numpy as np
import torch
import pandas as pd
import os
# from rpy2.robjects.packages import importr
# from gensim.models.keyedvectors import FastTextKeyedVectors
# from pyaspeller import Word

import pymorphy2
import subprocess

# utils = importr('udpipe')


SEED = 1234
import matplotlib.pyplot as plt
import networkx as nx
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

NPRO = {'я': ('он','она'),
        'мне': ('ему','ей'),
        'мной': ('им','ею'),
        'мною': ('им','ею'),
        'обо мне': ('о нем','о ней'),
        'мы': ('они','они'),
        'нас': ('их','их'),
        'ей': ('им','им'),
        'вами': ('ими','ими'),
        'о нас': ('о них','о них'),
        'нам': ('им','им'),
        'наши': ('их','их'),
        'общий': ('его','её'),
        }


def twostep():
    # if opt.readfile=='TRUE':
    command = 'Rscript'
    path2script = '/home/alex/PycharmProjects/coref/udpip.R'
    cmd = [command, path2script]+[opt.outdir+'input.txt']
    out = subprocess.check_output(cmd, universal_newlines=True)
    x = pd.read_csv('/home/alex/PycharmProjects/coref/outputR.csv')
    # else:
    #     with localconverter(ro.default_converter + pandas2ri.converter):
    #         x = ro.r('''udmodel <- udpipe_download_model(language = "russian-syntagrus", overwrite=FALSE)
    #                  x <- udpipe(x = "'''+opt.string+'''",  object = udmodel)''')
    return x

def threestep(tokens):
    morph = pymorphy2.MorphAnalyzer()
    CASES = {

        ('pres', '<Настоящее время>'): [
            {'1per', 'sing'},
            {'3per', 'sing'},
            {'1per', 'plur'},
            {'3per', 'plur'}
        ]
    }
    rows_list = []
    gender = 'masc'
    # predgender = None
    list_pre_gender = []
    for index, row in tokens.iterrows():
        word = morph.parse(row['head'].lower())[0]
        # if word.score < 0.8:
        #     check = Word(row['head'].lower())
        #     if not check.correct:
        #         ya = check.spellsafe
        #         if ya!= None:
        #             word = morph.parse(ya)[0]
        #             # tokens.loc[index]['head'] = word.word
        #         # row['head'] = word.word
        #         else:
        #             m = FastTextKeyedVectors.load("/home/alex/PycharmProjects/coref/coreference_resolution/emb_fasttext/model.model")
        #             near = m.wv.most_similar(row['head'].lower())[0]
        #             score = near[1]
        #             if score >  word.score:
        #                 word = morph.parse(near[0])[0]
                    # tokens.loc[index]['head'] = word.word
                    # row['head'] =  word.word
        # print(word)
        gender_word = word.tag.gender

        if word.tag.POS == 'VERB' and (gender_word == 'femn' or gender_word == 'masc'):
            # predgender = gender
            list_pre_gender.append(gender_word)
    if len(list_pre_gender) != 0:
        gender = max(list_pre_gender,key=list_pre_gender.count)

    be_nom = False
    if os.path.isfile(opt.outdir+'alias.txt'):
        with open(opt.outdir+'alias.txt', 'r') as ff:
            nom = ff.read()
            wordnom = morph.parse(nom.lower())[0]
            # be_nom = True



    for index, row in tokens.iterrows():
        word = morph.parse(row['head'].lower())[0]
        # if word.score < 0.8:
        #     check = Word(row['head'].lower())
        #     if not check.correct:
        #         ya = check.spellsafe
        #         if ya!= None:
        #             word = morph.parse(ya)[0]
        #             # tokens.loc[index]['head'] = word.word
        #         # row['head'] = word.word
        #         else:
        #             m = FastTextKeyedVectors.load("/home/alex/PycharmProjects/coref/coreference_resolution/emb_fasttext/model.model")
        #             near = m.wv.most_similar(row['head'].lower())[0]
        #             score = near[1]
        #             if score >  word.score:
        #                 word = morph.parse(near[0])[0]
                    # tokens.loc[index]['head'] = word.word
                    # row['head'] =  word.word
        # print(word)
        person = word.tag.person # 1per
        number = word.tag.number # sing
        tense = word.tag.tense # pres

        if be_nom:
            row['nom3d'] = wordnom.inflect({'3per', number, tense}).word
        else:
            pre = 'н'if row.adj_lem  and row.adj_head == row.token_id else ''
            if gender == 'masc' or gender == None or gender == 'neut':
                if row.token.lower() in NPRO:
                    row['nom3d'] = pre+NPRO[row.token.lower()][0]
                else:
                    row['nom3d'] = pre+NPRO['общий'][0]
            elif gender == 'femn':
                if row.token.lower() in NPRO:
                    row['nom3d'] = pre+NPRO[row.token.lower()][1]
                else:
                    row['nom3d'] = pre+NPRO['общий'][1]

        if row.adj and row.adj[-1] == 'о':
            rows_list.append({'token_id': row.adj_id, 'tokenold': row.adj, 'tokennew': row.adj[:-1],
                              'sentence_id': row.sentence_id})

        if person == '1per' and word.tag.POS in {'INFN', 'VERB'}:
            cases = {'3per', number, tense}
            try:
                w = word.inflect(cases).word
                row['verb3d'] = w
            except AttributeError:
                pass
        else:
            row['verb3d'] = word.word

        if row.token:
            rows_list.append({'token_id': row.token_id, 'tokenold': row.token, 'tokennew': row.nom3d, 'sentence_id':row.sentence_id})
        rows_list.append({'token_id': row.head_token_id, 'tokenold': row['head'], 'tokennew': row.verb3d, 'sentence_id':row.sentence_id})



    newDF = pd.DataFrame(rows_list)
    newDF['onesort'] = pd.to_numeric(newDF.token_id, errors='coerce')
    newDF['twosort'] = pd.to_numeric(newDF.sentence_id, errors='coerce')
    newDF = newDF.sort_values(['twosort','onesort'], ascending=[False,False])

    return newDF

def getcluster(heads):
    det = heads[heads['lemma'].isin(['я', 'мой', 'мы', 'наш'])]

    adj = heads[heads['upos']=='ADP']
    per1 = heads[(heads['upos']=='VERB') & (heads['feats'].str.contains("Person=1", na = False))]
    onlyverb = []
    for index, row in per1.iterrows():
        onlyverb.append({'token_id':'',
                         'token':'',
                         'Unnamed':row['Unnamed: 0'],
                         'head_token_id':row.token_id,
                         'sentence_id':row.sentence_id,
                         'head':row.token,
                         'adj':'',
                         'adj_lem':'',
                         'adj_id':'',
                         'adj_head':'',
                         })


    newDF = pd.DataFrame(det[['token_id','token','head_token_id','sentence_id']], columns=['token_id','token','head_token_id','sentence_id'])
    newDF['head'] = newDF.apply(lambda x: ( heads [ ( heads['token_id']==x.head_token_id) & ( heads['sentence_id']==x.sentence_id)   ]['token'].values[0]), axis=1)
    newDF['Unnamed'] = newDF.apply(lambda x: ( heads [ ( heads['token_id']==x.head_token_id) & ( heads['sentence_id']==x.sentence_id)   ]['Unnamed: 0'].values[0]), axis=1)
    newDF['adj'] = newDF.apply(lambda x: ( adj [ ( adj['token_id']==(x.token_id-1)) & ( adj['sentence_id']==x.sentence_id)   ]['token'].values[0]
                                        if adj [ ( adj['token_id']==(x.token_id-1)) & ( adj['sentence_id']==x.sentence_id)   ]['token'].values else ''), axis=1)
    newDF['adj_lem'] = newDF.apply(lambda x: ( adj [ ( adj['token_id']==(x.token_id-1)) & ( adj['sentence_id']==x.sentence_id)   ]['lemma'].values[0]
                                        if adj [ ( adj['token_id']==(x.token_id-1)) & ( adj['sentence_id']==x.sentence_id)   ]['lemma'].values else ''), axis=1)
    newDF['adj_id'] = newDF.apply(lambda x: ( adj [ ( adj['token_id']==(x.token_id-1)) & ( adj['sentence_id']==x.sentence_id)   ]['token_id'].values[0]
                                        if adj [ ( adj['token_id']==(x.token_id-1)) & ( adj['sentence_id']==x.sentence_id)   ]['token_id'].values else ''), axis=1)
    newDF['adj_head'] = newDF.apply(lambda x: ( adj [ ( adj['token_id']==(x.token_id-1)) & ( adj['sentence_id']==x.sentence_id)   ]['head_token_id'].values[0]
                                        if adj [ ( adj['token_id']==(x.token_id-1)) & ( adj['sentence_id']==x.sentence_id)   ]['head_token_id'].values else ''), axis=1)
    dfonlyverb = pd.DataFrame(onlyverb)
    if len(dfonlyverb) != 0:
        dfonlyverb = dfonlyverb[~(dfonlyverb['Unnamed'].isin(newDF['Unnamed']) )]
    result = pd.concat([newDF,dfonlyverb])
    result['nom3d'] = ''
    result['verb3d'] = ''

    return result

def getnewtext(alltext, cluster3d, oldtext):
    newtext = oldtext
    newtext2 = oldtext
    coordinat = {}
    outlist = []
    outlist2 = []
    outlist3 = []
    nom = 0

    for indx, row in cluster3d.iterrows():
        rr = alltext[(alltext['token_id'] == row.token_id) & (alltext['sentence_id'] == row.sentence_id)]
        start = rr['start'].values[0]
        end = rr['end'].values[0]
        nom += 1
        if row.tokennew.lower() != row.tokenold.lower():
            if row.tokenold[0] == row.tokenold[0].upper():
                newtext = newtext[:(start - 1)] + '{}({})'.format(row.tokennew.capitalize(), row.tokenold) + newtext[end:]
                newtext2 = newtext2[:(start - 1)] + '{}'.format(row.tokennew.capitalize()) + newtext2[end:]
                outlist.append({'start': start, 'end': end, 'tokenold': row.tokenold, 'tokennew': row.tokennew.capitalize()})
            else:
                newtext = newtext[:(start - 1)] + '{}({})'.format(row.tokennew, row.tokenold) + newtext[end:]
                newtext2 = newtext2[:(start - 1)] + '{}'.format(row.tokennew) + newtext2[end:]
                outlist.append({'start':start, 'end':end, 'tokenold':row.tokenold, 'tokennew':row.tokennew})

            delta = len(row.tokenold) - len(row.tokennew)
            for key, value in coordinat.items():
                coordinat[key] = value - delta
            coordinat['{}%{}%{}'.format(nom, row.tokenold, row.tokennew)] = start - 1

            addi = 2 + len(row.tokennew)
            for co in outlist3:
                co[0] += addi
            outlist3.append([start+len(row.tokennew), len(row.tokenold)])


    for key, val in coordinat.items():
        outlist2.append({'token':key, 'shift':val})
    df = pd.DataFrame(outlist2)
    df.to_csv(opt.outdir+'coordinat.csv', header=False)

    outlist3dict = []
    for co in outlist3:
        outlist3dict.append({'start':co[0], 'end':co[1]})
    df = pd.DataFrame(outlist3dict)
    df.to_csv(opt.outdir+'coordinat2.csv', header=False)

    df2 = pd.DataFrame(outlist)
    df2.to_csv(opt.outdir+'output.csv')


    with open(opt.outdir+'output.txt', 'w') as f:
        f.write(newtext)
    with open(opt.outdir+'output2.txt', 'w') as f:
        f.write(newtext2)
    return newtext

if __name__ == '__main__':
    #     with open('/home/alex/PycharmProjects/coref/input.txt','w') as fw:
    #         fw.write(text)
    parser = argparse.ArgumentParser()
    # parser.add_argument('--readfile', type=str, default='TRUE', help='')
    # parser.add_argument('--file', type=str, default='./flask/static/input.txt', help='')
    parser.add_argument('--outdir', type=str, default='./outdataoffice/', help='')
    # parser.add_argument('--string', type=str, default='', help='')

    opt = parser.parse_args()
    file = opt.outdir+'input.txt'
    with open(file, 'r') as f:
        text = f.read()
    # кластера кореференции
    # onestep()

    #вызов R
    alltext = twostep()

    cluster1d = getcluster(alltext)
    # подключение морфологии
    cluster3d = threestep(cluster1d)

    newtext = getnewtext(alltext, cluster3d, text )

    # print(opt.string)
    print(newtext)

