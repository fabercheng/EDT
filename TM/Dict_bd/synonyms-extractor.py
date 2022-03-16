###This is the api of pubchem.
###Convert the cas number to a symbol that can be recognized by pubchem, such as cid, substanceid, etc.,
###to obtain relevant physical and chemical information.
import re
import pubchempy
import pandas as pd
import numpy as np
with open('./chemspi-id-name.txt','r',encoding='utf-8-sig') as file1:
    file1 = pd.read_csv('./merger.csv',dtype={'code':str})
    file_lines=file1.readlines()
    name_list=[] ##create column in csv
    a=[]
    e=[]
    f=[]

for i in file_lines:
    j=i.strip()
    name_list.append(str(j))
for k in name_list:
    #results = pubchempy.Compound.from_cid(k)
    results = pubchempy.get_synonyms(k,'name') 
    ###The get_ command here can be replaced with the compound properties contained in pubchem.
    ###The detailed property list can be found here>>>https://pypi.org/project/PubChemPy/1.0/
    with open('chem-1.txt','a',encoding='utf-8-sig') as f1:
        print(results,file=f1)

