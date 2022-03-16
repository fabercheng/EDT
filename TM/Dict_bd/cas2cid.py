#######The cid can be obtained for most compounds. The following code is preferred.
# If there are missing cas numbers that have not been obtained, cas2substanceID.py can be used for the second round of obtaining.
import re
from fake_useragent import UserAgent
from selenium import webdriver
import pandas as pd
#########webdriver can be considered as the driver of the browser.


browser = webdriver.Safari()
########Add URL, PubChem


url='https://www.ncbi.nlm.nih.gov/pcsubstance/?term=cas'
########Create a list of CAS and CID, import request headers


CASid=[]
CIDid=[]
ua=UserAgent()
####### Open file, import original file, convert CAS number to list storage

cas_cid = open('./CAS-CID.txt', 'a',encoding='utf-8-sig', newline='')
CAS_file=open('~/merger1.txt','r',encoding='utf-8-sig',dtype={'code':str})
cas_line=CAS_file.readlines()
for i in range(len(cas_line)):
    cas_url=re.sub(r'cas',cas_line[i],url)
   # Replace 'cas' in url to make it 'cas' encoded
    headers = {"User-Agent": ua.random}
    # Headers are set to random
    browser.get(cas_url)
    text=browser.page_source
    # browser.page_source is to get all the html of the web page.
    # The following is to retrieve the text content of the web page through regular expressions,
    # that is, to obtain the CID number on the retrieved web page.
    if re.search(text):
        CID = re.findall(text)
        CASid.append(cas_line[i])
        CIDid.append(CID)
        str1=str(CIDid[i])+','+str(CASid[i])
        cas_cid.write(str1)
        str2 = str(CIDid[i]) + ',' + str(CASid[i])
        cas_cid_num=cas_cid.write(str2)
    else:
        CASid.append(cas_line[i])
        CIDid.append('none')
        str1 = str(CIDid[i]) + ',' + str(CASid[i])
        cas_cid.write(str1)
        str2 = str(CIDid[i]) + ',' + str(CASid[i])
        cas_cid_no=cas_cid.write(str2)
#Then export the result to a new text
for i in range(len(CASid)):
    cas_cid = open('./CAS-CID.txt', 'a',encoding='utf-8-sig', newline='')
    str1 = str(CIDid[i]) + ',' + str(CASid[i])
    cas_cid.write(str1)
browser.close()
