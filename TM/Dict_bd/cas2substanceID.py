
import re
from fake_useragent import UserAgent
from selenium import webdriver
#########webdriver can be considered as the driver of the browser.


browser = webdriver.Safari()
########Add URL, PubChem


url='https://pubchem.ncbi.nlm.nih.gov/search/#query=sid'
########Create a list of CAS and CID, import request headers


CASid=[]
SIDid=[]
ua=UserAgent()
####### Open file, import original file, convert CAS number to list storage

cas_SID = open('./CAS-SID.txt', 'a',encoding='utf-8-sig', newline='')
CAS_file=open('~/merger.txt','r',encoding='utf-8-sig')
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
    # that is, to obtain the SID number on the retrieved web page.
    if re.search(r'SID' + '\d+', text):
        SID = re.findall(r'SID' + '\d+', text)
        CASid.append(cas_line[i])
        SIDid.append(SID)
        str1=str(SIDid[i])+','+str(CASid[i])
        cas_SID.write(str1)
        str2 = str(SIDid[i]) + ',' + str(CASid[i])
        cas_SID_num=cas_SID.write(str2)
    else:
        CASid.append(cas_line[i])
        SIDid.append('none')
        str1 = str(SIDid[i]) + ',' + str(CASid[i])
        cas_SID.write(str1)
        str2 = str(SIDid[i]) + ',' + str(CASid[i])
        cas_SID_no=cas_SID.write(str2)
##Then export the result to a new text
for i in range(len(CASid)):
    cas_SID = open('./CAS-SID.txt', 'a',encoding='utf-8-sig', newline='')
    str1 = str(SIDid[i]) + ',' + str(CASid[i])
    cas_SID.write(str1)
browser.close()
