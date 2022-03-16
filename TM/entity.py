#encoding:utf-8
import os
import re

#----------------------------Function: Get entity type and number---------------------------------
def get_entities(dirPath):
    entities = {}                 #Store entity type
    files = os.listdir(dirPath)   #Traverse the path

    #Get the names of all files and deduplicate 0.ann => 0
    filenames = set([file.split('.')[0] for file in files])
    filenames = list(filenames)

    #Reconstruct ANN filenames and traverse over files
    for filename in filenames:
        path = os.path.join(dirPath, filename+".ann")
        with open(path, 'r', encoding='utf8') as f:
            for line in f.readlines():
                #TAB key split to get entity type
                name = line.split('\t')[1]
                value = name.split(' ')[0]
                #Entities are added to the dictionary and counted
                if value in entities:
                    entities[value] += 1
                else:
                    entities[value] = 1
    #Return entity set
    return entities

#----------------------------Function: Named Entity BIO annotation--------------------------------
def get_labelencoder(entities):
    #Sort
    entities = sorted(entities.items(), key=lambda x: x[1], reverse=True)
    print(entities)
    #Get entity type name
    entities = [x[0] for x in entities]
    print(entities)
    #Tag entity
    id2label = []
    id2label.append('O')
    #Generate entity tags
    for entity in entities:
        id2label.append('B-'+entity)
        id2label.append('I-'+entity)

    #Dictionary key generation
    label2id = {id2label[i]:i for i in range(len(id2label))}

    return id2label, label2id

#-------------------------Function:Text segmentation------------------------------
def split_text(text, outfile):
    #split subscript
    split_index = []
    fw = open(outfile, 'w', encoding='utf8')
    #Split by symbol
    pattern = '。|，|,|;|；|？|\?|\.'
    
    #Get the subscript position of a character
    for m in re.finditer(pattern, text):
        """
        print(m)
        start = m.span()[0] 
        print(text[start])
        start = m.span()[0] - 5
        end = m.span()[1] + 5
        print('****', text[start:end], '****')
        """
        #Special symbol subscript
        idx = m.span()[0]
        #Judging whether to segment the sentence 'contniue' means that the sentence cannot be directly segmented
        if text[idx-1]=='\n':
            continue
        if text[idx-1].isdigit() and text[idx+1].isdigit():  #Numbers or numbers + spaces before and after the symbol
            continue
        if text[idx-1].isdigit() and text[idx+1].isspace() and text[idx+2].isdigit():
            continue
        if text[idx-1].islower() and text[idx+1].islower():  #Lowercase letters before and after the symbol
            continue
        if text[idx-1].isupper() and text[idx+1].isupper():  #Capital letters before and after the symbol
            continue
        if text[idx-1].islower() and text[idx+1].isdigit():  #Symbols are preceded by lowercase letters followed by numbers
            continue
        if text[idx-1].isupper() and text[idx+1].isdigit():  #Symbols are preceded by capital letters followed by numbers
            continue
        if text[idx-1].isdigit() and text[idx+1].islower():  #Symbols are preceded by numbers followed by lowercase letters
            continue
        if text[idx-1].isdigit() and text[idx+1].isupper():  #Symbols are preceded by numbers followed by capital letters
            continue
        if text[idx+1] in set('.。;；,，'):                  #Punctuation before and after symbol
            continue
        if text[idx-1].isspace() and text[idx-2].isspace() and text[idx-3].isupper():
            continue
        if text[idx-1].isspace() and text[idx-3].isupper():
            continue
        
        #Store split subscripts into a list
        split_index.append(idx+1)

    #--------------------------------------------------------------------
    #The second part is divided by custom symbols
    #Sentence segmentation in the following form
    pattern2 = '\([1234567890]\)|[1234567890]、|'
    pattern2 += 'and |or |with |by |because of |as well as '
    for m in re.finditer(pattern2, text):
        idx = m.span()[0]
        #The conjunction is in the middle of the word and cannot be separated such as 'goodbye'
        if (text[idx:idx+2] in ['or', 'by'] or text[idx:idx+3] == 'and' or text[idx:idx+4] == 'with')\
            and (text[idx-1].islower() or text[idx-1].isupper()):
            continue
        split_index.append(idx)

    #--------------------------------------------------------------------
    #The third part number segmentation
    pattern3 = '\n\d\.'  #numeber+dot such as '1.'
    for m in  re.finditer(pattern3, text):
        idx = m.span()[0]

    #newline+number+brackets  such as '(1)'
    for m in re.finditer('\n\(\d\)', text):
        idx = m.span()[0]
        split_index.append(idx+1)

    #--------------------------------------------------------------------
    #After obtaining the sentence segmentation subscript, perform the sorting operation and add the first and last lines
    split_index = sorted(set([0, len(text)] + split_index))
    split_index = list(split_index)
    #print(split_index)

    #Calculate max and min
    lens = [split_index[i+1]-split_index[i] for i in range(len(split_index)-1)]
    #print(max(lens), min(lens))
        
    #--------------------------------------------------------------------
    #                     long and short sentence processing
    #--------------------------------------------------------------------
    other_index = []        
    for i in range(len(split_index)-1):
        begin = split_index[i]
        end = split_index[i+1]
        if (text[begin] in '1234567890') or \
            (text[begin]=='(' and text[begin+1] in '1234567890'):
            for j in range(begin,end):
                if text[j]=='\n':
                    other_index.append(j+1)

    split_index += other_index
    split_index = list(sorted(set([0, len(text)] + split_index)))

    #--------------------------------------------------------------------
    #Part 1 Long Sentence Processing: Splitting Sentences with a Length Over 150
    other_index = []
    for i in range(len(split_index)-1):
        begin = split_index[i]
        end = split_index[i+1]
        other_index.append(begin)
            
        #Sentence length over 150 cuts and a minimum of 15 characters
        if end-begin>150:
            for j in range(begin,end):
                #If the subscript position is more than 15 than the last time, it will be divided
                if(j+1-other_index[-1])>15:
                    #newline split
                    if text[j]=='\n':
                        other_index.append(j+1)
                    #space + numbers before and after
                    if text[j]==' ' and text[j-1].isnumeric() and text[j+1].isnumeric():
                        other_index.append(j+1)
    split_index += other_index
    split_index = list(sorted(set([0, len(text)] + split_index)))

    #--------------------------------------------------------------------
    #Part 2 Sentences with spaces removed
    for i in range(1, len(split_index)-1):
        idx = split_index[i]
        #The current subscript is compared with the previous subscript. If it is a space, the comparison continues
        while idx>split_index[i-1]-1 and text[idx-1].isspace():
            idx -= 1
        split_index[i] = idx
    split_index = list(sorted(set([0, len(text)] + split_index)))

    #--------------------------------------------------------------------
    #Part 3 Short sentence processing - splicing
    temp_idx = []
    i = 0
    while i<(len(split_index)-1):
        begin = split_index[i]
        end = split_index[i+1]
        #First count the number of English characters in the sentence
        num_en = 0
        if end - begin <15:
            for en in text[begin:end]:
                if en.islower() or en.isupper():
                    num_en += 1
                if 0.5 *num_en>5:  #More than 5 indicates that the length is sufficient
                    temp_idx.append(begin)
                    i += 1                 #Note that i add 1 before break, otherwise an infinite loop
                    break
            #The length is less than or equal to 5 and the following sentences are merged
            if 0.5*num_en<=5:
                temp_idx.append(begin)
                i += 2
        else:
            temp_idx.append(begin)  #Add subscript directly if greater than 15
            i += 1
    split_index = list(sorted(set([0, len(text)] + temp_idx)))

    #Check the length of the sentence
    lens = [split_index[i+1]-split_index[i] for i in range(len(split_index)-1)][:-1] #remove last newline
    print(max(lens), min(lens))
        
    #for i in range(len(split_index)-1):
    #    print(i, '****', text[split_index[i]:split_index[i+1]])

    #store the result
    result = []
    for i in range(len(split_index)-1):
        result.append(text[split_index[i]:split_index[i+1]])
        fw.write(text[split_index[i]:split_index[i+1]])
    fw.close()

    #Check: Are characters reduced after preprocessing
    s = ''
    for r in result:
        s += r
    assert len(s)==len(text)
    return result



#-------------------------------Function: main function--------------------------------------
if __name__ == '__main__':
    dirPath = "data/train_data"
    outPath = 'data/train_data_pro'

    #Get entity type and number
    entities = get_entities(dirPath)
    print(entities)
    print(len(entities))

    #Complete entity tag list dictionary
    #Get a map of labels and subscripts
    label, label_dic = get_labelencoder(entities)
    print(label)
    print(len(label))
    print(label_dic, '\n\n')

    #Traverse the path
    files = os.listdir(dirPath)   
    filenames = set([file.split('.')[0] for file in files])
    filenames = list(filenames)
    for filename in filenames:
        path = os.path.join(dirPath, filename+".txt")  #TXT file
        outfile = os.path.join(outPath, filename+"_pro.txt")
        #print(path)
        with open(path, 'r', encoding='utf8') as f:
            text = f.read()
            #split text
            print(path)
            split_text(text, outfile)
    print("\n")
