#encoding:utf-8
import os
import pickle
import pandas as pd
from collections import Counter
from entity import split_text
from tqdm import tqdm
from nltk import word_tokenize, pos_tag
import shutil
from random import shuffle
from glob import glob

train_dir = "train_data"

#----------------------------Function: Text Preprocessing---------------------------------
def process_text(idx, split_method=None, split_name='train'):
    """
    Function: Read the text and cut it, then mark it and extract features such as word boundaries and parts of speech
    param idx: file name without extension
    param split_method: cut text method
    param split_name: Store data set, Default training set, and test set
    return
    """

    #Definition dictionary Saves all word tags, word boundaries, parts of speech and other features
    data = {}

    #--------------------------------------------------------------------
    #                            Get sentence
    #--------------------------------------------------------------------
    if split_method is None:
        #No text segmentation function -> read file
        with open(f'data/{train_dir}/{idx}.txt', encoding='utf8') as f:
            texts = f.readlines()
    else:
        #Give text segmentation function -> segmentation by function
        with open(f'data/{train_dir}/{idx}.txt', encoding='utf8') as f:
            outfile = f'data/train_data_pro/{idx}_pro.txt'
            print(outfile)
            texts = f.read()
            texts = split_method(texts, outfile)

    #Extract sentences
    data['word'] = texts
    print(texts)

    #--------------------------------------------------------------------
    #             get label (entity type, starting position)
    #--------------------------------------------------------------------
    #Initially mark all words as O
    tag_list = ['O' for s in texts for x in s]    #Traverse over the words in each sentence

    #Read the ANN file to get the type, start position and end position of each entity
    tag = pd.read_csv(f'data/{train_dir}/{idx}.ann', header=None, sep='\t')


    for i in range(tag.shape[0]):  #tag.shape[0] is the number of lines
        tag_item = tag.iloc[i][1].split(' ')    #Second column of each line space separated
        #print(tag_item)
        #Get start and end positions
        cls, start, end = tag_item[0], int(tag_item[1]), int(tag_item[-1])
        #print(cls,start,end)
        
        #Modify tag_list
        tag_list[start] = 'B-' + cls
        for j in range(start+1, end):
            tag_list[j] = 'I-' + cls

    #Assert   The two lengths are inconsistent and an error is reported
    assert len([x for s in texts for x in s])==len(tag_list)
    #print(len([x for s in texts for x in s]))
    #print(len(tag_list))

    #--------------------------------------------------------------------
    #             Sentence matching labels after segmentation
    #--------------------------------------------------------------------
    tags = []
    start = 0
    end = 0
    #Traverse over text
    for s in texts:
        length = len(s)
        end += length
        tags.append(tag_list[start:end])
        start += length    
    print(len(tags))
    #Tag is stored in a dictionary
    data['label'] = tags

    #--------------------------------------------------------------------
    #            Extract parts of speech and word boundaries
    #--------------------------------------------------------------------
    #Initially marked as M
    word_bounds = ['M' for item in tag_list]    #Boundary
    word_flags = []                             #parts of speech
    
    #Tokenize
    for text in texts:
        token_words = word_tokenize(text)
        token_words = pos_tag(token_words)
        for word, tag in token_words:
            if len(word)==1:
                start = len(word_flags)
                word_bounds[start] = 'S'   #Single word
                word_flags.append(flag)
            else:
                start = len(word_flags)
                word_bounds[start] = 'B'         #Start border
                word_flags += [flag]*len(word)   #Guaranteed part-of-speech and word correspondence
                end = len(word_flags) - 1
                word_bounds[end] = 'E'           #End border
    #store
    bounds = []
    flags = []
    start = 0
    end = 0
    for s in texts:
        length = len(s)
        end += length
        bounds.append(word_bounds[start:end])
        flags.append(word_flags[start:end])
        start += length
    data['bound'] = bounds
    data['flag'] = flags

    #--------------------------------------------------------------------
    #                              Store data
    #--------------------------------------------------------------------
    #Get the number of samples
    num_samples = len(texts)     #Number of lines
    num_col = len(data.keys())   #Number of columns
    print(num_samples)
    print(num_col)
    
    dataset = []
    for i in range(num_samples):
        records = list(zip(*[list(v[i]) for v in data.values()]))   #zip
        dataset += records+[['sep']*num_col]                        #Each processing sentence is separated by sep
    #records = list(zip(*[list(v[0]) for v in data.values()]))
    #for r in records:
    #    print(r)
    
    #The last line sep delete
    dataset = dataset[:-1]
    #Convert to dataframe and add header
    dataset = pd.DataFrame(dataset,columns=data.keys())
    #Save file test set training set
    save_path = f'data/prepare/{split_name}/{idx}.csv'
    dataset.to_csv(save_path, index=False, encoding='utf-8')

    #--------------------------------------------------------------------
    #                  Handling newlines w represents a word
    #--------------------------------------------------------------------
    def clean_word(w):
        if w=='\n':
            return 'LB'
        if w in [' ','\t','\u2003']:
            return 'SPACE'
        if w.isdigit():              #Convert all numbers to one symbol Numeric training can cause interference
            return 'NUM'
        return w
    

    dataset['word'] = dataset['word'].apply(clean_word)

    #Store data
    dataset.to_csv(save_path, index=False, encoding='utf-8')
    
    
    #return texts, tags, bounds, flags
    #return texts[0], tags[0], bounds[0], flags[0]


#----------------------------Function: Preprocess all text---------------------------------
def multi_process(split_method=None,train_ratio=0.8):
    """
    Function: Preprocess all text
    param split_method: Cut text method
    param train_ratio: The ratio of training set and test set
    return
    """
    
    #Delete directory
    if os.path.exists('data/prepare/'):
        shutil.rmtree('data/prepare/')
        
    #Create a directory
    if not os.path.exists('data/prepare/train/'):
        os.makedirs('data/prepare/train/')
        os.makedirs('data/prepare/test/')

    #Get all filenames
    idxs = set([file.split('.')[0] for file in os.listdir('data/'+train_dir)])
    idxs = list(idxs)
    
    #Randomly split training and test sets
    shuffle(idxs)                         #Mess up the order
    index = int(len(idxs)*train_ratio)    #Get the subscript of the training set
    #Get training set and test set filename set
    train_ids = idxs[:index]
    test_ids = idxs[index:]

    #--------------------------------------------------------------------
    #                             Multiprocessing
    #--------------------------------------------------------------------
    import multiprocessing as mp
    num_cpus = mp.cpu_count()           #Get the number of CPUs
    pool = mp.Pool(num_cpus)
    
    results = []
    #Training set processing
    for idx in train_ids:
        result = pool.apply_async(process_text, args=(idx,split_method,'train'))
        results.append(result)
    #Test set processing
    for idx in test_ids:
        result = pool.apply_async(process_text, args=(idx,split_method,'test'))
        results.append(result)
    #Close the process pool
    pool.close()
    pool.join()
    [r.get for r in results]


#----------------------------Function: Generate mapping dictionary---------------------------------
#Statistical function: list, frequency calculation threshold
def mapping(data,threshold=10,is_word=False,sep='sep',is_label=False):
    #Count the number of various types in the list data
    count = Counter(data)

    #Remove the previously customized sep
    if sep is not None:
        count.pop(sep)

    #Unregistered word processing Less frequent occurrence Set to Unknown
    if is_word:
        #Set the frequency of the following two words to sort first
        count['PAD'] = 100000001          #Padding characters to ensure the same length
        count['UNK'] = 100000000
        #Descending sort
        data = sorted(count.items(),key=lambda x:x[1], reverse=True)
        #Remove elements with frequency less than threshold
        data = [x[0] for x in data if x[1]>=threshold]
        #Convert to dictionary
        id2item = data
        item2id = {id2item[i]:i for i in range(len(id2item))}
    elif is_label:
        data = sorted(count.items(),key=lambda x:x[1], reverse=True)
        data = [x[0] for x in data]
        id2item = data
        item2id = {id2item[i]:i for i in range(len(id2item))}
    else:
        count['PAD'] = 100000001
        data = sorted(count.items(),key=lambda x:x[1], reverse=True)
        data = [x[0] for x in data]
        id2item = data
        item2id = {id2item[i]:i for i in range(len(id2item))}
    return id2item, item2id

#Generate mapping dictionary
def get_dict():
    #Get all content
    all_w = []         #Word
    all_label = []     #Label
    all_bound = []     #Bound
    all_flag = []      #Parts of speech

    #Read file
    for file in glob('data/prepare/train/*.csv') + glob('data/prepare/test/*.csv'):
        df = pd.read_csv(file,sep=',')
        all_w += df['word'].tolist()
        all_label += df['label'].tolist()
        all_bound += df['bound'].tolist()
        all_flag += df['flag'].tolist()


    #Save the returned result dictionary
    map_dict = {} 

    #Call statistics function
    map_dict['word'] = mapping(all_w,threshold=20,is_word=True)
    map_dict['label'] = mapping(all_label,is_label=True)
    map_dict['bound'] = mapping(all_bound)
    map_dict['flag'] = mapping(all_flag)


    #Dictionary save content
    #return map_dict

    #Save dictionary data to file
    with open(f'data/dict.pkl', 'wb') as f:
        pickle.dump(map_dict,f)
        
#-------------------------------Function: main function--------------------------------------
if __name__ == '__main__':
    #print(process_text('0',split_method=split_text,split_name='train'))

    #1.Multiprocessing
    #multi_process(split_text)

    #2.Generate mapping dictionary
    #print(get_dict())
    get_dict()

    #3.Read the dictionary file saved by the get_dict function
    with open(f'data/dict.pkl', 'rb') as f:
        data = pickle.load(f)
    print(data['bound'])
    
    
