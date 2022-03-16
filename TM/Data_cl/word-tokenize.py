import nltk
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
f = open('./output.txt')
cb = f.read()
text = word_tokenize(cb)

print(pos_tag(text))
