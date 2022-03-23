# Event-driven taxonomy (EDT) based non-target analysis tool
## Data process
### Literature collection
"Liter_col" is for collecting literature data, including generating document profile by automated crawling, which is supported by "sci-hub-crawler". The "pdf2txt-csv" can transform main text of literature into string metadata repository. 
### Dictionary contribution
Self-built list of chemical names was used as the dictionary for term recognition. Essential factors of compounds included names and synonyms, chemical identities like CASRN, PubChem CID or SID, structural formula like InChI key string, IUPAC name.  "Dict_bd" provides an API for the acquisition of the above parameters, and can realize the improvement of the dictionary.
### Corpus cleaning
"Data_cl" is to process data cleaning by tokenization and pos-tagging of the corpus.
Tokenization performs sentence segmentation, word segmentation, stem, removal of stop words and unification into lowercase letters in "Tok". "Pos" performs part-of-speech tagging to extract nouns and numbers.
## Named entity recognition (NER)
The NER performed based on NLTK in this section.
"entity" would show entity types and numbers, self-build and annotate entity (BIO) dictionary, perform sentences segmentation based on the cleaned corpus. "prep" preprocess the text by labeling BIO and mapping entities to build dictionary matrix. "utils" improves data processing, including batchly obtaining sentences and converting them into corresponding ids, splicing sentences, and processing data. "modle" is the main program of the NER model, including feature embedding, BI-LSTM layer, fully connected layer, CRF loss function, optimization, decoding and predictions. And "metrics" used for the model performance evaluations. 
## Fingerprint training
### Data curation
The dataset of compounds will be managed by the SQLite. Training set for ANN model should include the molecular parameters and toxicological terms, which can obtain from PubChem and CompTox, respectively. 
### Generation
Based on the RDKit library, MACCS keys string was translated from SMILES string. 
### Toxicity fingerprint training
The ANN model with 3 hidden layers were used for toxicity fingerprint training.
### Ion prediction
The CCNN model were used for ion prediction training.
