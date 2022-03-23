from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, AtomPairs, MolFromSmiles
from rdkit.Chem.MACCSkeys import GenMACCSKeys
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.AtomPairs import Pairs,Torsions
from rdkit.Chem.AtomPairs.Torsions import GetTopologicalTorsionFingerprintAsIntVect as TopologicalTorsionFingerPrint
from rdkit.Chem.Draw import IPythonConsole, SimilarityMaps
from rdkit.Chem.AtomPairs.Pairs import ExplainPairScore as ExplainAtomPairScore
from rdkit.Chem.Draw.SimilarityMaps import GetAPFingerprint as AtomPairFingerprint, GetTTFingerprint as TopologicalFingerprint,GetMorganFingerprint as MorganFingerprint
import time
import warnings
import platform
import sys
import os


def getCountInfo(m, fpType):
#     m = Chem.MolFromSmiles(formula)
    fp = None
    if fpType=='AtomPair' or fpType.lower()=='atom':
        fp = Pairs.GetAtomPairFingerprint(m)
        return fp.GetNonzeroElements()
    elif fpType.lower()=='morgan' or fpType.lower()=='circular':
        fp = AllChem.GetMorganFingerprint(m,2)
        return fp.GetNonzeroElements()
    elif fpType=='Topological' or fpType.lower()=='topo':
        fp = Torsions.GetTopologicalTorsionFingerprint(m)
        Dict = fp.GetNonzeroElements()
        convertedDict = {}
        for elem in Dict:
            convertedDict[int(elem)] = Dict[elem]
        return convertedDict

def getKeys(mol,fpType):
    return getCountInfo(mol,fpType).keys()

def generateUnFoldedFingerprint(mols,fpType):
#     mols = []
#     for SMILE in SMILES:
#         mols += [MolFromSmiles(SMILE)]

    fpKeys,unfoldedFP = [],[]
    for mol in mols:
        fpKeys += getKeys(mol,fpType)
    fpKeys = list(set(fpKeys))

    ## Iterating over each molecule
    for mol in mols:
        fpDict = getCountInfo(mol,fpType)
        keys = fpDict.keys()
        x = list(set(fpKeys)-set(keys))
        y = [0]*len(x)
        dictionary = dict(zip(x,y))
        dictionary.update(fpDict)
        List = []
        for attribute in fpKeys:
            List += [dictionary[attribute]]
        unfoldedFP += [List]
    # saveData(fpKeys,fpType+'Positions')
    # print('saving positions to:'+fpType+'Positions.pkl')
    # saveData(unfoldedFP,fpType+'FoldedCount')
    # print('saving count to:'+fpType+'FoldedCount.pkl')
    return unfoldedFP, fpKeys

def getAtomPair(mol,nBits=1024):
	return SimilarityMaps.GetAPFingerprint(mol, fpType='bv',nBits=nBits)

def getTopological(mol,nBits=1024):
	return SimilarityMaps.GetTTFingerprint(mol, fpType='bv',nBits=nBits)

def getCircular(mol,nBits=1024):
    if nBits==1024:
        return AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024)
    else:
	    return SimilarityMaps.GetMorganFingerprint(mol, fpType='bv')

def getMACCS(mol):
	return MACCSkeys.GenMACCSKeys(mol)

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from ml_util import *


# Display training progress by printing a single dot for each completed epoch
class PrintEpochs(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 10 == 0:
            print('')
        else:
            print("epoch "+str(epoch)+" completed...")

def save_model(model, history, EPOCHS, lr, batch_size, fp="atom", data="cep"):
    file_suffix = "_"+fp+"_"+data+"_epochs_"+str(EPOCHS)+"_lr_"+str(lr)+"_batch_"+str(batch_size)
    model_file = join("model","model"+file_suffix+".json")
    weights_file = join("model","weights"+file_suffix+".h5")
    history_file = join("model","history"+file_suffix+".h5")

    model.save_weights(weights_file)
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_file, "w") as json_file:
        json_file.write(model_json)
    saveNumpy(history,history_file)

    print("History saved successfully in", history_file)
    print("Model saved successfully in", model_file)
    print("Weights saved successfully in", weights_file)


def build_general_model(X_train, optimizer='adam', lr=0.001, dropout=0.0):
    '''
    It is used by morgan fingerprints as well as atom_v2 fingerprints
    '''
    model = Sequential()
    model.add(Dense(4096, activation= keras.activations.relu, input_dim = X_train.shape[1]))
    if dropout!=0.0:
        model.add(Dropout(dropout))
    model.add(Dense(1024, activation= keras.activations.relu))
    if dropout!=0.0:
        model.add(Dropout(dropout))
    model.add(Dense(256, activation= keras.activations.relu))
    if dropout!=0.0:
        model.add(Dropout(dropout))
    model.add(Dense(64, activation= keras.activations.relu))
    if dropout!=0.0:
        model.add(Dropout(dropout))
    model.add(Dense(1))

#     optimizer = tf.train.RMSPropOptimizer(0.001)

    if optimizer == "adam" and lr!= 0.001:
        print("Setting learning rate to"+str(lr))
        optimizer = tf.train.AdamOptimizer(lr)
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mape'])
    return model

def deep_regress_terminal(X,Y,EPOCHS=50,optimizer='adam',lr=0.001,dropout=False,deep=False,batch_size=32,fp="atom",data="cep"):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=1024)
    if deep:
        print("We are building a deeper network")
    else:
        print("We are building default network")

    if len(X[0])==167: #if fingeprint is maccs or topo
        model = build_maccstopo_model(X_train, optimizer,lr,dropout,deep)
    elif len(X[0])==586: #if fingerprint is atom
        model = build_atom_model(X_train, optimizer,lr,dropout,deep)
    else:
        model = build_general_model(X_train, optimizer,lr,dropout)
    model.summary()

    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    # Store training stats
    start = time.time()
    if batch_size!=32:
        print("Batch size is set to: "+str(batch_size))
    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=batch_size,
                        validation_split=0.2, verbose=0,
                        callbacks=[PrintEpochs()])
    end = time.time()
    print()
    print('Time elapsed:'+str(end-start)+' seconds')
    # plot_history(history)

    [loss, mape] = model.evaluate(X_test, y_test, verbose=0)

    print("Testing set Mean Abs Error: {:2.4f}".format(mape ))

    save_model(model, history, EPOCHS, lr, batch_size)

if __name__ == "__main__":
    ignored_compound_num = loadNumpy('missed_cep_ap')
    HOMO_orig = loadNumpy('HOMO_cep_all')
    HOMO = []
    for i in range(300000):
        if i in ignored_compound_num:
            continue
        HOMO += [HOMO_orig[i]]
    HOMO = np.array(HOMO)

    data="cep"
    fp = "atom"
    batch_size = 64
    if len(sys.argv)>2:
        batch_size = int(sys.argv[2])
        if len(sys.argv)>3:
            fp = sys.argv[3]
            if len(sys.argv)>4:
                data = sys.argv[4]

    if fp == "atom"
    atompair = loadNumpy('ap_cep_merged_1','atom_cep')
    for i in range(2,4):
        atompair = np.concatenate((atompair, loadNumpy('ap_cep_merged_'+str(i),'atom_cep')))
    print("Completed loading the atom pair fingerprint")
    atompair, HOMO = shuffle(atompair, HOMO, random_state=1024)
    print("Shuffling completed and training starting")

    model, history = \
    deep_regress_terminal(atompair, HOMO, int(sys.argv[1]), batch_size=64, fp=fp, data=data)
    print("Training complete")