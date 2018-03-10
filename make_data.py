import os
import cPickle
import os.path as op
import numpy as np
from multiprocessing import pool
from finsymbols import symbols
from progressbar import ProgressBar

CUR_DIR = os.getcwd()
DATA_DIR = op.join(CUR_DIR, 'data')
DUMP_DIR = op.join(CUR_DIR, 'data_32')

sp500 = symbols.get_sp500_symbols()
sp_name = [sym['symbol'] for sym in sp500]

def concat_data():
    bar = ProgressBar()
    x_pile = []
    y_pile = []
    for cmpny in bar(sp_name):
        fx = op.join(DATA_DIR, "x_%s.pkl"%cmpny)
        if op.exists(fx):
            fy = op.join(DATA_DIR, "y_%s.pkl"%cmpny)
            with open(fx, 'r') as npx:
                x_pile.append(cPickle.load(npx))
            with open(fy, 'r') as npy:
                y_pile.append(cPickle.load(npy))
    print("finished loading")
    trX = np.concatenate(x_pile)
    del x_pile
    print("finished compiling x")
    trY = np.concatenate(y_pile)
    del y_pile
    print("finished compiling y")
    
    permu = np.random.permutation(trX.shape[0])
    np.take(trX,permu,axis=0,out=trX)
    np.take(trY,permu,axis=0,out=trY)
    div = int(trX.shape[0]*0.8)
    with open(op.join(DATA_DIR, "trX.npy"), 'w') as tx:
        np.save(tx, trX[0:div])
    with open(op.join(DATA_DIR, "teX.npy"), 'w') as tx:
        np.save(tx, trX[div:])
    with open(op.join(DATA_DIR, "trY.npy"), 'w') as ty:
        np.save(ty, trY[0:div])
    with open(op.join(DATA_DIR, "teY.npy"), 'w') as ty:
        np.save(ty, trY[div:])

def split_val():
    with open(op.join(DATA_DIR, 'trX.pkl'),'r') as f:
        trX = np.load(f)
    with open(op.join(DATA_DIR, 'trY.pkl'),'r') as f:
        trY = np.load(f)
    div = int(trX.shape[0]/8.*7.)
    print('saving valX')
    with open(op.join(DATA_DIR, "val_X.pkl"),'w') as f:
        cPickle.dump(trX[div:], f)
    with open(op.join(DATA_DIR, 'val_Y.pkl'),'w') as f:
        cPickle.dump(trY[div:], f)
    print('saving trainX')
    with open(op.join(DATA_DIR, 'tr_X.pkl'),'w') as f:
        cPickle.dump(trX[:div], f)
    with open(op.join(DATA_DIR, 'tr_X.pkl'),'w') as f:
        cPickle.dump(trY[:div], f)

if __name__ == "__main__":
    concat_data()
    # split_val()
