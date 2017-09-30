import numpy as np


def read_nb(path):
    n2n = {}
    with open(path, 'r') as f:
        for l in f.readlines():
            parts = l.strip().split()
            n2n[parts[0]] = ' '.join(parts[1:])
    return n2n
            
                
def read_fea(path):
    idx2feas = {}
    with open(path, 'r') as f:
        exs = f.readlines()
        for i, l in enumerate(exs):
            ex = l.strip().split()[1:]
            feas = ' '.join([f.split(':')[1] for f in ex])
            idx2feas[i] = feas
    return idx2feas

def read_fea_libsvm(path):
    idx2feas = {}
    with open(path, 'r') as f:
        for i, l in enumerate(f.readlines()):
            ex = l.strip().split()[1:]
            ids = ' '.join([f.split(':')[0] for f in ex])
            feas = ' '.join([f.split(':')[1] for f in ex])
            idx2feas[i] = (ids, feas)
    return idx2feas

def read_cnt2v(path):
    idx2feas = {}
    with open(path, 'r') as f:
        for i, l in enumerate(f.readlines()):
            idfea = l[l.index(' ') + 1 :].strip()
            idx2feas[i] = idfea
    return idx2feas

def read_i2l(path):
    idx2value = {}
    with open(path, 'r') as f:
        exs = f.readlines()
        for i, value in enumerate(exs):
            idx2value[i] = value.strip()
    
    return idx2value

def read_i(path):
    l = []
    with open(path, 'r') as f:
        exs = f.readlines()
        for i, value in enumerate(exs):
            l.append(int(value.strip()))
    return l


feapath = r'/Users/jianbinlin/DDNN/project/stru2vec_data/feature'
labelpath = r'/Users/jianbinlin/DDNN/project/stru2vec_data/label'
trainidpath = r'/Users/jianbinlin/DDNN/project/stru2vec_data/train'
testidpath = r'/Users/jianbinlin/DDNN/project/stru2vec_data/test'
adjpath = r'/Users/jianbinlin/DDNN/project/stru2vec_data/graph'

# feapath = r'/Users/jianbinlin/DDNN/project/stru2vec/data/svd_features_10.txt'
# labelpath = r'/Users/jianbinlin/DDNN/project/stru2vec/data/label.txt'
# trainidpath = r'/Users/jianbinlin/DDNN/project/stru2vec/data/train_idx.txt'
# testidpath = r'/Users/jianbinlin/DDNN/project/stru2vec/data/test_idx.txt'
# adjpath = r'/Users/jianbinlin/DDNN/project/stru2vec/data/adj_list.txt'

feai2v = read_cnt2v(feapath)
labeli2v= read_i2l(labelpath)
n2n = read_cnt2v(adjpath)
traini2v = read_i(trainidpath)
testi2v = read_i(testidpath)


out_path = r'/Users/jianbinlin/DDNN/project/stru2vec_data/samples_train.txt'

i2v = {}

for i in traini2v:
    
    label = labeli2v[i]    

    fea = ''
    if i in feai2v:
        fea = feai2v[i]
    else:
        print "{} fea no found".format(i)

    neig = ""
    if i in n2n:
        neig = n2n[i]
    else:
        print "{} n2n not found".format(i)

    i2v[i] = "{0}\t{1}\t{2}".format(label, fea, neig)


with open(out_path, 'w') as outf:
    for k, v in i2v.iteritems():
        outf.writelines("{}\t{}\n".format(k, v))
