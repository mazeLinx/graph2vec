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
        exs = f.readlines()
        for i, l in enumerate(exs):
            ex = l.strip().split()[1:]
            feas = ' '.join([f.split(':')[1] for f in ex])
            idx2feas[i] = feas
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

feai2v = read_fea(feapath)
labeli2v= read_i2l(labelpath)
n2n = read_i2l(adjpath)
traini2v = read_i(trainidpath)
testi2v = read_i(testidpath)

fea_size = 2

out_path = r'/Users/jianbinlin/DDNN/project/stru2vec_data/samples_test.txt'

i2v = {}

for i in testi2v:
    
    label = labeli2v[i]    

    fea = ' '.join(map(str, np.zeros(fea_size)))
    if i in feai2v:
        fea = feai2v[i]
    else:
        print "{} fea no found".format(i)

    neig = ""
    if i in n2n:
        neig = n2n[i]
    else:
        print "{} n2n not found".format(i)

    i2v[i] = "{0} {1} {2}".format(label, fea, neig)


with open(out_path, 'w') as outf:
    for k, v in i2v.iteritems():
        outf.writelines("{} {}\n".format(k, v))


