

path = r'/Users/jianbinlin/DDNN/project/stru2vec_data/samples_train.txt'
edage = 0
linecnt = 0
with open(path, 'r') as inf:
    for l in inf.readlines():
        if l:            
            linecnt += 1

            parts = l.split('\t')[3].split(' ')
            edage += len(parts)

print linecnt, edage
        