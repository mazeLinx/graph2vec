

path = r'/Users/jianbinlin/DDNN/project/stru2vec/data/samples_test.txt'
edage = 0
linecnt = 0
with open(path, 'r') as inf:
    for l in inf.readlines():
        if l:            
            parts = l.split(' ')[50:]
            linecnt += 1
            edage += len(parts)

print linecnt, edage
        