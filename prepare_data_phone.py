
import random

test_rate = 0.2

inpath = r'/Users/jianbinlin/DDNN/project/phone_data/phone_sample'
out_train_path = r'/Users/jianbinlin/DDNN/project/phone_data/samples_train.txt'
out_test_path = r'/Users/jianbinlin/DDNN/project/phone_data/samples_test.txt'

out_train = open(out_train_path, 'w')

shuffle_size = 100000
ls_train = []
ls_test = []
with open(inpath, 'r') as f:   
    shuffle_cnt = 0
    cnt = 0
    for l in f.readlines():
        parts = l.strip().split(',')
        id = parts[1]
            
        label = "1 0"
        if parts[0] == "0":
            label = "0 1"
            
        fea = "100:1"

        nei = parts[2]

        if random.random() < test_rate:
            ls_test.append(id + '\t' + label + '\t' + fea + '\t' + nei)
        else:
            ls_train.append(id + '\t' + label + '\t' + fea + '\t' + nei)

        if shuffle_cnt >= shuffle_size:
            random.shuffle(ls_train)
            for item in ls_train:
                out_train.write(item)
                out_train.write('\n')
            ls_train = []
            shuffle_cnt = 0

            print shuffle_cnt, cnt
        
        shuffle_cnt += 1
        cnt += 1

    print shuffle_cnt, cnt

random.shuffle(ls_train)
for item in ls_train:
    out_train.write(item)
    out_train.write('\n')


out_train.close()

out_test = open(out_test_path, 'w')
for item in ls_test:
    out_test.write(item)
    out_test.write('\n')

out_test.close()