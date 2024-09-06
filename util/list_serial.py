# list的存储和读取

def store_list_of_list(llis, filepath):
    with open(filepath, 'w') as f:
        for lis in llis:
            f.write(','.join(lis) + '\n')


def read_list_of_list(filepath):
    llis = []
    with open(filepath, 'r') as f:
        for line in f:
            lis = line.split(',')
            # lis的最后一个元素是换行符，需要去掉
            lis[-1] = lis[-1][:-1]
            llis.append(lis)
    return llis
