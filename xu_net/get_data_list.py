
import numpy as np

cover_files = '/data/cbl/BossBase-1.01-cover/{}.pgm'
stego_files = '/data/cbl/BossBase-1.01-s-uniward-0.4/{}.pgm'
train_data_list = 'train_data_list.txt'
test_data_list = 'test_data_list.txt'

file_num = 10000
train_num = 5000

np.random.seed(1234)

index_list = np.arange(1, file_num + 1)

np.random.shuffle(index_list)



with open(train_data_list, 'w') as f:
  for index in index_list[: train_num]:
    f.write('{} 0\n'.format(cover_files.format(index)))
    f.write('{} 1\n'.format(stego_files.format(index)))


with open(test_data_list, 'w') as f:
  for index in index_list[train_num :]:
    f.write('{} 0\n'.format(cover_files.format(index)))
    f.write('{} 1\n'.format(stego_files.format(index)))

