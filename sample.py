import sys
import numpy as np
from pathlib import Path
from utils import UCIHAR

ucihar_dir = Path('E:/datasets/UCI_HAR_Dataset/UCI HAR Dataset/')
if not ucihar_dir.exists():
    sys.stderr.write(' >>> Error: {} not found'.format(ucihar_dir))
    sys.exit(1)

ucihar_ds = UCIHAR(ucihar_dir)

all_person_list = np.array(UCIHAR.all_person_id)
train_person_list = all_person_list[:len(all_person_list)//2]
test_person_list = all_person_list[len(all_person_list)//2:]
x_train, y_train, persons_train = ucihar_ds.load_data(train=True, person_list=train_person_list, include_gravity=True)
x_test, y_test, persons_test = ucihar_ds.load_data(train=False, person_list=test_person_list, include_gravity=True)

print('x_train: {}'.format(x_train.shape))
print('y_train: {}'.format(y_train.shape))
print('x_test: {}'.format(x_test.shape))
print('y_test: {}'.format(y_test.shape))
print(persons_train)
print(persons_test)


