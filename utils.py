import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, Optional

__all__ = ['UCIHAR']


class UCIHAR(object):
    all_person_id = list(range(1, 31))

    def __init__(self, ucihar_dir):
        if type(ucihar_dir) == str: ucihar_dir = Path(ucihar_dir)
        self.ucihar_dir = ucihar_dir
        self.load_meta()
    
    def load_meta(self):
        # train
        train_labels = pd.read_csv(str(self.ucihar_dir/'train'/'y_train.txt'), header=None)
        train_subjects = pd.read_csv(str(self.ucihar_dir/'train'/'subject_train.txt'), header=None)
        self.train_metas = pd.concat([train_labels, train_subjects], axis=1)
        self.train_metas.columns = ['activity', 'person_id']

        # test
        test_labels = pd.read_csv(str(self.ucihar_dir/'test'/'y_test.txt'), header=None)
        test_subjects = pd.read_csv(str(self.ucihar_dir/'test'/'subject_test.txt'), header=None)
        self.test_metas = pd.concat([test_labels, test_subjects], axis=1)
        self.test_metas.columns = ['activity', 'person_id']
   
    def load_data(self, train:bool=True, person_list:Optional[list]=None, include_gravity:bool=True) -> tuple:
        """Sliding-Windowをロード

        Parameters
        ----------
        train: bool
            select train data or test data. if True then return train data.
        person_list: Option[list]
            specify persons.
        include_gravity: bool
            select whether or not include gravity information.
       
        Returns
        -------
        sensor_data:
            sliding-windows
        labels:
            activity labels
        person_id_list:
            the list of person id
        """

        if include_gravity:
            if train:
                x = pd.read_csv(str(self.ucihar_dir/'train'/'Inertial Signals'/'total_acc_x_train.txt'), sep='\s+', header=None).to_numpy()
                y = pd.read_csv(str(self.ucihar_dir/'train'/'Inertial Signals'/'total_acc_y_train.txt'), sep='\s+', header=None).to_numpy()
                z = pd.read_csv(str(self.ucihar_dir/'train'/'Inertial Signals'/'total_acc_z_train.txt'), sep='\s+', header=None).to_numpy()
                metas = self.train_metas
            else:
                x = pd.read_csv(str(self.ucihar_dir/'test'/'Inertial Signals'/'total_acc_x_test.txt'), sep='\s+', header=None).to_numpy()
                y = pd.read_csv(str(self.ucihar_dir/'test'/'Inertial Signals'/'total_acc_y_test.txt'), sep='\s+', header=None).to_numpy()
                z = pd.read_csv(str(self.ucihar_dir/'test'/'Inertial Signals'/'total_acc_z_test.txt'), sep='\s+', header=None).to_numpy()
                metas = self.test_metas
        else:
            if train:
                x = pd.read_csv(str(self.ucihar_dir/'train'/'Inertial Signals'/'body_acc_x_train.txt'), sep='\s+', header=None).to_numpy()
                y = pd.read_csv(str(self.ucihar_dir/'train'/'Inertial Signals'/'body_acc_y_train.txt'), sep='\s+', header=None).to_numpy()
                z = pd.read_csv(str(self.ucihar_dir/'train'/'Inertial Signals'/'body_acc_z_train.txt'), sep='\s+', header=None).to_numpy()
                metas = self.train_metas
            else:
                x = pd.read_csv(str(self.ucihar_dir/'test'/'Inertial Signals'/'body_acc_x_test.txt'), sep='\s+', header=None).to_numpy()
                y = pd.read_csv(str(self.ucihar_dir/'test'/'Inertial Signals'/'body_acc_y_test.txt'), sep='\s+', header=None).to_numpy()
                z = pd.read_csv(str(self.ucihar_dir/'test'/'Inertial Signals'/'body_acc_z_test.txt'), sep='\s+', header=None).to_numpy()
                metas = self.test_metas
 
        flags = np.zeros((x.shape[0],), dtype=np.bool)
        if person_list is None: person_list = np.array(UCIHAR.all_person_id)
        for person_id in person_list:
            flags = np.logical_or(flags, np.array(metas['person_id'] == person_id))

        x, y, z = x[flags], y[flags], z[flags]
        
        sensor_data = np.concatenate([x[:, np.newaxis, :], y[:, np.newaxis, :], z[:, np.newaxis, :]], axis=1)
        labels = metas['activity'].to_numpy()[flags]
        labels -= 1 # scale: [1, 6] => scale: [0, 5]
        person_id_list = np.array(metas.iloc[flags]['person_id'])

        return sensor_data, labels, person_id_list


if __name__ == '__main__':
    import sys
    ucihar_dir = Path('E:/datasets/UCI_HAR_Dataset/UCI HAR Dataset/')
    if not ucihar_dir.exists():
        sys.stderr(' >>> Error: {} not found'.format(ucihar_dir))
        sys.exit(1)
    
    ucihar_ds = UCIHAR(ucihar_dir)

    all_person_list = np.array(UCIHAR.all_person_id)
    p = np.random.permutation(len(all_person_list))
    train_person_list = all_person_list[p][:len(all_person_list)//2]
    test_person_list = all_person_list[p][len(all_person_list)//2:]
    x_train, y_train, meta_train = ucihar_ds.load_data(train=True, person_list=train_person_list)
    x_test, y_test, meta_test = ucihar_ds.load_data(train=False, person_list=test_person_list)

    print('x_train: {}'.format(x_train.shape))
    print('y_train: {}'.format(y_train.shape))
    print('x_test: {}'.format(x_test.shape))
    print('y_test: {}'.format(y_test.shape))
    print(meta_train)
    print(meta_test)
 

