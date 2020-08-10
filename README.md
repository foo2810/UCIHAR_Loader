# UCIHAR_Loader

## About UCIHAR

[UCIHAR](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)

### Sensor Attributes

- Total Acceleration (3 axis): Include the gravity
- Body Acceleration (3 axis): Subtracting the gravity from total
- Gyroscope (3 axis) <- **未対応**

## Expected Dataset Structure

```
UCI HAR Dataset + train + Inertial Signals + 
                + test
                + ...
                :
```

## Sample Code

```python
from pathlib import Path
from utils import UCIHAR

# データセットのパスは"UCI HAR Dataset"を指定
ucihar_dir = Path('...anywhere.../UCI HAR Dataset/')
ucihar_ds = UCIHAR(ucihar_dir)

# 被験者の選択
all_person_list = np.array(UCIHAR.all_person_id)
train_person_list = all_person_list[:len(all_person_list)//2]
test_person_list = all_person_list[len(all_person_list)//2:]

# ロード
## trainがTrueならば訓練データを，Falseならば検証データを返す()
## person_listがNoneの場合はすべての被験者を読み込む
## include_gravityは重力(姿勢情報の有無を指定)
x_train, y_train, persons_train = ucihar_ds.load_data(train=True, person_list=train_person_list, include_gravity=True)
x_test, y_test, persons_test = ucihar_ds.load_data(train=False, person_list=test_person_list, include_gravity=True)

print('x_train: {}'.format(x_train.shape))
print('y_train: {}'.format(y_train.shape))
print('x_test: {}'.format(x_test.shape))
print('y_test: {}'.format(y_test.shape))
print(persons_train)
print(persons_test)
```


