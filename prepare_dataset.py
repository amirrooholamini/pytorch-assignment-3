import pandas as pd
import os
import cv2 as cv
import numpy as np

def getData(dataset_dir = 'archive/crop_part1',width=224 ,height=224):
    images = []
    ages = []
    for image_name in os.listdir(dataset_dir)[0:9000]:
        parts = image_name.split('_')
        ages.append(int(parts[0]))
        image = cv.imread(f'{dataset_dir}/{image_name}')
        image = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        images.append(image)
    images = pd.Series(images, name = 'Images')
    ages = pd.Series(ages, name = 'Ages')
    df = pd.concat([images,ages ], axis=1)
    df.head()
    under_4 = []
    for i in range(len(df['Ages'])):
        if(df['Ages'].iloc[i] < 4):
            under_4.append(df.iloc[i])

    under_4 = pd.DataFrame(under_4)
    under_4 = under_4.sample(frac = 0.2)
    up_4 = df[df['Ages'] > 4]
    df = pd.concat([under_4, up_4])
    df = df[df['Ages'] < 90]
    
    X = []
    Y = []
    for i in range(len(df)):
        df['Images'].iloc[i] = cv.resize(df['Images'].iloc[i], (width, height))
        X.append(df['Images'].iloc[i])
        Y.append(df['Ages'].iloc[i])
    X = np.array(X)
    Y = np.array(Y)
    return X,Y
