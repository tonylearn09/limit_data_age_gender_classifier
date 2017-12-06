import csv
import sys
import os
from sklearn.externals import joblib
import face_lbp
import numpy as np
from skimage import io

model_path = sys.argv[1]   # Model Path
data_path = sys.argv[2]

clf = joblib.load(model_path)

X_test = np.zeros((640, 8496))
# Read image
for i in range(640):
    img = io.imread(os.path.join(data_path, str(i) + '.jpg'))
    X_test[i], _, _ = face_lbp.extract_face(img, False)

result = clf.predict(X_test)

# Store the file
fp1 = open('result.csv','wb')    #use 'wb', can prevent empty line in csv file
cursor1 = csv.writer(fp1)

print('Start Writing csv files')
# Write the predicted results into the csv file
for i in range(640):
    cursor1.writerow([i]+[result[i]])

fp1.close()
