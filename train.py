import numpy as np
import lbp 
from sklearn.svm import NuSVC
from skimage import color
from sklearn.externals import joblib
import os
import face_detector
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#test_images = np.load('test_images.npy')
#test_target_ages = np.load('test_target_ages.npy')
#test_target_genders = np.load('test_target_genders.npy')

if __name__ == '__main__':
    print 'Loading training data and labels'
    train_images = np.load('train_images.npy')
    train_targets = np.load('train_targets.npy')
    print train_images.shape

    print 'Loading validation data and labels'
    val_images = np.load('val_images.npy')
    val_targets = np.load('val_targets.npy')
    print val_images.shape


    nu_svm = NuSVC(nu=0.3, kernel='rbf')
    print 'starting train'
    nu_svm.fit(train_images, train_targets)

    print 'validation acc:'
    print nu_svm.score(val_images, val_targets)
    #print gender_nu_svm.score(val_images_genders, val_target_genders)

    joblib.dump(nu_svm, 'SVM_8496_3.pkl')
