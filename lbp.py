from skimage import feature
import numpy as np
import scipy

class LBP:
    def __init__(self, num_neighbors, radius):
        # store the number of points and radius
        self.num_neighbors = num_neighbors
        self.radius = radius

    def hist(self, image, eps = 1e-6):
        # Build the histogram from lbp
        lbp = feature.local_binary_pattern(image, self.num_neighbors, self.radius, method='nri_uniform')
        #return lbp.ravel()
        '''(histogram, _) = np.histogram(lbp.ravel(), 
                                      bins=np.arange(0, self.num_neighbors + 3),
                                      range=(0, self.num_neighbors + 2))'''
        window_size = 10
        block_size =12 
        histogram = np.zeros(59*block_size*block_size)
        for x in range(block_size): 
            for y in range(block_size):
                tmp_1 = lbp[window_size*y:window_size*(y+1), window_size*x: window_size*(x+1)]
                (temp_hist, a) = np.histogram(tmp_1.ravel(), bins=np.arange(0,60))
                # Normalization
                temp_hist = temp_hist.astype('float')
                temp_sum = np.sum(temp_hist)
                temp_hist /= (temp_sum + eps)
                # Store the result into histogram
                #print y, x
                #histogram[59*(y+block_size*x):59*(y+block_size*x+1)] = temp_hist
                histogram[59*(y+10*x):59*(y+10*x+1)] = temp_hist

        #print histogram
        return histogram
