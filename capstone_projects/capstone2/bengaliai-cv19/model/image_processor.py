import cv2
import numpy as np
import pandas as pd


def image_processor_func(X, resize=True, size=(118, 68)):
    """
    This function applies threshold filters and resizes images if resize flag is true.
    By default the function 

    Inputs
    ------
    X -  Image dataframe
    size - New size of images

    Outputs
    -------
    Xoutdf - Output image dataframe post thresholding and resizing
    
    """
    X_length = X.shape[0]
    Xout = {}
    for i in range(X_length):
        image = X[i]
        ret, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if resize:
            resized_image = cv2.resize(thresh, size, interpolation=cv2.INTER_AREA)
            Xout[i] = resized_image.reshape(-1)
        else:
            Xout[i] = thresh.reshape(-1)

    Xoutdf = pd.DataFrame(Xout).T
    return Xoutdf
