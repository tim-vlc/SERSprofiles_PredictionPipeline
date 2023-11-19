import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

import os
import ramanspy as rp
from ramanspy.preprocessing import baseline, normalise
from ramanspy.preprocessing.denoise import SavGol 
from ramanspy import Spectrum

from alive_progress import alive_bar
import numpy as np

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

import warnings
warnings.filterwarnings("ignore")

df_all = pd.read_csv('../../CSVs/diabetes_raw.csv')

# Splitting our data
X = df_all.iloc[:, :-2]
y = df_all['labels']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=100) # 25% test by default

train_set = X_train.copy()
train_set['labels'] = y_train

test_set = X_test.copy()
test_set['labels'] = y_test

def create_pipeline(lb, ub, smoothmeth, bcmeth, normmeth):
    """
    lb: the lower bound for the pixel numbers which need to be kept
    ub: the upper bound for the pixels numbers which need to be kept
    smoothmeth: the preprocessing method used to smooth the spectra
    bcmeth: the preprocessing method used to baseline-correct the spectra
    normmeth: the preprocessing method used to normalize the spectra
    
    returns: the preprocessing pipeline
    """
    cropper = rp.preprocessing.misc.Cropper(region=(lb, ub))
    
    smoothdict = {'SavGol':rp.preprocessing.denoise.SavGol(window_length=22, polyorder=5), 
                  'Whittaker':rp.preprocessing.denoise.Whittaker(),
                  'Gaussian':rp.preprocessing.denoise.Gaussian()}
    
    bcdict = {'IASLS':rp.preprocessing.baseline.IASLS(),
              'AIRPLS':rp.preprocessing.baseline.AIRPLS(),
              'ARPLS':rp.preprocessing.baseline.ARPLS(),
              'ASPLS':rp.preprocessing.baseline.ASPLS()}
    
    normdict = {'Vector':rp.preprocessing.normalise.Vector(),
                'MinMax':rp.preprocessing.normalise.MinMax(),
                'MaxIntensity':rp.preprocessing.normalise.MaxIntensity(),
                'AUC':rp.preprocessing.normalise.AUC()}
    
    pipe = rp.preprocessing.Pipeline([
        cropper, # cropping the wanted region
        rp.preprocessing.despike.WhitakerHayes(), # Cosmic ray removal
    ])
    
    procctuple = [(smoothdict, smoothmeth), (bcdict, bcmeth), (normdict, normmeth)]
    for dict_, meth in procctuple:
        if meth is None:
            continue
        
        pipe.append(dict_[meth])
    return pipe

def preprocess_metrics(train_set, test_set):
    lb, ub = 150, 1000       # upper and lower bounds for cropping
    pixel_num = 1650

    smoothdict = {'SavGol':rp.preprocessing.denoise.SavGol(window_length=22, polyorder=5), 
                  'Whittaker':rp.preprocessing.denoise.Whittaker(),
                  'Gaussian':rp.preprocessing.denoise.Gaussian()}
    
    bcdict = {'IASLS':rp.preprocessing.baseline.IASLS(),
              'AIRPLS':rp.preprocessing.baseline.AIRPLS(),
              'ARPLS':rp.preprocessing.baseline.ARPLS(),
              'ASPLS':rp.preprocessing.baseline.ASPLS()}
    
    normdict = {'Vector':rp.preprocessing.normalise.Vector(),
                'MinMax':rp.preprocessing.normalise.MinMax(),
                'MaxIntensity':rp.preprocessing.normalise.MaxIntensity(),
                'AUC':rp.preprocessing.normalise.AUC()}
    
    smoothlist = __builtins__.list(smoothdict.keys()) + ['None']
    bclist = __builtins__.list(bcdict.keys()) + ['None']
    normlist = __builtins__.list(normdict.keys()) + ['None']
    
    X_train, y_train = train_set.iloc[:, :-1], train_set['labels']
    X_test, y_test = test_set.iloc[:, :-1], test_set['labels']
    
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X_train, y_train)
    
    # Performance values
    predictions = clf.predict(X_test)
    conf_mat = confusion_matrix(predictions, y_test).ravel()
    acc = accuracy_score(y_test, predictions)
    f1 = f1_score(predictions, y_test, average='weighted')
    
    df = pd.DataFrame({'processing_methods':'Raw', 'accuracy':acc, 
                             'F1 score': f1, 'Confusion Matrix':conf_mat})
    df_list = [df]
    
    
    
    for smoothmeth in smoothlist:
        for bcmeth in bclist:
            for normmeth in normlist:
                processing_method = smoothmeth + '-' + bcmeth + '-' + normmeth
                
                smoothmeth = None if smoothmeth == 'None' else smoothmeth
                bcmeth = None if bcmeth == 'None' else bcmeth
                normmeth = None if normmeth == 'None' else normmeth
                
                train_processed = preprocess(train_set, lb, ub, pixel_num, smoothmeth=smoothmeth, bcmeth=bcmeth, normmeth=normmeth)
                test_processed = preprocess(test_set, lb, ub, pixel_num, smoothmeth=smoothmeth, bcmeth=bcmeth, normmeth=normmeth)
                train_processed = train_processed.dropna()
                test_processed = test_processed.dropna()
                
                X_train, y_train = train_processed.iloc[:, :-1], train_processed['labels']
                X_test, y_test = test_processed.iloc[:, :-1], test_processed['labels']
                
                clf = QuadraticDiscriminantAnalysis()
                clf.fit(X_train, y_train)

                # Performance values
                predictions = clf.predict(X_test)
                conf_mat = confusion_matrix(predictions, y_test).ravel()
                acc = accuracy_score(y_test, predictions)
                f1 = f1_score(predictions, y_test, average='weighted')
                
                df = pd.DataFrame({'processing_methods':processing_method, 'accuracy':acc, 
                             'F1 score': f1, 'Confusion Matrix':conf_mat})
                df_list.append(df)
                
    score_df = pd.concat(df_list, axis=0)
                
    return score_df

def preprocess(data, lb, ub, pixel_num, smoothmeth=None, bcmeth=None, normmeth=None):
    """
    data: the dataframe whose spectra need to be preprocessed
    lb: the lower bound for the pixel numbers which need to be kept
    ub: the upper bound for the pixels numbers which need to be kept
    pixel_num: the amount of pixels of the RS in the dataframe given
    
    returns: the preprocessed dataframe
    """
    num_pixels = ub - lb + 1    # number of pixels in the processed dataframe

    pipe = create_pipeline(lb, ub, smoothmeth=smoothmeth, normmeth=normmeth, bcmeth=bcmeth)

    spectra_array = np.zeros((len(data), num_pixels))
    labels = data['labels']
    #patient_num = data['patient#']
    length = len(data)
    
    with alive_bar(length, force_tty=True) as bar:
        for j in range(length):
            spectrum = Spectrum(data.iloc[j, :-1], range(pixel_num))
            proc_spectrum = pipe.apply(spectrum).spectral_data
            spectra_array[j, :] = proc_spectrum
            bar()

    df = pd.DataFrame(spectra_array, columns = __builtins__.list(range(num_pixels)))
    df['labels'] = __builtins__.list(labels.to_numpy())
    #df['patient#'] = list(patient_num.to_numpy())
    
    print('Finished preprocess!')
    
    return df

preprocess_metrics(train_set, test_set).to_csv('PreprocessingMetrics.csv', index=False)