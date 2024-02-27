import numpy as np  
#Numpy is for handling of   multi-dimesional arrays and matrices
import pandas as pd 
 # Pandas is  library for data manipulation and data analysis


def load_custom_dataset(dtaset_pth):
    dta = []
    lbls = []  
    df = pd.read_csv(dtaset_pth) 

    # Iterate over each row in the dataset
    for _, row in df.iterrows():
        cstm_lbl = int(row[0]) 
        cstm_imge = np.array([int(x) for x in row[1:]], dtype="uint8")  
        cstm_imge = cstm_imge.reshape((28, 28))  
        dta.append(cstm_imge)  
        lbls.append(cstm_lbl) 

    cstm_dta = np.array(dta, dtype='float32')  
    cstm_lbls = np.array(lbls, dtype="int") 
    return (cstm_dta, cstm_lbls) 