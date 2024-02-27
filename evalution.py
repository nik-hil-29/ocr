import save  
import ocr  # Importing the 'ocr' trained model
import mnistdataset as mndt  
import numpy as np #Numpy is for handling large  multi-dimesional arrays and matrices

from sklearn.metrics import classification_report 
 # Importing the 'classification_report' function from the 'sklearn.metrics' module to  generate the  classification reports

lblNms = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
lbl_Nms = [l for l in lblNms]
print("[INFO] evluatng ntwrk ")
prdctions = save.model.predict(mndt.tst_dta_X, batch_size=ocr.btch_sze)


print(classification_report(mndt.tst_dta_Y.argmax(axis=1),
  prdctions.argmax(axis=1), target_names=lbl_Nms,zero_division=0))