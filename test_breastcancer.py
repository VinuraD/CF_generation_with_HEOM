import gen_cf
import numpy as np

dat=load_breast_cancer(return_X_y=True, as_frame=False)
from sklearn.datasets import load_breast_cancer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

np.random.seed(1)
sc=StandardScaler()

X,y=dat[0],dat[1]
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=13)
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


clf = LogisticRegression(random_state=0).fit(x_train,y_train)
cf_all=[]
ft_type=['ct']*30
n_ft=30
for k in range(len(x_test)):
  i=x_test[k]
  target=1-y_test[k]
  vars,obj_var=gen_cf.create_obj_var(i,n_ft,ft_type)
  cf=[]
  target=0.001 if target==0 else 0.999
  vals=gen_cf.optimize(vars,obj_var,clf,i,target)
  for j in vals:
    cf.append(j.value)
  cf_all.append(cf)

validity=0
for i in range(len(cf_all)):
  if clf.predict(np.array(cf_all[i]).reshape(1,-1))==1-y_test[i]:
    validity+=1

print(validity)