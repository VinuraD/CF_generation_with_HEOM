import cvxpy as cp
# import gurobipy as gp
import numpy as np



def create_obj_var(x,n_ft,ft_type):
  vars=[]
  obj_var=[0]*n_ft
  for i in range(n_ft):
    # vars[i]=cp.Variable()
    # print(ft_type[i])
    if ft_type[i]=='ct':
      # print(vars[i])
      v=cp.Variable()
      obj_var[i]=(cp.norm(v-x[i],1)) #should add denominator
    else:
      v=cp.Variable(boolean=True)
      obj_var[i]=(cp.maximum(1,(v-x[i])/1000))
    vars.append(v)
  vars=cp.Variable(n_ft)
  return vars,obj_var

def check_pred(cf,clf,target):
  cf=cp.hstack(cf)
  # p=cp.log_sum_exp(cp.hstack([-1000,cf@clf.coef_[0].T+clf.intercept_[0]]))
  p=cp.matmul(cf,clf.coef_[0])+clf.intercept_[0]
  # p=cp.exp(p)/(1+cp.exp(p))
  p=cp.sum(cp.exp(p - cp.log(target)))
  return p

# Create constraint.
def optimize(vars,obj_var,clf,x,target):
  # vars.append(cp.Variable())
  # x=x.reshape(1,-1)

  # p=cp.sum(vars*np.transpose(np.array(clf.coef_))) +clf.intercept_[0]
  # print(p)
  # p=1/1+cp.exp(-p)


  # cf=[]
  # for j in vars:
  #   j.value=np.random.randn(1)[0]

  # v=cp.Variable(30)
  constraints=[check_pred(vars,clf,target)<=1]

  # Form objective.
  objective = cp.Minimize(cp.sqrt(cp.sum_squares(cp.vstack(obj_var)))) #
  # objective = cp.Minimize((cp.sum(cp.vstack(obj_var))))
  # objective = cp.Minimize(cp.norm(v-x,1))

  # vars.value = numpy.random.randn(30)


  # Form and solve problem.
  prob = cp.Problem(objective,constraints)
  prob.solve(solver=cp.ECOS_BB,qcp=True,verbose=True) #cp.SCS, cp.ECOS_BB
  # prob.solve(warm_start=True)
  print("status: {}".format(prob.status))
  return vars