import numpy as np
from numpy import linalg
import pandas as pd
from sympy import diff, symbols, sympify, Symbol, poly
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from time import time
from arimafd import diff_integ
import pickle
from scipy.signal import argrelextrema

class online_arima:
    def __init__(self, order=4, lrate=0.001, random_state=42):
        self.order=order
        self.lrate=lrate
        self.random_state=random_state

    def fit(self, data, init_w=None):
        data=np.array(data)
        self.data=data
        np.random.seed(self.random_state)
        self.pred = np.zeros(data.shape[0] + 1)*np.nan
        self.w = np.random.rand(self.order+1)*0.01 if init_w is None else init_w.copy()
        self.ww=pd.DataFrame([self.w])
        self.diff=np.zeros(len(self.w))
        self.dif_w = pd.DataFrame([self.w])
        for i in range(self.order, data.shape[0]):
            self.pred[i] = self.w[:-1] @ data[i-self.order:i] + self.w[-1]          
            self.diff[:-1]= np.tanh(self.pred[i] - data[i])*data[i-self.order:i]
            self.diff[-1] = np.tanh(self.pred[i] - data[i])
            self.w -= self.lrate * self.diff

            self.ww=self.ww.append([self.w], ignore_index=True)
            self.dif_w = self.dif_w.append([self.diff], ignore_index=True)
        self.iii=i
        self.pred[-1]=self.w[:-1] @ data[-self.order:] + self.w[-1]                


class online_LR:
    def __init__(self, order=4, lrate=0.001, random_state=42):
        self.order=order
        self.lrate=lrate
        self.random_state=random_state

    def fit(self, data, init_w=None):
        data=np.array(data)
        self.data=data
        np.random.seed(self.random_state)
        self.pred = np.zeros(data.shape[0] + 1)*np.nan
        self.w = np.random.rand(self.order+1)*0.01 if init_w is None else init_w.copy()
        self.ww=pd.DataFrame([self.w])
        self.diff=np.zeros(len(self.w))

        self.dif_w = pd.DataFrame([self.w])
        for i in range(self.order, data.shape[0]):
            self.pred[i] = self.w[:-1] @ data[i-self.order:i] + self.w[-1]          
            self.diff[:-1]= np.tanh(self.pred[i] - data[i])*data[i-self.order:i]
            self.diff[-1] = np.tanh(self.pred[i] - data[i])
            self.w -= self.lrate * self.diff

            self.ww=self.ww.append([self.w], ignore_index=True)
            self.dif_w = self.dif_w.append([self.diff], ignore_index=True)
        self.iii=i
        self.pred[-1]=self.w[:-1] @ data[-self.order:] + self.w[-1]   

class my_NN:
    def __init__(self, input_size, K=5):
        self.K = K
        self.weights1 = np.random.rand(input_size, K)
        self.weights2 = np.random.rand(K, 1)
        self.b1 = np.zeros(K)
        self.b2 = np.zeros(1)
        self.K = K
    
    def learn(self, data, right, lr):
        if len(data.shape) == 1:
            data = data.reshape(1, -1)
        if len(right.shape) == 1:
            right = right.reshape(1, -1)
        
        self.tmp = data @ self.weights1 + self.b1
        self.tmp2 = self.tmp * (self.tmp > 0)
        self.res = self.tmp2 @ self.weights2 + self.b2
        
        diff_x = 2 * (self.res - right)

        dw2 = self.tmp2.T @ diff_x
        db2 = diff_x.sum(axis=0)
        next_grad = diff_x @ self.weights2.T
        
        next_grad = next_grad * (self.tmp > 0)
        
        dw1 = data.T @ next_grad
        db1 = next_grad.sum(axis=0)
        
        self.weights1 -= lr * dw1
        self.b1 -= lr * db1
        self.weights2 -= lr * dw2
        self.b2 -= lr * db2
        
        return np.r_[dw1.ravel(), db1.ravel(), dw2.ravel(), db2.ravel()]
    
    def get_weights(self):
        return np.r_[self.weights1.ravel(), self.b1.ravel(), self.weights2.ravel(), self.b2.ravel()]
    
    def loss(self, data, right):
        self.tmp = data @ self.weights1 + self.b1
        self.tmp2 = self.tmp * (self.tmp > 0)
        res = self.tmp2 @ self.weights2 + self.b2
        return np.mean((res - right) ** 2)
        

class online_NN:
    def __init__(self, order=4, lrate=1e-7, random_state=42, K=4):
        self.order=order
        self.lrate=lrate
        self.random_state=random_state
        self.K = K
        self.model = my_NN(order, K)
    
    def fit(self, data):
        data=np.array(data)
        self.data=data
        np.random.seed(self.random_state)
        
        self.w = self.model.get_weights()
        self.diff=np.zeros(len(self.w))
        self.dif_w = pd.DataFrame([self.w])
        for i in range(self.order, data.shape[0]):
            self.diff = self.model.learn(data[i-self.order:i], data[i], self.lrate)
            self.dif_w = self.dif_w.append([self.diff], ignore_index=True)
        self.iii=i



class anomaly_detection:

    def __init__(self, method: str = 'arima'):
        if method == 'arima':
            self.model = online_arima
        elif method == 'LR':
            self.model = online_LR
        elif method == 'NN':
            self.model = online_NN
        else:
            raise "NO METHOD"
        self.method = method
    
    def set_data(self, data):
        self.indices = data.index
        self.data = data


    def generate_tensor(self,ar_order=None, K=None, lrate=1e-3):
        data = self.data.copy()

        ss = StandardScaler()
        mms = MinMaxScaler()

        data=ss.fit_transform(data.copy())

        tensor = np.zeros((data.shape[0]-ar_order,data.shape[1],ar_order+1))
        if self.method == 'NN':
            tensor = np.zeros((data.shape[0]-ar_order,data.shape[1],(ar_order+1) * K + K + 1))
        j=0
        for i in range(data.shape[1]):
            t1=time()
            kkk=0

            if self.method == 'arima': 
                diffr=diff_integ([1])
                dif = diffr.fit_transform(data[:,i])
            else:
                dif = data[1:,i]
            if K is not None:
                model=self.model(ar_order, lrate=lrate, K=K)
            else:
                model=self.model(ar_order, lrate=lrate)
            model.fit(dif)
            t2=time()
            print('Time seconds:', t2-t1)

            tensor[:,i,:] = model.dif_w.values
        self.tensor = tensor
        return tensor
    
    def save_tensors(self, all_files, order, K=None, save_path='C:\data\pickle_arima\numenta\tensors_linear.pickle', lrate=1e-3):
        tensors = []
        for i in range(len(all_files)):
            df = pd.read_csv(all_files[i],index_col = 'timestamp', parse_dates=True)
            self.set_data(df)
            tensors.append(self.generate_tensor(ar_order=order, K=K, lrate=lrate))

        with open(save_path, 'wb') as handle:
            pickle.dump(tensors, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def proc_tensor(self,window=100,No_metric=1,window_insensitivity=100):

        tensor = self.tensor.copy()
        df = pd.DataFrame(tensor.reshape(len(tensor),-1),index=self.indices[-len(tensor):])
        if No_metric == 1:
            metric = (df.rolling(window).max().abs()/df.rolling(window).std().abs()).mean(axis=1)
        elif No_metric == 2:
            metric = df.abs().max(axis=1)
        elif No_metric == 3:
            metric = np.sqrt(np.square(df.diff(1)).sum(axis=1))
            metric[0] = np.nan
        elif No_metric == 4:
            member1 = df.abs().rolling(window).max().drop(df.index[-1],axis=0).values
            member2 = df.abs().drop(df.index[0],axis=0).values
            metric = np.abs(member2 - member1)
            sc = StandardScaler()
            metric = sc.fit_transform(metric).max(axis=1)
            metric = pd.Series(np.append(np.nan*np.zeros(1), metric),index=df.index)
        elif No_metric == 5:
            window = int(len(self.data)/50)
            def norm(x,mu=0,std=1/4.5):
                return np.exp(-1/2*((x-mu)/std)**2) 
            my_exp_weighted = lambda arr: np.sum(norm(np.linspace(-1,1,len(arr)))*np.array(arr))
            metric1 = df.abs().max(axis=1)
            metric2 = metric1.rolling(window, center=True).apply(my_exp_weighted)
            
            metric2 = metric2.drop(metric2[metric2.index.duplicated()].index).copy()
            ucl = metric2.sort_values()[int(len(metric2)*9/10)]
            raw_metric = metric2.copy()
            raw_metric[raw_metric<ucl]=0
            list_num_index_of_loc_max = argrelextrema(np.array(raw_metric), np.greater)
            series_of_local_maxima = raw_metric[raw_metric.index[list_num_index_of_loc_max]]
            bin_metric = pd.Series(np.zeros(len(metric2)),index=metric2.index)
            bin_metric[series_of_local_maxima.index]=1
            self.bin_metric = bin_metric
            return bin_metric


        ucl = metric.mean() + 3*metric.std()
        lcl = metric.mean() - 3*metric.std()
        self.metric = metric
        self.ucl = ucl
        self.lcl = lcl            
        bin_metric = ((metric > ucl) | (metric < lcl)).astype(int)

        winn = window_insensitivity
        for i in range(len(bin_metric)-winn):
            if ((bin_metric.iloc[i] == 1.0) & (bin_metric[i:i+winn].sum()>1.0)):
                bin_metric[i+1:i+winn]=np.zeros(winn-1)
        self.bin_metric = bin_metric        
        return bin_metric


    def evaluate_nab(self,anomaly_list,table_of_coef=None):
        if table_of_coef is None:
            table_of_coef = pd.DataFrame([[1.0,-0.11,1.0,-1.0],
                                 [1.0,-0.22,1.0,-1.0],
                                  [1.0,-0.11,1.0,-2.0]])
            table_of_coef.index = ['Standart','LowFP','LowFN']
            table_of_coef.index.name = "Metric"
            table_of_coef.columns = ['A_tp','A_fp','A_tn','A_fn']

        alist = anomaly_list.copy()
        bin_metric = self.bin_metric.copy()

        Scores,Scores_perfect,Scores_null=[],[],[]
        for profile in ['Standart','LowFP','LowFN']:       
            A_tp = table_of_coef['A_tp'][profile]
            A_fp = table_of_coef['A_fp'][profile]
            A_fn = table_of_coef['A_fn'][profile]
            def sigm_scale(y,A_tp,A_fp,window=1):
                return (A_tp-A_fp)*(1/(1+np.exp(5*y/window))) + A_fp
            score = 0
            if len(alist)>0:
                score += bin_metric[:alist[0][0]].sum()*A_fp
            else:
                score += bin_metric.sum()*A_fp
            for i in range(len(alist)):
                if i<=len(alist)-2:
                    win_space = bin_metric[alist[i][0]:alist[i+1][0]].copy()
                else:
                    win_space = bin_metric[alist[i][0]:].copy()
                win_fault = bin_metric[alist[i][0]:alist[i][1]]
                slow_width = int(len(win_fault)/4)

                if len(win_fault) + slow_width >= len(win_space):
                    win_fault_slow = win_fault.copy()
                else:
                    win_fault_slow= win_space[:len(win_fault)  +  slow_width]

                win_fp = win_space[-len(win_fault_slow):]

                if win_fault_slow.sum() == 0:
                    score+=A_fn
                else:
                    tr = pd.Series(win_fault_slow.values,index = range(-len(win_fault),len(win_fault_slow)-len(win_fault)))
                    tr_values= tr[tr==1].index[0]
                    tr_score = sigm_scale(tr_values, A_tp,A_fp,slow_width)
                    score += tr_score
                    score += win_fp.sum()*A_fp
            Scores.append(score)
            Scores_perfect.append(len(alist)*A_tp)
            Scores_null.append(len(alist)*A_fn)
        self.Scores,self.Scores_null,self.Scores_perfect = np.array(Scores), np.array(Scores_null) ,np.array(Scores_perfect)
        return np.array(Scores), np.array(Scores_null) ,np.array(Scores_perfect)

def get_score(list_metrics):
    sum1 = np.zeros((3,3))
    for i in range(len(list_metrics)):
        sum1 += list_metrics[i]
    desc = ['Standart','LowFP','LowFN']    
    for t in range(3):
        print(desc[t],' - ', 100*(sum1[0,t]-sum1[1,t])/(sum1[2,t]-sum1[1,t]))