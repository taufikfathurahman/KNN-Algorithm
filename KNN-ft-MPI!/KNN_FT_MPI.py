
# coding: utf-8

# # K-Nearest Neighbor
# 
# Berikut ini adalah langkah-langkah dalam menyelesaikan permasalahan klasifikasi menggunakan KNN :

# ### 1. Membaca Data Train
# 
# Membaca data yang akan *ditrain* untuk mendapatkan `K` terbaik, akurasi, dan yang akan digunakan untuk memprediksi *data test* nantinya.

# In[1]:


import pandas as pd
from mpi4py import MPI
import time

data_train_from_csv = pd.read_csv('dataTrain.csv')
data_train_from_csv.head()


# ### 2. Menghitung Euclidean Distance
# 
# Dalam melakukan perhitungan jarak pada KNN dapat digunakan bebrapa teori, diantanya adalah *euclidean* dan *manhattan*. Dalam percobaan kali ini akan dilakukan dengan menggunakan teori *euclidean* dalam menghitung jaraknya.
# 
# *Euclidean distance* melakukan perhitungan jarak antara dua buah titik dalam *euclidean space*. Berikut adalah formulanya :
# 
# <img src="assets/euclidean.png">

# In[2]:


import numpy as np
#par
def euclideanDistance(data_train, data_test):
    distanceSum = 0
    for i in range(len(data_train)-1):
        
        distanceSum += (data_train[i]-data_test[i])**2
    return np.sqrt(distanceSum)


# ### 3. KNN Algorithm
#
def distance(begin,end,data_train,data_test):
    distances=[]
    for i in range(begin,end):
        
        distance_loc = euclideanDistance(data_train.iloc[i], data_test)
        distances.append( distance_loc)
    return distances
def predictiondata(b,e,data_test,data_train,k):
    predic=[]
    for x in range(b,e):
                predic.append(kNearestNeighbor(data_train, data_test.iloc[x], k))
    return predic
def predicdataBk(b,e,data_test,data_train,bk):
    predict=[]
    for x in range(b,e):
        predict.append(int(kNearestNeighbor(data_train, data_test.iloc[x], bk)))
    return predict
# KNN merupakan algoritma yang digunakan dalam melakukan memecahkan permasalahan klasifikasi, sehingga menghasilkan output diskrit. Contoh untuk output berupa diskrit adalah output yang hasilnya pasti seperti ketika menghitung 1 + 1 = 2, jawabannya bukan mendekati 2. KNN akan melakukan klasifikasi terhadap objek berdasarkan data pembelajaran yang jaraknya paling dekat dengan objek tersebut.
# 
# KNN akan bekerja berdasarkan jarak minimum dari data baru ke data training untuk menentukan tetangga terdekat. Setelah itu akan didapatkan data mayoritas sebagai hasil prediksi dari data baru tadi.

# In[3]:



import operator

def kNearestNeighbor(data_train, data_test, k):
    distances = {}
    sort = {}
    neighbors = []
    vote_class = {}
    
    for i in range(len(data_train)):
        distance = euclideanDistance(data_train.iloc[i], data_test)
        distances[i] = distance
        
    sorted_distances = sorted(distances.items(), key=operator.itemgetter(1))
    
    for i in range(k):
        neighbors.append(sorted_distances[i][0])
    
    for x in range(len(neighbors)):
        class_in_datatrain = data_train.iloc[neighbors[x]][-1]
    
        if class_in_datatrain in vote_class:
            vote_class[class_in_datatrain] += 1
        else:
            vote_class[class_in_datatrain] = 1

    sorted_vote_class = sorted(vote_class.items(), key=operator.itemgetter(1))
    
    return sorted_vote_class[-1][0]


# ### 4. Menghitung Akurasi
# 
# Akurasi akan didapatkan dari perbandingan hasil prediksi dengan data sebenarnya.

# In[4]:


def predictionAccuracy(prediction_data, data_test):
    accurate = 0
    
    for i in range(len(prediction_data)):
        if prediction_data[i] == data_test.iloc[i][-1]:
            accurate += 1
    return (accurate/len(prediction_data)) * 100


# ### 5. Cross Validation & Tuning Parameter
# 
# **Cross Validation**
# 
# *Cross validation* merupakn metode statistik dalam melakukan evaluasi kinerja dari suatu model atau algoritma dengan melakukan pembagian data menjadi dua subset, yaitu `data pengujian` dan `data pelatihan`.
# 
# > **K-Fold Cross Validation**
# K-Fold Cross Validation merupakan salah satu metode Cross validation yang bekerja dengan melipat data sebanyak K dan melakukan perulangan sebanyak `K` juga. Contohnya untuk `K` = 10:
# 
# <img src="assets/k-fold.png">
# 
# **Tuning Parameter**
# 
# Untuk mendapatkan akurasi yang terbaik saat melakukan klasifikasi di KNN, akan sangat bergantung pada nilai `K` yang kita berikan. Proses dalam mencari `K` terbaik dapat disebut sebagain *Tuning Parameter8* atau *Hyperparameter*.

# In[5]:


import matplotlib.pyplot as plt

def crossValFtTunParam(data_train_from_csv):
    
    comm = MPI.COMM_WORLD
    
    # dapatkan rank proses
    
    # dapatkan total proses berjalan
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    predic_loc=[]
    
    dev_by5 = int(len(data_train_from_csv)/5)
    fold1 = data_train_from_csv.iloc[0:dev_by5]
    fold2 = data_train_from_csv.iloc[dev_by5:dev_by5*2]
    fold3 = data_train_from_csv.iloc[dev_by5*2:dev_by5*3]
    fold4 = data_train_from_csv.iloc[dev_by5*3:dev_by5*4]
    fold5 = data_train_from_csv.iloc[dev_by5*4:]
    
    if(len(data_train_from_csv)/5) > 100:
        k_range = range(1, 101)
    else:
        k_range = range(1, dev_by5+1)
        
    best_k = {}
    
    for k in k_range:
        accuracy_crossval = []
        for i in range(1,6):
            prediction_data = []
            if i == 1:
                data_test = fold1
                data_train = pd.concat([fold2, fold3, fold4, fold5])
            elif i == 2:
                data_test = fold2
                data_train = pd.concat([fold1, fold3, fold4, fold5])
            elif i == 3:
                data_test = fold3
                data_train = pd.concat([fold1, fold2, fold4, fold5])
            elif i == 4:
                data_test = fold4
                data_train = pd.concat([fold1, fold2, fold3, fold5])
            else:
                data_test = fold5
                data_train = pd.concat([fold1, fold2, fold3, fold4])
            var = len(data_test)/size
            prediction_data=[]
            predic_loc=predictiondata(int(rank*var),int((rank+1)*var),data_test,data_train,k)
            prediction_data=comm.allreduce(predic_loc,op=MPI.SUM)
            tmp_accuracy = predictionAccuracy(prediction_data, data_test)
            accuracy_crossval.append(tmp_accuracy)
        
        best_k[k] = sum(accuracy_crossval)/len(accuracy_crossval)
    if rank == 0:    
        plt.scatter(k_range, best_k.values())
        plt.title('Plot Hasil Pencarian K Terbaik')
        plt.xlabel('Nomor K')
        plt.ylabel('Akurasi K')
        plt.show()
    
    K = max(best_k.items(), key=operator.itemgetter(1))[0]
    if rank == 0:
        print(best_k)
        
    return K, best_k[K]


# ### Mendapatkan K terbaik dan Akurasinya
# 
# Memanggil fungsi yang telah dibuat sebelumnya untuk mendapatkan K terbaik beserta menampilakan besar akurasi yang dihasilkan.

# In[6]:
comm = MPI.COMM_WORLD
    
    # dapatkan rank proses
    
    # dapatkan total proses berjalan
size = comm.Get_size()
rank = comm.Get_rank()

start=time.time()
best_k, accuracy = crossValFtTunParam(data_train_from_csv[:100])
if rank == 0:
    print('===============================')
    print('|| Best k     : ', best_k)
    print('|| Accuracy   : ', accuracy, '%')
    print('===============================')
    


# ### Calculate Data Test CSV
# 
# Setelah mengolah data train untuk mendapatkan K terbaik beserta akurasinya, sekarang adalah waktunya untuk mengolah data test untuk mendapatkan kelasnya.
# 
# **1. Membaca Data Test**

# In[7]:


data_test_from_csv = pd.read_csv('dataTest.csv')
data_test_from_csv.head()


# **2. Mendapatkan Prediksi Kelas**

# In[8]:


prediction_data = []
predic_loc=[]
var = len(data_test_from_csv)/size
predic_loc=predictiondata(int(rank*var),int((rank+1)*var),data_test_from_csv,data_train_from_csv,best_k)
prediction_data=comm.allreduce(predic_loc,op=MPI.SUM)



# **3. Menyimpan Data Hasil Prediksi**

# In[9]:

end=time.time()

data_test_from_csv['kelas'] = prediction_data
data_test_from_csv.to_csv('dataHasilPrediksimpi.csv')
if rank == 0:
    print("time :",end-start )
    


# ### Hasil Prediksi

# In[10]:


data_hasil_from_csv = pd.read_csv('dataHasilPrediksi.csv')
data_hasil_from_csv.drop(columns=['Unnamed: 0'])

