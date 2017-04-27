import os
import numpy as np
import pandas as pd
#from scipy.misc import imread
from sklearn.metrics import accuracy_score
import tensorflow as tf
from __future__ import print_function
#from six.moves import cPickle as pickle
#from six.moves import range
import time
data_dir='/home/ym598d/Dispatch'
os.path.exists(data_dir)

dsp_weekly = pd.read_csv(os.path.join(data_dir, 'dsp_Aggregated_turf_weekly_all_withPrev4Weeks_avg_withZeros.csv'),sep='|',header=None)
dsp_weekly.columns=['network_type', 'VOLTYPE',  'JC'  ,'areaname_turf','WeekofMonth', 'year','network_type1', 'VOLTYPE1',  'JC1' ,'Org','areaname_turf1','year1','WeekofMonth1', 'cnt1', 'cnt_prev','cnt_prev2','cnt_prev3','cnt_prev4']
dsp_weekly=dsp_weekly[dsp_weekly.WeekofMonth !='WeekofMonth']
dsp_weekly[['cnt1', 'cnt_prev','cnt_prev2','cnt_prev3','cnt_prev4']]= dsp_weekly[['cnt1', 'cnt_prev','cnt_prev2','cnt_prev3','cnt_prev4']].fillna(0)
#dsp_weekly.WeekofMonth=pd.to_numeric(dsp_weekly.WeekofMonth)
dsp_weekly.WeekofMonth=dsp_weekly.WeekofMonth.astype(int)
dsp_weekly['cnt_prev_MA'] = (dsp_weekly['cnt_prev']+dsp_weekly['cnt_prev2'])/2
dsp_weekly['cnt_prev1_MA'] = (dsp_weekly['cnt_prev2']+dsp_weekly['cnt_prev3'])/2
dsp_weekly['cnt_prev2_MA'] = (dsp_weekly['cnt_prev3']+dsp_weekly['cnt_prev4'])/2
dsp_weekly['cnt_prev1_MA1'] = (dsp_weekly['cnt_prev2']+dsp_weekly['cnt_prev3']+dsp_weekly['cnt_prev'])/3
dsp_weekly['cnt_prev2_MA1'] = (dsp_weekly['cnt_prev3']+dsp_weekly['cnt_prev4']+dsp_weekly['cnt_prev2'])/3
dsp_weekly['cnt_prev_avg'] = (dsp_weekly['cnt_prev3']+dsp_weekly['cnt_prev4']+dsp_weekly['cnt_prev2']+dsp_weekly['cnt_prev'])/4
dsp_weekly = dsp_weekly[~((dsp_weekly.year==2015) & (dsp_weekly.WeekofMonth<=4))]
dsp_weekly = dsp_weekly[dsp_weekly['year']>=2015]
dsp_weekly = dsp_weekly[dsp_weekly['WeekofMonth']<=52]
#dsp_weekly['WeekofMonth] = as.numeric(dsp_weekly['WeekofMonth)
dsp_weekly['MonthofYear'] = np.ceil(dsp_weekly['WeekofMonth']/4.5)
#dsp_weekly['yearmonth] = paste(as.character(dsp_weekly['year),dsp_weekly['MonthofYear,sep="_")
dsp_weekly['WeekofMonth_f'] = dsp_weekly['WeekofMonth'].astype(str)
dsp_weekly['dsp_prev4weeks_sd'] = dsp_weekly[['cnt_prev','cnt_prev2','cnt_prev3','cnt_prev4']].std(axis=1)
dsp_weekly['areaname_turf'] = dsp_weekly['areaname_turf'].astype(str)
#dsp_weekly['holiday1'] = as.factor(ifelse(dsp_weekly['WeekofMonth==1,"Newyear",ifelse(dsp_weekly['WeekofMonth %in% c(22,27,36,37),"otherHoliday",ifelse(dsp_weekly['WeekofMonth %in% c(47,48),"Thanksgiving",ifelse(dsp_weekly['WeekofMonth %in% c(51,52),"Christmas","ord")))))
#dsp_weekly['holiday'] = as.factor(ifelse(dsp_weekly['holiday1=='ord',0,ifelse(dsp_weekly['holiday1=="otherHoliday",1,2)))
#dsp_weekly['if_holiday'] = as.factor(ifelse(dsp_weekly['holiday1=='ord',0,1))
dsp_weekly['has_dsp'] = dsp_weekly['cnt1'].apply(lambda x: 1 if x>0 else 0)


dummies = pd.get_dummies(dsp_weekly['network_type']).rename(columns=lambda x: 'networkType_' + str(x))
dsp_weekly = pd.concat([dsp_weekly, dummies], axis=1)

dummies = pd.get_dummies(dsp_weekly['VOLTYPE']).rename(columns=lambda x: 'voltype_' + str(x))
dsp_weekly = pd.concat([dsp_weekly, dummies], axis=1)
#dsp_weekly = dsp_weekly.drop(['VOLTYPE'], inplace=True, axis=1)
dummies = pd.get_dummies(dsp_weekly['JC']).rename(columns=lambda x: 'JC_' + str(x))
dsp_weekly = pd.concat([dsp_weekly, dummies], axis=1)
#dsp_weekly = dsp_weekly.drop(['JC'], inplace=True, axis=1)
dummies = pd.get_dummies(dsp_weekly['WeekofMonth_f']).rename(columns=lambda x: 'WeekofMonth_' + str(x))
dsp_weekly = pd.concat([dsp_weekly, dummies], axis=1)

dummies = pd.get_dummies(dsp_weekly['MonthofYear']).rename(columns=lambda x: 'MonthofYear_' + str(x))
dsp_weekly = pd.concat([dsp_weekly, dummies], axis=1)
#dummies = pd.get_dummies(dsp_weekly['has_dsp']).rename(columns=lambda x: 'has_dsp_' + str(x))
#dsp_weekly = pd.concat([dsp_weekly, dummies], axis=1)

one_hot=True
if one_hot:
    dsp_weekly['label']=dsp_weekly.has_dsp
else:
    dsp_weekly['label']=dsp_weekly.cnt


dsp_weekly.drop(['network_type','VOLTYPE','VOLTYPE1', 'JC','WeekofMonth_f','MonthofYear','network_type1','JC1','Org','areaname_turf1','year1','WeekofMonth1','cnt1','has_dsp'], inplace=True, axis=1)
dsp_weekly=dsp_weekly.iloc[np.random.permutation(len(dsp_weekly))]

dsp_weekly_tr=dsp_weekly[(dsp_weekly.year<=2016)&(dsp_weekly.WeekofMonth<45)]
dsp_weekly_ts=dsp_weekly[~((dsp_weekly.year<=2016)&(dsp_weekly.WeekofMonth<45))]
dsp_weekly_tr.drop(['areaname_turf','year','WeekofMonth'],axis=1,inplace=True)
dsp_weekly_ts.drop(['areaname_turf','year','WeekofMonth'],axis=1,inplace=True)

split_size = int(dsp_weekly_tr.shape[0]*0.9)

train_x, val_x = dsp_weekly_tr.iloc[:split_size,:-1], dsp_weekly_tr.iloc[split_size:,:-1]
train_y, val_y = dsp_weekly_tr.iloc[:split_size,-1], dsp_weekly_tr.iloc[split_size:,-1]
train_x = np.array(train_x)
val_x = np.array(val_x)
#val_x = np.array(dsp_weekly_ts.iloc[:,:-1])
train_y = np.array(train_y)
val_y=np.array(val_y)
#val_y = np.array(dsp_weekly_ts.iloc[:,-1])
train = dsp_weekly_tr.iloc[:split_size,:]
val = dsp_weekly_tr.iloc[split_size:,:]
test = dsp_weekly_ts
test_x=np.array(dsp_weekly_ts.iloc[:,:-1])
test_y=np.array(dsp_weekly_ts.iloc[:,-1])
#test = np.array(test)

###helper functions###############
def dense_to_one_hot(labels_dense, num_classes=2):
    """Convert class labels from scalars to one-hot vectors"""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    
    return labels_one_hot

def preproc(unclean_batch_x):
    """Convert values to range 0-1"""
    temp_batch = unclean_batch_x / unclean_batch_x.max()
    
    return temp_batch

def batch_creator(batch_size, dataset_length, dataset_name,one_hot=True):
    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length, batch_size)
    
    batch_x = eval(dataset_name + '_x')[[batch_mask]].reshape(-1, input_num_units)
    batch_x = preproc(batch_x)
    
    if dataset_name == 'train':
        batch_y = eval(dataset_name).iloc[batch_mask, -1].values
        if one_hot:
            batch_y = dense_to_one_hot(batch_y)
        
    return batch_x, batch_y


### set all variables ###############################

# number of neurons in each layer

input_num_units = 16+2+53+52+12+11 #Network_Type +JC+VOL+WeekofMonth+MonthofYear +MAs
hidden_num_units1 = 500
hidden_num_units2 = 100
hidden_num_units3 = 20
output_num_units = 2

# define placeholders
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

# set remaining variables
epochs = 5
batch_size = 128
learning_rate = 0.01

### define weights and biases of the neural network (refer this article if you don't understand the terminologies)
seed = 128
rng = np.random.RandomState(seed)
weights = {
    'hidden1': tf.Variable(tf.random_normal([input_num_units, hidden_num_units1], seed=seed)),
    'hidden2': tf.Variable(tf.random_normal([hidden_num_units1, hidden_num_units2], seed=seed)),
    'hidden3': tf.Variable(tf.random_normal([hidden_num_units2, hidden_num_units3], seed=seed)),
    'output': tf.Variable(tf.random_normal([hidden_num_units1, output_num_units], seed=seed))
}

biases = {
    'hidden1': tf.Variable(tf.random_normal([hidden_num_units1], seed=seed)),
    'hidden2': tf.Variable(tf.random_normal([hidden_num_units2], seed=seed)),
    'hidden3': tf.Variable(tf.random_normal([hidden_num_units3], seed=seed)),
    'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
}

hidden_layer1 = tf.add(tf.matmul(x, weights['hidden1']), biases['hidden1'])

hidden_layer2 = tf.add(tf.matmul(hidden_layer1, weights['hidden2']), biases['hidden2'])

hidden_layer3 = tf.add(tf.matmul(hidden_layer2, weights['hidden3']), biases['hidden3'])
hidden_layer = tf.nn.relu(hidden_layer1)

output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']
#output_layer = tf.case([(tf.less(output_layer, 0.000), 0.000)], default=output_layer)


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))
cost1 = tf.reduce_sum(tf.square(y - output_layer))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer1 =tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost1)
init = tf.global_variables_initializer()

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.8)
#run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
#run_metadata = tf.RunMetadata()
#saver = tf.train.Saver()

st = time.time()
with tf.Session(config = tf.ConfigProto(gpu_options = gpu_options)) as sess:
    # create initialized variables
    sess.run(init)
    
    ### for each epoch, do:
    ###   for each batch, do:
    ###     create pre-processed batch
    ###     run optimizer by feeding batch
    ###     find cost and reiterate to minimize
    
    one_hot=True
    for epoch in range(epochs):
        sample_y=np.array([])
        avg_cost = 0
        total_batch = int(train_x.shape[0]/batch_size/5)
        for i in range(total_batch):
            batch_x, batch_y = batch_creator(batch_size, train_x.shape[0], 'train',one_hot=one_hot)
            batch_y=batch_y.reshape(-1,output_num_units)
            _, c = sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
            #train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            #train_writer.add_summary(summary, i)
            #print('Adding run metadata for', i)
            #tl = timeline.Timeline(run_metadata.step_stats)
            #print(tl.generate_chrome_trace_format(show_memory=True))
            avg_cost += c /(i+1)
            sample_y=np.append(sample_y,batch_y)
            
        print ("Training Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost))
        mean = np.mean(sample_y)
        var = tf.reduce_sum(tf.square(sample_y-mean))
        print ("Training dataset total variance: ",var)
        #print ("\nTraining complete!")
        time_elapse = time.time()-st
        print  ("training time is :", time_elapse)
        # find predictions on val set
        if one_hot:
            pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
            print ("Classification Validation Accuracy:", accuracy.eval({x: val_x.reshape(-1, input_num_units), y: dense_to_one_hot(val_y)}))
        else:
            mean, var = tf.nn.moments(y,axes=[0])
            mean_hat,var = tf.nn.moments(output_layer,axes=[0])
            error = tf.reduce_sum(tf.abs(output_layer-y))
            var = tf.reduce_sum(tf.abs(y-mean))
            print ("Regression Validation total error :", error.eval({x: val_x.reshape(-1, input_num_units), y: val_y.reshape(-1,output_num_units)}))
            print ("Regression Validation total variance :", var.eval({x: val_x.reshape(-1, input_num_units), y: val_y.reshape(-1,output_num_units)}))       
            print ("y mean :", mean.eval({x: val_x.reshape(-1, input_num_units), y: val_y.reshape(-1,output_num_units)}))
            print ("Prediction mean :", mean_hat.eval({x: val_x.reshape(-1, input_num_units), y: val_y.reshape(-1,output_num_units)}))
