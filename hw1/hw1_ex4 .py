import pickle, gzip, urllib.request, json
import numpy as np
import time

data_url = "http://deeplearning.net/data/mnist/mnist.pkl.gz"

# load the dataset
urllib.request.urlretrieve(data_url,"mnist.pkl.gz")
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load( f , encoding='latin' )

# extract all data out of touples
train_data, train_label = train_set
valid_data, valid_label = valid_set
test_data, test_label = test_set

# change label to fit for binary classification
train_label = -2 * np.mod(train_label, 2) + np.ones(train_label.shape)
valid_label = -2 * np.mod(valid_label, 2) + np.ones(valid_label.shape)
test_label = -2 * np.mod(test_label, 2) + np.ones(test_label.shape)
t = time.time()
## section 1 - centralize data
train_mean = np.mean(train_data, axis=0)

centered_train = train_data - train_mean
centered_valid = valid_data - train_mean
centered_test = test_data - train_mean

# section 2 - build an analytic and a gradient descent linear regressor
(num_of_samples, dimension) = np.shape(train_data)
given_lambda = np.logspace(-5, 2, 8)
w_analytic = []
b_analytic = []
b_sgd = []
w_sgd = []
learning_rate = 0.001
mean_y = (1/num_of_samples) * np.sum(train_label)

for cur_lambda in given_lambda:
    tmp_b_analytic =  mean_y
    b_analytic.append(tmp_b_analytic)
    tmp_a = ((1/num_of_samples) * np.dot(np.transpose(train_data),train_data)) + (2 * cur_lambda * np.eye(dimension))
    tmp_b = (1/num_of_samples) * np.dot(np.transpose(train_data),(train_label))
    w_analytic.append(np.dot(np.linalg.inv(tmp_a),(tmp_b)))

    b_t = 0*np.random.randn(1)/100
    w_t = 0*np.random.randn(dimension)/100
    for idx in range(0, 100):
        b_tplus_one = b_t - learning_rate*(b_t-mean_y)
        tmp_a_sgd = np.dot((1/num_of_samples) * np.dot(np.transpose(train_data),train_data) + 2*cur_lambda * np.eye(dimension),w_t)
        tmp_b_sgd = (1/num_of_samples) * (np.dot(np.transpose(train_data),-train_label))
        w_tplus_one = w_t - learning_rate*(tmp_a_sgd + tmp_b_sgd)

        b_t = b_tplus_one
        w_t = w_tplus_one

    b_sgd.append(b_t)
    w_sgd.append(w_t)

## section 3 - calculate Loss functions for each lambda on validation set and on Training set
# start by calculating Losses for Training set
Loss_01_an_t = []
squared_Loss_an_t = []
Loss_01_gd_t = []
squared_Loss_gd_t = []
(num_of_samples_train,dimension) = np.shape(train_data)
for idx in range (0,8):
    analytic_reg = np.dot(train_data,w_analytic[idx]) + (b_analytic[idx]*np.ones(num_of_samples_train,))
    gd_reg = np.dot(train_data,w_sgd[idx]) + (b_sgd[idx]*np.ones(num_of_samples_train,))

    h_an = np.sign(analytic_reg) # for L_01
    h_gd = np.sign(gd_reg)

    Loss_01_an_t.append((1/num_of_samples_train) * np.sum(((h_an*train_label)-1)/(-2)))
    Loss_01_gd_t.append((1/num_of_samples_train) * np.sum(((h_gd*train_label)-1)/(-2)))

    squared_Loss_an_t.append((1/num_of_samples_train) * (np.linalg.norm(analytic_reg - train_label)**2))
    squared_Loss_gd_t.append((1/num_of_samples_train) * (np.linalg.norm(gd_reg - train_label)**2))

# now we will calculate Losses for Validation set
Loss_01_an = []
squared_Loss_an = []
Loss_01_gd = []
squared_Loss_gd = []
(num_of_samples_val,dimension) = np.shape(valid_data)
for idx in range (0,8):
    analytic_reg = np.dot(valid_data,w_analytic[idx]) + (b_analytic[idx]*np.ones(num_of_samples_val,))
    gd_reg = np.dot(valid_data,w_sgd[idx]) + (b_sgd[idx]*np.ones(num_of_samples_val,))

    h_an = np.sign(analytic_reg) # for L_01
    h_gd = np.sign(gd_reg)

    Loss_01_an.append((1/num_of_samples_val) * np.sum(((h_an*valid_label)-1)/(-2)))
    Loss_01_gd.append((1/num_of_samples_val) * np.sum(((h_gd*valid_label)-1)/(-2)))

    squared_Loss_an.append((1/num_of_samples_val) * (np.linalg.norm(analytic_reg - valid_label)**2))
    squared_Loss_gd.append((1/num_of_samples_val) * (np.linalg.norm(gd_reg - valid_label)**2))


## section 4 - calculate Loss functions for test data - of best perfomed lambda's
(num_of_samples_test, dimension) = np.shape(test_data)

# choose best lambda models and calculate Loss function on test sets
opt01_lambda_an_idx = np.where(Loss_01_an == min(Loss_01_an))[0][0]
opt01_lambda_an = given_lambda[opt01_lambda_an_idx]

opt01_lambda_gd_idx = np.where(Loss_01_gd == min(Loss_01_gd))[0][0]
opt01_lambda_gd = given_lambda[opt01_lambda_gd_idx]

# claculate loss 01  and Squared Loss on test for Analytic W & B
opt_analytic_reg = np.dot(test_data,w_analytic[opt01_lambda_an_idx]) + (b_analytic[opt01_lambda_an_idx]*np.ones(num_of_samples_test,))
opt_h_an = np.sign(opt_analytic_reg)

opt_Loss_01_an = ((1/num_of_samples_test) * np.sum(((opt_h_an*test_label)-1)/(-2)))
opt_squared_Loss_an = (1/num_of_samples_test) * (np.linalg.norm(opt_analytic_reg - test_label)**2)


# claculate loss 01  and Squared Loss on test for Gradient Descent W & B
opt_gd_reg = np.dot(test_data,w_sgd[opt01_lambda_gd_idx]) + (b_sgd[opt01_lambda_gd_idx]*np.ones(num_of_samples_test,))
opt_h_gd = np.sign(opt_gd_reg)

opt_Loss_01_gd = (1/num_of_samples_test) * np.sum(((opt_h_gd*test_label)-1)/(-2))
opt_squared_Loss_gd = (1/num_of_samples_test) * (np.linalg.norm(opt_gd_reg - test_label)**2)


## section 5 - plot curves of of ideal lambda validated with Gradient Descent model

# we will do once again Gradient Descent with chosen lambda on training set, and at ech iteration we will calculate the loss
# on test and validation set's
cur_lambda = opt01_lambda_gd
opt_b_t = 0*np.random.randn(1)/100
opt_w_t = 0*np.random.randn(dimension)/100
step_01_Loss_val = []
step_Squared_Loss_val = []
step_01_Loss_test = []
step_Squared_Loss_test = []
for idx in range(0,100):
    # calc temporal W and B
    b_tplus_one = opt_b_t - learning_rate*(opt_b_t-mean_y)
    tmp_a_sgd = np.dot((1/num_of_samples) * np.dot(np.transpose(train_data),train_data) + 2*cur_lambda * np.eye(dimension),opt_w_t)
    tmp_b_sgd = (1/num_of_samples) * (np.dot(np.transpose(train_data),-train_label))
    w_tplus_one = opt_w_t - learning_rate*(tmp_a_sgd + tmp_b_sgd)

    # calc temporal Losses (01 Loss and Squared Loss) - on validation set
    step_reg_val = np.dot(valid_data, w_tplus_one) + \
                             (b_tplus_one * np.ones(num_of_samples_val, ))
    step_h_val = np.sign(step_reg_val)

    # calc temporal Losses (01 Loss and Squared Loss) - on test set
    step_reg_test = np.dot(test_data, w_tplus_one) + \
                   (b_tplus_one * np.ones(num_of_samples_test, ))
    step_h_test = np.sign(step_reg_test)

    step_01_Loss_val.append((1 / num_of_samples_val) * np.sum(((step_h_val * valid_label) - 1) / (-2)))
    step_Squared_Loss_val.append((1 / num_of_samples_val) * (np.linalg.norm(step_reg_val - valid_label) ** 2))

    step_01_Loss_test.append((1 / num_of_samples_test) * np.sum(((step_h_test * test_label) - 1) / (-2)))
    step_Squared_Loss_test.append((1 / num_of_samples_test) * (np.linalg.norm(step_reg_test - test_label) ** 2))

    # update params for next step
    opt_b_t = b_tplus_one
    opt_w_t = w_tplus_one

# now lets plot these loss functions
import matplotlib.pyplot as plt

# evenly sampled time at 200ms intervals
iteration = np.arange(0, 100, 1)


plt.figure(1)
plt.plot(iteration, step_Squared_Loss_val,'bs', label='Squared Loss on validation set')
plt.plot(iteration, step_Squared_Loss_test, 'k', label='Squared Loss on test set')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(iteration, step_01_Loss_val, 'r--', label='01 Loss on validation set')
plt.plot(iteration, step_01_Loss_test, 'g^', label='01 Loss on test set')
plt.legend()
plt.show()
elapsed = time.time()-t
print(elapsed)
