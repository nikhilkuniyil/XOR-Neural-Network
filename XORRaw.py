#!/usr/bin/env python
# coding: utf-8

# # Homework 5

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


point1 = np.array([1,1])
point2 = np.array([0,0])
point3 = np.array([1,0])
point4 = np.array([0,1])
lst = [point1, point2, point3, point4]
points = np.array([(x, y) for x,y in lst])
labels = [1, 1, 0, 0]


# In[3]:


x_coords = []
y_coords = []

for point in points:
    x_coords.append(point[0])
    y_coords.append(point[1])
    
plt.scatter(x_coords[:2], y_coords[:2], color='red', label = 'Class 1')
plt.scatter(x_coords[2:4], y_coords[2:4], color='blue', label = 'Class 2')
plt.title('XOR Classes')
plt.legend()
plt.show()


# In[4]:


def sigmoid(val):
    return (1 / (1 + np.exp(-val)))


# In[5]:


def initialize_parameters(num_inputs, num_hidden_units, num_outputs):
    w1 = np.random.randn(num_inputs, num_hidden_units)
    w2 = np.random.randn(num_hidden_units, num_outputs)
    b1 = np.random.randn(2,1)
    b2 = np.random.randn()
    
    return w1, w2, b1, b2


# In[6]:


def forward_propagation(x, w1, w2, b1, b2, activation='sigmoid'):
    if activation == 'sigmoid':
        h1 = np.dot(w1, x).reshape((2,1)) + b1
        a = sigmoid(h1)
        z = np.dot(w2.T, a) + b2
        y_pred = sigmoid(z)
    
    return h1, a, z, y_pred        


# In[7]:


def compute_loss(prediction, label):
    return 0.5 * (prediction - label) ** 2


# In[8]:


def sigmoid_derivative(output):
    return output * (1 - output)


# In[9]:


def backpropagation(w2, x, h1, a, z, y_pred, label):
    
    dL_dyPred = -(label - y_pred)
    
    dL_dz = dL_dyPred * sigmoid_derivative(y_pred) 
    dyPred_dz = sigmoid_derivative(y_pred) 
    
    
    dL_dw2 = dL_dyPred * dyPred_dz * a
    dz_dw2 = a
    
   
    dL_db2 = dL_dyPred * dyPred_dz * 1
    dz_db2 = 1
    
    dL_da = dL_dyPred * dyPred_dz * w2
    dz_da = w2
    
    dL_dh1 = dL_dyPred * dyPred_dz * sigmoid_derivative(a)
    da_dh1 = sigmoid_derivative(a)
    
    dL_dw1 = dL_dyPred * dyPred_dz * dz_da * da_dh1 * x
    dh1_dw1 = x
    
    dL_db1 = dL_dyPred * dyPred_dz * dz_da * da_dh1 * 1
    dh1_db1 = 1
    
    return dL_dw2, dL_db2, dL_dw1, dL_db1


# In[10]:


def update_parameters(w1, w1_grad, w2, w2_grad, b1, b1_grad, b2, b2_grad, learning_rate):
    w1 -= learning_rate * w1_grad
    w2 -= learning_rate * w2_grad
    b1 -= learning_rate * b1_grad
    b2 -= learning_rate * b2_grad
    
    return w1, w2, b1, b2


# In[11]:


num_inputs = 2
num_hidden_units = 2
num_outputs = 1


# In[12]:


def correct(x, w1, w2, b1, b2, labels): 
    correct = 0
    for i in range(x.shape[0]):
        y_pred = forward_propagation(x[i], w1, w2, b1, b2)[3]
        y_pred = 1 if y_pred > 0.5 else 0
        if y_pred == labels[i]: 
            correct += 1
            
    return f'{correct/x.shape[0] * 100:.2f}%' 


# In[13]:


def train(points, labels, epochs=1500, learning_rate=1):
    avg_loss = []
    w1, w2, b1, b2 = initialize_parameters(num_inputs, num_hidden_units, num_outputs)
    
    for i in range(epochs):
        total_losses = 0
        
        for j in range(points.shape[0]):
            
            x = points[j]
            
            label = labels[j]
        
            h1, a, z, y_pred = forward_propagation(x, w1, w2, b1, b2, activation='sigmoid')
        
            loss = compute_loss(float(y_pred), label)
            total_losses += loss
    
            dL_dw2, dL_db2, dL_dw1, dL_db1 = backpropagation(w2, x, h1, a, z, y_pred, label)
            w1, w2, b1, b2 = update_parameters(w1, dL_dw1, w2, dL_dw2, b1, dL_db1, b2, dL_db2, learning_rate)
        
        if( i % 20 == 0 ): 
            print("----------------------EPOCH: ", i, end='')
            print("----------------------")
            print("Training Loss: ", total_losses/len(points))
            avg_loss.append(total_losses / len(points)) 
            print("Percent Correct: ", correct(points, w1, w2, b1, b2, labels))
            print()
        
    return w1, w2, b1, b2, avg_loss


# ## Run Model

# In[14]:


params = train(points, labels, epochs = 1000, learning_rate = 1)
print('Model Parameters: ', params[:4])


# In[29]:


avg_loss = params[4]
plt.figure()
plt.plot(avg_loss)
plt.title('Average Loss throughout training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()


# ### Hyperparameter Initialization 

# Through trial and error, by evaluating the model's loss graph based on the number of epochs and learning rate, I found that the model performed best when trained with 1000 epochs and with a learning rate of 1. 

# In[16]:


def plot_boundaries(params):
    x_vals = np.linspace(-2, 2, 100)
    y_vals = np.linspace(-2, 2, 100)
    xx, yy = np.meshgrid(x_vals, y_vals)

    classes = np.zeros((len(y_vals), len(x_vals)))

    for i in range(len(y_vals)):
        for j in range(len(x_vals)):
            input_vector = np.array([x_vals[j], y_vals[i]])

            output = forward_propagation(input_vector, params[0], params[1], params[2], params[3])[3]

            if output > 0.5:
                classes[i, j] = 1
            else:
                classes[i, j] = -1

    # Plot the contour map
    plt.contourf(xx, yy, classes, alpha=0.5, colors=('blue', 'red'))
    plt.scatter([0, 0, 1, 1], [1, 0, 1, 0], c=['blue', 'red', 'red', 'blue'], s=100)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()  

plot_boundaries(params[:4])


# ## Observations

# The model, trained on the 4 central points, (0,0), (1,0), (0,1), (1,1), was able to classify the points perfectly when we plotted the decision boundary. As shown above, our loss was able to decrease to a value of 0.0009, meaning that when ran on our 4 points, the model performed nearly perfectly. 
# 
# Moreover, we can visualize such decision boundaries using a mesh grid, where we set a region that we want our grid to cover. Then we run each point in our grid through our model and compute our predictions, using a color to denote each specfic predicted class. The point where the colors interchange is our 'boundary'.

# ## Introduce Noise

# In[17]:


def generate_noise(points, size, noise_std, labels):
    points = points.copy()
    labels = labels.copy()
    
    labels = np.array(labels * size)
        
    for point in points:
        new_points = np.array(list(point) * size)
        points_noise = np.random.normal(loc=0.0, scale=noise_std, size=new_points.shape)
        noise_final = points_noise + new_points
    
    pairs = list(zip(noise_final[::2], noise_final[1::2]))
    pairs = np.array([(x, y) for x,y in pairs])
    
    return pairs, labels


# In[18]:


num_inputs = 2
num_hidden_units = 2
num_outputs = 1

labels = [1, 1, 0, 0]

noise, final_labels = generate_noise(points, 100, 0.5, labels)

w1, w2, b1, b2 = initialize_parameters(num_inputs, num_hidden_units, num_outputs)


# ### Standard Deviation of 0.5

# In[19]:


params_noise = train(noise, final_labels, epochs = 5000, learning_rate = 1.5)


# In[30]:


avg_loss = params_noise[4]
plt.figure(1)
plt.plot(avg_loss)
plt.title('Average Loss throughout Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()


# In[21]:


plot_boundaries(params_noise[:4])


# ### Standard Deviation of 1

# In[22]:


num_inputs = 2
num_hidden_units = 2
num_outputs = 1

labels = [1, 1, 0, 0]

noise, final_labels = generate_noise(points, 100, 1, labels)

w1, w2, b1, b2 = initialize_parameters(num_inputs, num_hidden_units, num_outputs)


# In[23]:


params_noise = train(noise, final_labels, epochs = 5000, learning_rate = 1.5)


# In[24]:


plot_boundaries(params_noise[:4])


# ### Standard Deviation of 2

# In[25]:


num_inputs = 2
num_hidden_units = 2
num_outputs = 1

labels = [1, 1, 0, 0]

noise, final_labels = generate_noise(points, 100, 2, labels)

w1, w2, b1, b2 = initialize_parameters(num_inputs, num_hidden_units, num_outputs)


# In[26]:


params_noise = train(noise, final_labels, epochs = 5000, learning_rate = 1.5)


# In[27]:


plot_boundaries(params_noise[:4])


# ## Observations/Report

# I noticed that the model performed worse with any level of noise, than without the noise which means that the model needs more tuning in order to be able to generalize to unseen data. When trained on noisy data that was 0.5 standard deviations from our central points, (1,1), (1,0), (0,1), (0,0), the model achieved an accuracy of 54%, which is significantly lower than the accuracy of the model trained on just our 4 points: 100%. The model trained on data 1 standard deviation from our central points peformed the best on the noisy data and achieved an accuracy of 69%, and graphically better classifies the central points than the other two models. The final model was trained on noisy data 2 standard deviations away from the central points and achieved an accuracy of 63%.
