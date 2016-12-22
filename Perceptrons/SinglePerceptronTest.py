from Perceptron import evaluate_perceptron, train_perceptron
import numpy as np
import matplotlib.pyplot as plt
import math

"""
    Author Jones Agwata
    Script to test the working perceptron functions
"""
def generate_dataset():
    """Function that generates three datasets
    of 3x1000, 3x500 and 3x500.
    Params:
        None

    Returns:
        training_data, validation_data, test_data

    """
    #Generate 1000 random data points of 500X2
    data1 = np.random.rand(500,2);
    data2 = np.random.rand(500,2);
    # Shift points from first dataset by two points up
    #and the second to a point down then multiply x and y values

    data1[:,0]-=2
    data1[:,1]*=2
    data1[:,1]-=2

    data2[:,0]-=4
    data2[:,1]-=2
    data2[:,1]*=2

    #define angle to be rotated
    theta = 75

    #Define rotation matrix
    R = np.array([[math.cos(theta), -(math.sin(theta))],
        [math.sin(theta), math.cos(theta)]])

    #Rotate data points
    data1_p = np.dot(data1,R)
    data2_p = np.dot(data2,R)


    #append bias node elements

    #define class values as 1 for data1 and 2 for data2
    C1 = np.array([0 for i in range(500)])
    C2 = np.array([1 for i in range(500)])
    bias = np.array([-1 for i in range(500)])
    #combine points with class names
    cfd_data1 = np.column_stack((data1_p,bias))
    cfd_data2 = np.column_stack((data2_p,bias))

    cfd_data1 = np.column_stack((cfd_data1,C1))
    cfd_data2 = np.column_stack((cfd_data2,C2))




    #Create third and fourth class
    data3 = np.random.rand(500,2)
    data4 = np.random.rand(500,2)

    mean_data3 = np.array([2,1])
    mean_data4 = np.array([6,5])

    cov_data3 = np.array([[0.5, 0],[0,5]])
    cov_data4 = np.array([[4,0],[0,0.5]])


    #Create empty list to hold new x and y values after applying gaussian function

    data3_xy = np.random.multivariate_normal(mean_data3,cov_data3,500)
    data4_xy =  np.random.multivariate_normal(mean_data4,cov_data4,500)


    #Optional command to plot data points
    # plt.plot(data1[:,0], data1_p[:,1],'go',data2_p[:,0], data2_p[:,1],'ko',data3_xy[:,0],data3_xy[:,1],'rx',data4_xy[:,0],data4_xy[:,1],'bx')
    # plt.axis('equal')
    # plt.show()

    #Create Class 3 and 4 class definitions
    C3 = np.array([2 for i in range(500)])
    C4 = np.array([3 for i in range(500)])

    #Add class definitions to data points for class 3 and 4
    cfd_data3 = np.column_stack((data3_xy,bias))
    cfd_data4 = np.column_stack((data4_xy,bias))

    cfd_data3 = np.column_stack((cfd_data3,C3))
    cfd_data4 = np.column_stack((cfd_data4,C4))

    full_data = np.vstack((cfd_data1,cfd_data2,cfd_data3,cfd_data4))
    np.random.shuffle(full_data)


    training_data = full_data[0:1000,:]

    validation_data = full_data[1000:1500,:]

    test_data = full_data[1500:2000,:]

    return training_data, validation_data, test_data


#Initialise the weights and necessary epoch sizes
w = np.random.random_sample(3)-0.5
epochs =[]
test_errors = []
errors=0
start = 0
eta = 0.35
epch = 20

#Generates the dataset
train,test,validate = generate_dataset()
#Filters out the required classes
train_01  = train[train[:,3] <2]
test_01 = test[test[:,3] <2]
validate_01 = validate[validate[:,3]<2]

val = len(test_01)/epch
#loops through list while evaluating test data
for x in xrange(val):
    w = train_perceptron(w,train_01[start:epch,:],eta)

    epochs.append(epch)
    test_inputs = test_01
    er = evaluate_perceptron(w,test_inputs)

    test_errors.append(er)
    start+=20
    epch +=20

percentage_error = ((sum(test_errors)*2)/(len(test_01)*val))*100
#Plot error values
plt.title("Plot of epochs against errors")
plt.plot(epochs,test_errors,'b')
plt.show()
