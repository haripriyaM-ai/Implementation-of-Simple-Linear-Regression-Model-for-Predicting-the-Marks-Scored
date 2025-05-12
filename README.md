# EXPERIMENT NO: 1
# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. import the needed packages. 
2. Assigning hours to x and scores to y.
3. Plot the scatter plot.
4. Use mse,rmse,mae formula to find the values.
   
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: HARI PRIYA M
RegisterNumber: 212224240047
*/
```

    import pandas as pd
    import numpy as np
    from sklearn.metrics import mean_absolute_error,mean_squared_error
    import matplotlib.pyplot as plt
    df=pd.read_csv('/content/student_scores.csv')
    print(df)
    print(df.head())
    print(df.tail())
    x=df.iloc[:,:-1].values
    print(x)
    y=df.iloc[:,1].values
    print(y)
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
    from sklearn.linear_model import LinearRegression
    reg=LinearRegression()
    reg.fit(x_train,y_train)
    y_pred = reg.predict(x_test)
    print(y_pred)
    print(y_test)
    plt.scatter(x_train,y_train,color='purple')
    plt.plot(x_train,reg.predict(x_train),color='black')
    plt.title("Hours vs Scores(TRAINING SET)")
    plt.xlabel("Hours")
    plt.ylabel("Scores")
    plt.show()
    plt.scatter(x_test,y_test,color='red')
    plt.plot(x_train,reg.predict(x_train),color='black')
    plt.title("Hours vs Scores(TESTING SET)")
    plt.xlabel("Hours")
    plt.ylabel("Scores")
    plt.show()
    MSE=mean_absolute_error(y_test,y_pred)
    print('Mean Square Error = ',MSE)
    MAE=mean_absolute_error(y_test,y_pred)
    print('Mean Absolute Error = ',MAE)
    RMSE=np.sqrt(MSE)
    print("Root Mean Square Error = ",RMSE)
## Output:

Printing the Dataset

![Screenshot 2025-05-12 165404](https://github.com/user-attachments/assets/fa402243-c376-4c45-95ff-76f74fb67753)

Reading Head and Tail Files

![Screenshot 2025-05-12 165521](https://github.com/user-attachments/assets/be68ad7b-634c-40ce-a084-2ea0774a4947)

Comparing the Datasets

![Screenshot 2025-05-12 190708](https://github.com/user-attachments/assets/664626a8-5d50-4813-a6b4-4c6f2b7dd6aa)


Predicting Values

![image](https://github.com/user-attachments/assets/8d2d537a-f95a-4767-a9f6-b60672aca0d2)

GRAPH : TRAINING SET

![Screenshot 2025-05-12 190748](https://github.com/user-attachments/assets/809dd72b-5a77-4171-bc48-2cc751867da8)




GRAPH : TESTING SET

![Screenshot 2025-05-12 190812](https://github.com/user-attachments/assets/038af4a4-227f-4987-abf3-401ecd2fe3ef)

Finding Errors

![Screenshot 2025-05-12 190822](https://github.com/user-attachments/assets/ca335892-6a29-439d-888c-484e35f3171c)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
