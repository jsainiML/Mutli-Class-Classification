# Mutli-Class-Classification
I wrote this code on Pytorch that shows how a NN model trained to perfrom classification with methods like Backpropogation and gradient descent.
Project from 'PyTorch for Deep Learning' by Daniel Bourke

## Toy Dataset 

     plt.figure(figsize=(10,5))
     plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)

![image](https://github.com/jsainiML/Mutli-Class-Classification/assets/135480841/2b14ee39-f76b-48d2-9144-1c0633d26db1)

## Model's learning trend
The fastest accuraccy with steep loss was seen with 'Adam' optimizer. SGD was able to produce similar results but on much higher epoch run. 

![image](https://github.com/jsainiML/Mutli-Class-Classification/assets/135480841/249d2d99-e309-44a0-86c6-6b50b8857236)


## Model Prediction

    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model0, Xb_train, yb_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model0, Xb_test, yb_test)

![image](https://github.com/jsainiML/Mutli-Class-Classification/assets/135480841/6d3a9bd0-4790-4405-b1b4-7439955d1ac3)




