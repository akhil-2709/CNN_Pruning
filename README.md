                                                   #### CNN Pruning ####


In this project, we intend to implement two different CNN pruning algorithms, one-shot pruning, and iterative magnitude-based pruning, on a deep neural network model (ResNet-18) optimized for image classification tasks using CIFAR-10 dataset. We will compare the pruning algorithms’ influence on model sparsity, test accuracy, and speedups. We will use weight pruning, an unstructured pruning algorithm that removes connections/weights from a neural network. We will use global pruning, which removes the lowest x% of connections across the whole model. 


The two pruning algorithms that we will use are One-shot pruning and Iterative pruning with sparsity ratios of 50%, 75% and 90%. We will experiment with different values of hyperparameters like number of epochs, number of iterations (for Iterative pruning) in combination with the aforementioned sparsity ratios.



**How to Run the Test Cases** - We have written a test script py file for running on local and a notebook(.pynb) to run on colab. Either of the methods can be used to run the test  script.





**Method - 1**(Running py file on the local)

* Clone the repository and go inside the root directory.
* Required libraries for running the code - numpy, torch, torchvision 
* Please install all the above libraries using pip3 eg. - “pip3 install torch”
* Please install any other dependencies required and not mentioned above.
* Run the test_script.py file with the command “sudo python3 testing_script.py”
* Each model must take approximately a minute to run and print the accuracy and inference time


**Method - 2** (Running the notebook on colab)

* Clone the repository to a folder named “cs532_project2” in your local
* Upload the “cs532_project2” folder to google drive 
* The folder “cs532_project2” must contain the “src” folder and other .pt model files
* Open the “src” folder and open the testing_script.ipynb file in google colab and run each cell
* Each model must take approximately a minute to run and print the accuracy and inference time

PyTorch files:
We have named the pytorch files by including the sparsity percentage, type of algorithm(oneshot or iterative) and the number of epochs in the end, in the following manner:

resnet18_prune_’’Sparsity’perc_’oneshot/iterative’ epochs_iterations(only for iterative)

Here we have considered 3 epochs and 5 epochs only
Thus we have a total of 12 pytorch files:
1. Resnet18_prune_50perc_iterative3_3
2. Resnet18_prune_50perc_iterative3_5
3. Resnet18_prune_50perc_oneshot3
4. Resnet18_prune_50perc_oneshot5
5. Resnet18_prune_75perc_iterative3_3
6. Resnet18_prune_75perc_iterative3_5
7. Resnet18_prune_75perc_oneshot3
8. Resnet18_prune_75perc_oneshot5
9. Resnet18_prune_90perc_iterative3_3
10. Resnet18_prune_90perc_iterative3_5
11. Resnet18_prune_90perc_oneshot3
12. Resnet18_prune_90perc_oneshot5

