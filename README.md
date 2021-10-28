# COL780 Re-ID project
This repository provides the starter code and the dataset (train and val) for the project.

### Dataset
* The dataset has 114 unique persons. The train and val set contain 62 and 12 persons, respectively.
* We have held out the test set of 40 persons.
* Each person has been captured using 2 cameras from 8 different angles. That is, each person would have 16 images. All images of a unique person is stored in a single directory (numbered from 001 to 114).
* The dataset has the following directory structure:

        |__ train
        |        |__ 001
        |        |__ 002
        |            ...
        |__ val
        |        |__ query
        |        |        |__ 004
        |        |        |__ 012
        |        |            ...
        |        |__ gallery  
        |        |         
        |        |__ all_imgs

* The images of a person in the val set in split into query and gallery images. The query is the set of images which will be used to retrieve the images of the same person from the gallery. 
* Note that query and gallery are mutually exclusive sets.

### TODO
1. You need to write the code for a Person Re-ID model and train it. 
2. Write the training script and save the model after training.
3. Evaluate and analyze the results. 
	* The quantitative evaluation script has been given to calculate CMC@rank-1, CMC@rank-5, and mAP scores. 
	* You need to write code for any visualization.


