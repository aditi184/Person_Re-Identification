![Python](https://img.shields.io/badge/python-3.7-blue?style=flat-square&logo=python)
# Person Re-Identification

## Results

On validation set

|     Model      | CMC@rank-1 | CMC@rank-5 | mAP  |                           Download                           |
| :------------: | :--------: | :--------: | :--: | :----------------------------------------------------------: |
|    Baseline    |    92.9    |    96.4    | 91.5 | [model](https://drive.google.com/file/d/1IxTAUOjS3_S4sF1mRJ72Mp5Xo-omQu6a/view?usp=sharing) |
| LA-TF++ (ours) |    92.9    |    1.0     | 93.2 | [model](https://drive.google.com/file/d/1C5fkNlcLTduBjZ0YYSKQeeZWhXemCT9p/view?usp=sharing) |

## Installation

```bash
pip install -r requirements.txt
```

## Running Models

### **Training** 

**Locally-Aware Transformer (Baseline)**

```bash
python train_baseline.py --train_data_dir ./data/train --model_name la-tf_baseline --model_dir ./model
```

**LA-TF++ (Our model)** 

```bash
python run-train.py --train_data_dir ./data/train --model_name la-tf++_final --model_dir ./model
```

### **Testing**

```bash
python run-test.py --model_path <path-to-saved-model> --test_data ./data/val
```

The script `run-test.py` takes in the query and gallery images (present in the `test_data`) and computes the following metrics:

1. CMC@rank-1
2. CMC@rank-5
3. mean Average Precision (mAP)

#### Visualization

```bash
python run-test.py --model_path <path-to-saved-model> --test_data ./data/val --visualize --save_preds <path-to-save-images>
```

## Dataset

* The dataset has 114 unique persons. The train and val set contain 62 and 12 persons, respectively.
* We have held out the test set of 40 persons.
* Each person has been captured using 2 cameras from 8 different angles. That is, each person would have 16 images. All images of a unique person is stored in a single directory (numbered from 001 to 114).
* The dataset has the following directory structure:

* The images of a person in the val set in split into query and gallery images. The query is the set of images which will be used to retrieve the images of the same person from the gallery. 
* Note that query and gallery are mutually exclusive sets.

## Acknowledgements

- Locally Aware Transformer (LA-TF) is adaped from [Person Re-Identification with a Locally Aware Transformer](https://github.com/SiddhantKapil/LA-Transformer).
- Triplet Loss and Label Smoothing are adapted from [Alignedreid++](https://github.com/michuanhaohao/AlignedReID).


## Authors

- [Shubham Mittal](https://www.linkedin.com/in/shubham-mittal-6a8644165/)
- [Aditi Khandelwal](https://www.linkedin.com/in/aditi-khandelwal-991b1b19b/)

Computer Vision course project ([course webpage](https://www.cse.iitd.ac.in/~chetan/teaching/col780-2020.html)) taken by [Prof. Chetan Arora](https://www.cse.iitd.ac.in/~chetan)
