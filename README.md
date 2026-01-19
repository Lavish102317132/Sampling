Assignment – Sampling

Objective
The aim of this assignment is to study how different sampling techniques affect an imbalanced dataset and how these techniques change the accuracy of various Machine Learning models.


Dataset Used
The dataset used is a credit card transaction dataset (imbalanced) downloaded from:

https://github.com/AnjulaMehto/Sampling_Assignment/blob/main/Creditcard_data.csv

Target column:
Class
0 = Normal Transaction
1 = Fraud Transaction


Problem Statement
Fraud datasets are usually imbalanced, meaning normal transactions are much higher than fraud transactions.
If we train a model directly on such data, it can show high accuracy but still fail to detect fraud properly.

To handle this problem, the following steps were performed:
1. Balance the dataset
2. Create 5 different samples
3. Apply 5 sampling techniques
4. Train 5 ML models
5. Compare accuracy results



Methodology Followed

Step 1: Load Dataset
- The dataset is loaded using pandas.
- The dataset is divided into:
  X = all feature columns
  y = Class column (output)



Step 2: Balance Dataset
The dataset is originally imbalanced.
To balance it:
- RandomOverSampler is applied
- Minority class is increased until both classes have equal records

This balanced dataset is then used for the rest of the assignment.


Step 3: Create Five Samples
To make results more reliable and avoid dependency on a single split:
- 5 random samples are created from the balanced dataset
- Each sample contains around 20 percent of the balanced data
- Different random seeds are used

The samples are:
Sample1, Sample2, Sample3, Sample4, Sample5


Step 4: Apply Five Sampling Techniques
Each sample is processed with five different sampling methods.
Sampling is applied only to the training dataset so that the testing set remains unbiased.

Sampling techniques used:
1. Sampling1 – RandomOverSampler
   - Duplicate minority class records randomly

2. Sampling2 – RandomUnderSampler
   - Remove majority class records randomly

3. Sampling3 – SMOTE
   - Generate synthetic minority class records using nearest neighbors

4. Sampling4 – NearMiss
   - Undersampling method that keeps majority samples close to minority samples

5. Sampling5 – SMOTETomek
   - SMOTE oversampling plus Tomek link cleaning to remove noisy points


Step 5: Train Five Machine Learning Models
Each sampled dataset is trained and tested using five models:

M1: Logistic Regression
M2: Decision Tree Classifier
M3: Random Forest Classifier
M4: Naive Bayes (GaussianNB)
M5: Support Vector Machine (SVM)

Note:
Logistic Regression and SVM work better after scaling, so StandardScaler is applied for those two models.



Step 6: Accuracy Evaluation
For each combination of:
- 5 samples
- 5 sampling methods
- 5 ML models

Accuracy is calculated as:
Accuracy = Correct Predictions / Total Predictions

Final accuracy is calculated by averaging results from all 5 samples:
Final Accuracy = (accuracy over 5 samples) / 5


Results Generated
After execution, two output files are created:

1. accuracy_table.csv
   - Contains accuracy values for 5 models vs 5 sampling techniques

2. best_sampling_per_model.csv
   - Contains best sampling technique for each model based on maximum accuracy

