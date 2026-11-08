# Machine Learning Assignment 2

Student ID: 2026aj05632  
Course: M.Tech (AIML), Machine Learning  
Dataset: UCI Diabetic Retinopathy Debrecen (Messidor features)

## a. Problem statement

The task is to decide whether a Messidor fundus image contains signs of diabetic retinopathy. Each image is already reduced to 19 numeric measurements (lesion counts, a pre-screen flag, and a few anatomic measurements). The label is binary: `dr_signs = 1` if the image was graded as showing signs, and `0` if it was not.

A majority-class guess is right on 53.25% of the held-out rows. Precision, recall, and F1 below are for the positive class (signs present). AUC uses the predicted probability of that class.

## b. Dataset description

Source: UCI Machine Learning Repository, Diabetic Retinopathy Debrecen Data Set (dataset id 329), built from features of the Messidor images. Antal and Hajdu published the feature set with their screening study (Knowledge-Based Systems, 2014). The table used here is stored at `data/messidor_retinopathy.csv`.

Size: **1,151 instances** and **19 features**. Both are above the assignment minimum (500 instances, 12 features). There are no missing values. The label counts are 540 for class 0 and 611 for class 1.

| Column | Meaning |
| --- | --- |
| `quality_ok` | Quality assessment. 1 = usable image, 0 = poor quality. |
| `prescreen_abnormal` | Pre-screening flag. 1 = severe retinal abnormality suspected. |
| `ma_count_alpha_0_5` … `ma_count_alpha_1_0` | Microaneurysm counts at confidence levels 0.5, 0.6, 0.7, 0.8, 0.9, and 1.0. |
| `exudate_norm_1` … `exudate_norm_8` | Eight normalized exudate measurements supplied in the UCI file (attributes 8–15). |
| `macula_optic_distance` | Distance between the macula center and the optic-disc center, divided by the ROI diameter. |
| `optic_disc_diameter` | Diameter of the optic disc. |
| `am_fm_label` | Binary result of the AM/FM-based classifier included with the UCI file. |
| `dr_signs` | Target. 1 = signs of diabetic retinopathy (Messidor grades 1, 2, or 3). 0 = no signs. |

The microaneurysm counts move together: correlation between `ma_count_alpha_0_5` and `ma_count_alpha_1_0` is 0.860. Every model uses the same stratified 80/20 split (`random_state = 42`): 920 training rows and 231 test rows in `test_data.csv`. The scaler is fit on the training rows only. Settings were chosen by stratified 5-fold cross-validation on training ROC AUC.

## c. Github Repository Link

https://github.com/darshikapundir/ml-assignment-2-messidor-dr

## d. Models used

All five models were fit on the same Messidor split. The test metrics below are from the saved pipelines scored on `test_data.csv`.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.7576 | 0.8579 | 0.8526 | 0.6585 | 0.7431 | 0.5363 |
| Decision Tree | 0.6320 | 0.6322 | 0.8065 | 0.4065 | 0.5405 | 0.3326 |
| KNN | 0.6190 | 0.6926 | 0.6667 | 0.5691 | 0.6140 | 0.2455 |
| Naive Bayes | 0.6494 | 0.7178 | 0.6544 | 0.7236 | 0.6873 | 0.2924 |
| Support Vector Machine | 0.7013 | 0.8066 | 0.7812 | 0.6098 | 0.6849 | 0.4205 |

### Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Strongest model on this split. Test AUC 0.8579, accuracy 0.7576, precision 0.8526, F1 0.7431, MCC 0.5363. CV AUC was 0.8163, so the hold-out score is in line with the training folds. `C = 10` and no class weight fits a nearly balanced label (540 vs 611). It is conservative: a positive call is usually right, and recall is 0.6585, so about one third of the true DR-sign images are missed. The largest weights are on the microaneurysm counts, with opposite signs. Those columns are redundant (correlation 0.86), so the large coefficients are shared lesion signal, not six separate effects. |
| Decision Tree | The search kept a shallow tree (depth 4, at least 8 rows per leaf). CV AUC is only 0.6857. On the test set, accuracy is 0.6320 and AUC is 0.6322. Precision looks high (0.8065) only because recall falls to 0.4065, and F1 is the worst of the five at 0.5405. A depth-4 tree has only a few distinct probabilities, which is why AUC sits on top of accuracy instead of ranking the rows. Importance is concentrated on `exudate_norm_7` (0.36) and `ma_count_alpha_0_5` (0.32). It beats the 0.5325 majority baseline and is still a weak ranker. |
| KNN | 15 neighbors with distance weights. CV AUC 0.7010. Test accuracy 0.6190, AUC 0.6926, F1 0.6140, MCC 0.2455. MCC is the lowest of the five, so the predicted labels barely agree with the truth after chance is removed. Scaling puts the columns on one range, but the microaneurysm counts still move together, and Euclidean neighbors mix the two classes. Storing the training rows did not produce a clean boundary on the 231 test images. |
| Naive Bayes | Gaussian NB with `var_smoothing = 1e-9`. CV AUC 0.6865. Test accuracy 0.6494, AUC 0.7178, precision 0.6544, recall 0.7236, F1 0.6873, MCC 0.2924. Recall is the best of the five. The independence assumption is false here, because the six microaneurysm counts are the same lesions read at different thresholds. The model therefore flags more positive images, and more of those flags are wrong: precision drops from 0.8526 (logistic regression) to 0.6544. The recall gain over logistic regression is real but small (0.7236 vs 0.6585). |
| Support Vector Machine | RBF kernel, `C = 8`, `gamma = scale`, balanced class weights. CV AUC 0.7994, second only to logistic regression. Test AUC 0.8066, accuracy 0.7013, precision 0.7812, recall 0.6098, F1 0.6849, MCC 0.4205. Balancing the classes did not raise recall above the unweighted logistic model (0.6098 vs 0.6585). The RBF kernel also did not beat the linear score. On these 19 measurements the extra flexibility is not paying for itself. |
| Overall, Winner for your dataset? | Logistic regression. It leads on accuracy, AUC, precision, F1, and MCC, and its CV AUC and test AUC agree. Naive Bayes wins recall only, at a large cost in precision. The default model in the Streamlit app is logistic regression. |
