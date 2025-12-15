# EasyVisa: US Visa Approval Prediction



This project builds a machine learning model for EasyVisa (a consultancy working with the US OFLC) to predict which employment-based visa applications are likely to be certified or denied, helping officers prioritize reviews and understand key drivers of

The final model is a tuned XGBoost classifier that achieves an F1 score of around 0.80 on the held-out test set, capturing the majority of approvals and denials without severe overfitting. 



## 1. Business problem



The US Department of Labor’s Office of Foreign Labor Certification (OFLC) processes hundreds of thousands of temporary and permanent visa applications each year, making manual review increasingly time‑consuming.   



EasyVisa wants to:



- Shortlist applications with a higher chance of visa approval using a supervised classification model.   

- Identify applicant and job attributes that most strongly influence whether a case is certified or denied.   

- Recommend applicant profiles and hiring patterns that improve the likelihood of certification.   



Success is measured using F1 score on a held‑out test set, with particular attention to recall on the “Certified” class so that strong candidates are not missed. 



## 2. Dataset



The dataset contains 25,480 visa applications with 12 fields describing the applicant, employer, and job.   

Key fields include: 



- Applicant and employer: `continent`, `educationofemployee`, `hasjobexperience`, `requiresjobtraining`, `noofemployees`, `yrofestab`.  

- Job and wage: `regionofemployment`, `prevailingwage`, `unitofwage`, `fulltimeposition`.  

- Target: `casestatus` (Certified / Denied).  



There are no duplicate rows and no missing values in the raw data.   

The raw CSV is provided as part of the Great Learning “Machine Learning 2” course; this repository only includes the CSV and derived artifacts (code, models, and analysis), respecting the course’s intellectual property terms. 



## 3. Approach



The project follows an end‑to‑end classification workflow. 



1. **Problem framing**  

&nbsp;  - Define the business objective (prioritize likely approvals) and select F1 as the main metric, with recall on “Certified” as a secondary focus.   



2. **Feature engineering & data cleaning**  

&nbsp;  - Convert negative `noofemployees` values to their absolute values, since employee counts cannot be negative; summary statistics (mean, std) remain stable.   

&nbsp;  - Encode binary fields (`hasjobexperience`, `requiresjobtraining`, `fulltimeposition`, `casestatus`) as 0/1 categorical variables.   

&nbsp;  - Drop the unique identifier `caseid` and create a 60/20/20 split into train, validation, and test sets with similar class proportions.   



3. **Exploratory Data Analysis (EDA)**  

&nbsp;  - Study distributions of `noofemployees`, `yrofestab`, and `prevailingwage` (heavily skewed with many valid outliers).   

&nbsp;  - Examine categorical patterns across continent, education, region, unit of wage, and full‑time status.   

&nbsp;  - Use boxplots and bar charts to understand how education, full‑time employment, job training, and continent relate to visa approval rates.   



4. **Baseline and ensemble models**  

&nbsp;  - Train and compare five tree‑based models on the original split:  

&nbsp;    - Decision Tree Classifier.   

&nbsp;    - Random Forest Classifier.   

&nbsp;    - Bagging Classifier.   

&nbsp;    - AdaBoost Classifier.   

&nbsp;    - XGBoost Classifier.   

&nbsp;  - Evaluate each model on train, validation, and test sets using accuracy, precision, recall, and F1.   



5. **Class imbalance handling**  

&nbsp;  - Since about 67% of cases are certified and 33% denied, experiment with:   

&nbsp;    - SMOTE oversampling on the training set.   

&nbsp;    - Random undersampling on the training set.   

&nbsp;  - Refit all models on oversampled and undersampled data and compare F1 scores; oversampling modestly improves recall but does not dramatically change F1, while undersampling often degrades performance due to information loss.   



6. **Hyperparameter tuning and model selection**  

&nbsp;  - Use GridSearchCV and RandomizedSearchCV to tune Bagging, AdaBoost, and XGBoost.   

&nbsp;  - Select the tuned XGBoost model (with categorical support enabled and F1‑based scoring) as the final model based on its superior F1 and recall on validation and test sets. 



## 4. Key insights



From the EDA and model feature importances: 



- **Education level is critical**  

&nbsp; - Applicants with at least a high‑school or master’s degree have markedly higher approval rates; high school and master’s categories contribute most to model importance.   

&nbsp; - Recommendation: Candidates should aim for higher formal education (high school completion or above) before applying.



- **Full‑time jobs and no training requirement help**  

&nbsp; - Full‑time positions show higher approval rates compared to part‑time roles.   

&nbsp; - Applicants who do not require job training have a stronger likelihood of certification, consistent with the idea that experienced, job‑ready candidates are easier to approve.   



- **Geography and wages matter, but less than education**  

&nbsp; - Applications from Asia and Europe make up most of the volume and show relatively higher approval counts compared to some other continents, though wage distributions are broadly similar across regions.   

&nbsp; - Yearly wage contracts are more common and tend to be associated with higher approval likelihood, but wage alone does not guarantee certification.   



These insights translate into actionable guidance for EasyVisa on how to counsel clients (education, job type, training, geography).



## 5. Final model



### Tuned XGBoost Classifier (final model)



The final model is an XGBoost classifier trained on the oversampled training data and tuned via GridSearchCV using F1 as the scoring metric. 



- Test performance (approximate):  

&nbsp; - Accuracy ≈ 0.72  

&nbsp; - Recall ≈ 0.85  

&nbsp; - Precision ≈ 0.76  

&nbsp; - F1 ≈ 0.80   

- The model maintains reasonably similar performance on train, validation, and test sets, indicating no severe overfitting after tuning.   

- Confusion matrices show that the model identifies roughly 57–58% of true positives and around 15% of true negatives, which is acceptable given the class imbalance and business focus on not missing strong candidates. 



### Model comparison snapshot



| Model                  | F1 (test, normal) | F1 (test, oversampled) | Notes                                                   |
|------------------------|-------------------|------------------------|---------------------------------------------------------|
| Decision Tree          | ~0.74             | similar                | Simple baseline; some over/underfitting across splits.  |
| Random Forest          | >0.74             | similar                | Better than single tree; moderate imbalance issues.     |
| Bagging                | ~0.75–0.78        | similar                | Slight improvement; oversampling yields small gains.    |
| AdaBoost               | ~0.81             | ~0.81                  | Stable across train/val/test; strong baseline.          |
| XGBoost (tuned, final) | ~0.80             | ~0.80                  | Best trade‑off of F1 and recall; chosen as final.       |



The tuned XGBoost model is selected as the final model because it offers one of the highest F1 scores, strong recall on the certified class, and clear feature importances that align with domain intuition (education, region, and job characteristics). 



## 6. Repository structure



```text

easyvisa-visa-approval/

├─ data/

│ └─ EasyVisa.csv

├─ notebooks/

│ └─ EasyVisa_Visa_Approval.ipynb

├─ report/

│ └─ EasyVisa_Visa_Approval_Report.pdf

├─ README.md

└─ requirements.txt

```



## 7. How to run



### Option 1: Run locally (Jupyter)



1. Install dependencies:


```bash

pip install -r requirements.txt

```


2. Place the `EasyVisa.csv` dataset into the `data/` folder. 


3. Open and run the notebook:
```bash

jupyter notebook notebooks/EasyVisa_Visa_Approval.ipynb
```

### Option 2: Run in Google Colab



1. Upload the notebook file (`EasyVisa_Visa_Approval.ipynb`) to Google Colab.   



2. Upload `EasyVisa.csv` to Colab (or mount Google Drive) and adjust the data path in the first data‑loading cell if needed.  

&nbsp;

3. Run all cells from top to bottom to reproduce the analysis and model training. 



