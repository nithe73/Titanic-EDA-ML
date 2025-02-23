# Titanic-EDA-ML

## Titanic Survival Prediction (EDA & Machine Learning)

### **Steps**

#### 1. Data Loading
- **Load the Data:**
  - Read the dataset.
- **Initial Exploration:**
  - Use `df.head()` to view the first few rows.
  - Use `df.info()` and `df.describe()` to understand data types, ranges, and missing values.
- **Identify Key Columns:**
  - `PassengerId`, `Survived` (target), `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, and `Embarked`.

#### 2. Exploratory Data Analysis (EDA)
- **Plot the survival distribution** (using matplotlib).
- **Feature Distributions:**
  - Visualize distributions for numerical features like `Age` and `Fare`.
  - Examine the distributions of categorical features such as `Pclass`, `Sex`, and `Embarked` using count plots.
- **Relationship Analysis:**
  - Compare survival rates across different groups (e.g., survival by `Pclass`, `Sex`, or `FamilySize` once created).
  - Use visualizations like bar charts, scatter plots, and heatmaps to uncover relationships.
- **Correlation Analysis:**
  - Plot a correlation matrix to see how features relate to one another and to the target variable.

#### 3. Data Cleaning & Feature Engineering
- **Handle Missing Values:**
  - Impute missing `Age` values (e.g., with the median or mean).
  - Fill missing values in `Embarked` (e.g., with the mode).
  - Deal with missing `Cabin` values (Deleting the whole column).
- **Create New Features:**
  - `FamilySize`:  
    ```python
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    ```
- **Drop Irrelevant Columns:**
  - Remove columns that don't add predictive value (e.g., `PassengerId`, `Name`, `Ticket`).

#### 4. Data Preprocessing
- **Encode Categorical Variables:**
  - Use techniques like label encoding for features such as `Sex`, `Embarked`.
- **Feature Scaling:**
  - Standardize or normalize numerical features if required by your models.
- **Train-Test Split:**
  - Split your data into training and testing sets (common split is 70/30) using `train_test_split`.

#### 5. Model Building
- **Select Models:**
  - Choose a range of classifiers to evaluate (e.g., Logistic Regression, Decision Tree, Random Forest, Support Vector Machine).
- **Training:**
  - Fit each model on the training data.
- **Prediction:**
  - Make predictions on the test data for each model.

#### 6. Model Evaluation
- **Calculate Evaluation Metrics:**
  - Compute metrics such as Accuracy, Precision, Recall, F1 Score, and the Confusion Matrix.
- **Detailed Evaluation:**
  - Use `classification_report` to get a full summary of each model's performance.
- **Visualization:**
  - Optionally, create a grouped bar chart to compare the performance of different models.

---

### **Libraries Used**
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `sklearn`
- `xgboost`
