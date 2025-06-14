# AlphaCare Insurance Solutions Risk Analytics
## Project description

This project analyzes historical car insurance claim data in South Africa to optimize marketing and pricing strategies. The analysis includes EDA, hypothesis testing, and predictive modeling.

## Tasks:
- EDA & Stats
- DVC Pipeline
- A/B Hypothesis Testing
- Machine Learning Modeling
### Structure
- /data: Raw & processed data
- /notebooks: Jupyter notebooks
- /scripts: Reusable Python code
- /outputs: Plots & results
### Requirements
- pandas
- seaborn
- matplotlib
- scikit-learn
- statsmodels
- dvc 

# 📊 Task 1: Git Setup & Exploratory Data Analysis (EDA)

---

## 📌 Objective

This task focuses on establishing a solid foundation for the project by:

- Setting up **version control** using Git and GitHub with CI/CD via GitHub Actions.
- Performing a comprehensive **Exploratory Data Analysis (EDA)** to understand the insurance dataset.
- Applying statistical methods and visualizations to uncover insights on risk and profitability.

---

## 🛠️ What Was Done

### 1. Git and GitHub Setup

- Created a Git repository dedicated to the project.
- Established a `task-1` branch to isolate analysis work.
- Implemented continuous integration using **GitHub Actions** to automate code checks and workflows.
- Committed work regularly with descriptive messages to maintain clear version history.

### 2. Exploratory Data Analysis (EDA) & Statistics

- **Data Understanding:**  
  Reviewed dataset structure, verified data types, and assessed data quality including missing values.

- **Descriptive Statistics:**  
  Calculated variability and central tendencies for key numeric variables such as `TotalPremium` and `TotalClaims`.

- **Univariate Analysis:**  
  Visualized distributions using histograms (numerical data) and bar charts (categorical data).

- **Bivariate & Multivariate Analysis:**  
  Explored relationships between variables, including monthly changes in premiums and claims across ZipCodes, using scatter plots and correlation matrices.

- **Outlier Detection:**  
  Identified outliers with boxplots to understand their potential impact on the analysis.

- **Creative Visualizations:**  
  Produced three insightful and visually engaging plots highlighting key trends and patterns related to loss ratios, claim distributions, and geographic trends.

---

## 🔍 Key Insights

- The overall **Loss Ratio** (`TotalClaims` / `TotalPremium`) varies significantly across Provinces, Vehicle Types, and Gender.
- Identified outliers in claim amounts which may influence risk modeling.
- Detected temporal trends in claim frequency and severity over the 18-month period.
- Certain vehicle makes/models show notably higher or lower claim amounts, indicating varying risk profiles.

---

## 📁 Files and Branches

| File/Branch             | Description                          |
|-------------------------|------------------------------------|
| `task-1` (Git branch)    | Branch dedicated to EDA analysis    |
| `eda_analysis.ipynb`      | Jupyter notebook containing EDA code, statistics, and visualizations |
| `.github/workflows/ci.yml` | GitHub Actions CI pipeline configuration |

---

## ✅ Outcome

- The repository is well-structured for collaborative development and version control.
- Comprehensive EDA provides a deep understanding of the dataset’s characteristics and risk factors.
- Solid groundwork laid for subsequent modeling and analysis phases.

---

## 📚 References & Learning

- Applied statistical concepts and data visualization best practices.
- Used Python libraries such as `pandas`, `matplotlib`, and `seaborn` for analysis.
- Integrated automated testing via GitHub Actions to maintain code quality.

---

**Status:** ✅ Task 1 complete and ready for further analysis



# ✅ Task 2: Reproducible Data Pipeline with DVC

## 📌 Objective

Establish a reproducible and auditable data pipeline using [Data Version Control (DVC)](https://dvc.org/), a standard practice in regulated industries like finance and insurance. The goal is to ensure that datasets used in analysis and modeling are versioned and can be reproduced exactly for auditing, compliance, or collaboration.

---

## 🛠️ What Was Done

- **Initialized DVC** in the project to enable data tracking and decouple datasets from Git.
- **Configured a local DVC remote storage** directory to store versioned datasets outside the Git repository.
- **Tracked a dataset** (`machineLearningRating_v3.txt`) using DVC, generating a `.dvc` metadata file.
- **Ignored raw data files** using `.gitignore`, while versioning only the `.dvc` tracking files.
- **Pushed data** to the configured remote using `dvc push`.
- **Committed all configuration and tracking metadata** to Git.
- **Created a dedicated Git branch** (`task-2`) and merged it via a Pull Request to `main`.

---

## 📁 Key Files Committed

| File                                      | Purpose                                      |
|-------------------------------------------|----------------------------------------------|
| `data/machineLearningRating_v3.txt.dvc`   | Tracks the version of the dataset            |
| `.dvc/config`                             | Stores DVC remote configuration              |
| `.dvc/.gitignore`                         | Prevents internal DVC cache files from Git   |
| `.dvcignore`                              | Ignore rules for DVC                         |
| `.gitignore`                              | Prevents raw datasets from being tracked by Git |

---

## ✅ Outcome

- The dataset is now version-controlled and safely stored outside Git history.
- The Git repository remains clean and lightweight.
- Team members or auditors can reproduce exact experiments by using:
  ```bash
  git clone <repo-url>
  dvc pull
