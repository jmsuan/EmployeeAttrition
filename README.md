# EmployeeAttrition - Leveraging AI to Retain Employees and Reduce Attrition
An application of machine learning that helps reduce organizational resource strain. 

By Jonathan Lane
## Overview
The goal of this project is to use [an open data set provided by IBM](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data) to determine what employee features contribute to the likelihood of their attrition (leaving/quitting the company). This project is an introductory machine learning project of mine, and leverages the extensive classification model options provided by the scikit-learn library. Ultimately, the model achieved an impressive F1-score of approximately 0.85, which is relatively high considering the size of the dataset (roughly 122KB). The model also exceeds the following goals I set prior to training: 
 - >80% accuracy
 - >0.70 AUC_ROC value
## Features
- A **predictive method** of determining the likelihood of any employee's attrition.
- Two **descriptive figures** describing correlations found in the [available data set](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data).
- The ability to adjust features for each employee, which can help the user implement smart retention strategies (the user can identify which feature has the greatest impact on probability).
- Reproducible data set preprocessing (see `notebooks/02_data_preprocessing.ipynb`).
- Data exploration and plotting using [Pandas](https://pandas.pydata.org/) and [Matplotlib](https://matplotlib.org/).
- An interactive UI for adjusting a single input to the predictive model.
- Utilization of a [StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html) and other hyperparameter-optimized classification models from the [scikit-learn library](https://scikit-learn.org/stable/index.html).
- An impressive weighted F1-score of 0.85 considering the size of the data set.
- Self-hosted operation of the model, allowing files to be shared securely by an organization.
- An organized Jupyter Notebook directory structure.
- A clean Jupyter Notebook for the end user to use the trained ML model.
## Installation
### Prerequisites
- A reasonably performant computer running Microsoft Windows 11.
- [Git](https://git-scm.com/downloads) installed and added to your Windows environment PATH variable.
- The latest version of [Python 3.12](https://www.python.org/downloads/).
- The latest version of [Miniconda](https://docs.anaconda.com/miniconda/miniconda-other-installer-links/) that supports Python 3.12 (I used 24.4.0).
### User Guide
#### 1. Clone the Project Repository to Your Project Directory
**Using your Miniconda PowerShell Terminal,** clone the project from GitHub with this command:
```bash
git clone https://github.com/jmsuan/EmployeeAttrition.git
```
#### 2. Go to the Project Folder
Change to the project directory using:
```bash
cd EmployeeAttrition
```
#### 3. Set Up the `prod` Environment
This will create a conda environment installed with the same libraries I tested the application with.
```bash
conda env create --file .\envs\prod.yml --prefix .\envs\prod
```
## Using the App
#### 1. Activate the `prod` Environment
**Using your Miniconda PowerShell Terminal navigated to the project's root folder,** activate the environment we created during the installation process:
```bash
conda activate .\envs\prod
```
#### 2. Start Jupyter Notebook
Start self-hosting a Jupyter Notebook interface by entering in the terminal:
```bash
jupyter notebook
```
If done correctly, this should open your Jupyter Notebook instance in your computer's default internet browser.
#### 3. Open the `application` Notebook
- Double-click the `notebooks` directory to expand it.
- Double-click on `application.ipynb`.
#### 4. Run the Python Cell(s) in the Notebook
- In the Notebook header, you should see a `Run` menu. Click it.
- Click `Run All Cells`.
- Scroll up to see the application interface widgets.
#### 5. Use the Widgets to Experiment
- The sliders are set to median values calculated from our available data, and are limited to the range of the data's input.
- Experiment with different inputs using the interactive widgets to visually see how they affect an employee's likelihood of attrition.
#### 6. When Finished, Close the Application
You can close the application by opening the Anaconda Terminal window that's running the Jupyter Notebook and using the `Ctrl+C` hotkey. You may then close the environment (and Conda) by entering `exit` in the terminal, or by closing the window.
## Implementation Details
- The classification model that I trained for this project was a [StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html), which classifies using inputs from various other scikit-learn estimators. In this case, the estimators I used included [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html), [Gaussian Naive-Bayes](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html), [Histogram-based Gradient Boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html), and an [AdaBoost classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html). See `notebooks/03_model_training.ipynb` for further details.
- The program is designed with extensibility in mind, using Jupyter Notebooks as a method for documenting and prototyping the ready-to-polish application.
## Environment
- **Software**: Developed in Python 3.12.2 using Jupyter Notebooks and the following libraries:
	- numpy
	- pandas
	- scikit-learn
	- ydata-profiling
- **Hardware**: Tested on an ASUS ROG Zephyrus Duo 16 (2023) laptop with the follow specifications:
	- **Processor:** AMD Ryzen 9 7945HX with 2501 MHz, 16 Cores, 32 LPs
	- **Graphics Card:** NVIDIA GeForce RTX 4090 Laptop GPU
	- **Memory (RAM):** 32.0 GB
	- **Storage:** 1.9 TB SSD
## Acknowledgments
IBM for providing this [project's open data set](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data).