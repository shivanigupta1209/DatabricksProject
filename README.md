# COVID-19 Patient Outcome Prediction using Azure Databricks

This project leverages the **NeurIPS 2020 Data Science for COVID-19 (DS4C) dataset** to analyze patient-level COVID-19 data and predict health outcomes. It demonstrates the integration of **data engineering, machine learning, and visualization** using modern cloud-based tools.

---

## 📌 Project Overview
- **Objective:** Analyze COVID-19 patient-level data and predict patient outcomes (e.g., recovery or mortality) using machine learning models.  
- **Tech Stack:**  
  - **Azure Databricks** for scalable data engineering (PySpark, Delta Lake, Medallion Architecture).  
  - **Machine Learning** for outcome prediction (XGBoost and other models explored).  
  - **Power BI** for interactive dashboards and visualization.  
  - **Azure Data Lake Gen2** for storage integration.  

---

## 🗂️ Dataset
- **Source:** [NeurIPS 2020 Data Science for COVID-19 (DS4C) Challenge Dataset](https://www.kaggle.com/datasets/kimjihoo/coronavirusdataset)  
- **Description:** Contains detailed patient-level data including demographics, exposure, and outcomes.  
- **Key Features Used:**  
  - Age, Gender  
  - Region  
  - Infection Case / Source of exposure  
  - Dates (confirmation, release, death)  
  - Outcome (recovered, deceased)  

---

## ⚙️ Architecture
The pipeline follows the **Medallion Architecture**:  

1. **Bronze Layer (Raw Data)**  
   - Ingest raw CSVs into Azure Databricks.  
   - Store in **Delta format** for reliability.  

2. **Silver Layer (Cleaned Data)**  
   - Data cleaning, handling nulls, renaming columns.  
   - Applied **Slowly Changing Dimensions Type 2 (SCD2)** for patient history tracking.  

3. **Gold Layer (Curated Data)**  
   - Aggregated and feature-engineered tables.  
   - Prepared for machine learning and reporting.  

4. **Machine Learning Layer**  
   - Applied ML models (XGBoost, Logistic Regression).  
   - Evaluated performance using metrics like AUC-PR, LogLoss, and Accuracy.  

5. **Visualization Layer**  
   - Power BI dashboards for trend analysis and outcome prediction visualization.  

---

## 📊 Power BI Dashboards
- Interactive COVID-19 trends (regional spread, infection cases).  
- Patient demographics analysis.  
- Predictive insights on patient outcomes.  

---

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/shivanigupta1209/DatabricksProject.git

## 📽️ Presentation

You can view the project presentation here:  
[![View Presentation](https://img.shields.io/badge/Canva-Presentation-blue?logo=canva)](https://www.canva.com/design/DAGzBsWIOKw/HlXKts4J6b1JrWmAHUQ7iw/edit?utm_content=DAGzBsWIOKw&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)


