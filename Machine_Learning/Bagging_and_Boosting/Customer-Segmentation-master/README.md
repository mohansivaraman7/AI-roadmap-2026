# Customer Segmentation Using Bagging & Boosting

## Table of Contents
- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project aims to segment customers for an automobile company using various machine learning techniques, including Bagging and Boosting. The goal is to predict the right customer segment for new potential customers based on their behavior and characteristics.

## Data Description
The dataset contains information about customers, including their demographics, profession, spending score, and more. The data is used to classify customers into four segments (A, B, C, D).

### Columns

| Feature          | Description                                      |
|------------------|--------------------------------------------------|
| ID               | Unique ID                                        |
| Gender           | Gender of the customer                           |
| Ever_Married     | Marital status of the customer                   |
| Age              | Age of the customer                              |
| Graduated        | Is the customer a graduate?                      |
| Profession       | Profession of the customer                       |
| Work_Experience  | Work Experience in years                         |
| Spending_Score   | Spending score of the customer                   |
| Family_Size      | Number of family members for the customer        |
| Category         | Anonymised Category for the customer             |
| Segmentation     | (target) Customer Segment of the customer        |

Data Source: [Kaggle - Customer Segmentation](https://www.kaggle.com/datasets/kaushiksuresh147/customer-segmentation)

## Results
The project evaluates various machine learning models, including Decision Tree, Random Forest, Bagging Classifier, AdaBoost, Gradient Boosting, and XGBoost. The performance of these models is compared, and the best model is selected based on the evaluation metrics.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements.