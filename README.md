# House Price Prediction Using Gradient Descent

This project implements a **Linear Regression** model to predict house prices based on features such as square footage, number of bedrooms, and house age. The model is built from scratch using **Python**, employs **gradient descent** for optimization and Z-normalization feature scaling, offering a hands-on demonstration of fundamental machine learning concepts.

## Features
- Manual implementation of linear regression with gradient descent.
- Animated loading indicator in the terminal for real-time feedback.
- Z-normalization of data for efficient training.
- Model evaluation with detailed metrics:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R-squared (R²)
- Visualizations for better understanding of model performance:
  - Actual vs Predicted Prices
  - Feature-wise Scatter Plots with Predicted Values

## Dataset
The dataset contains information about houses with the following columns:
- **Square_Footage**: The area of the house in square feet.
- **Num_Bedrooms**: Number of bedrooms in the house.
- **House_Age**: Age of the house in years.
- **House_Price**: The target variable representing the price of the house.

## Requirements
Install the necessary Python libraries:
```bash
pip install numpy pandas matplotlib scikit-learn
```

## Dataset License and Attribution

The dataset used in this project is [House Prices Dataset](https://huggingface.co/datasets/RashidIqbal/house_prices_data) and is licensed under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

**Author(s) of the Dataset**: Rashid Iqbal

**License Details**:
- You are free to:
  - **Share** — copy and redistribute the material in any medium or format.
  - **Adapt** — remix, transform, and build upon the material for any purpose, even commercially.
- Under the following terms:
  - **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.

The dataset is provided "as is," without warranties or guarantees.

### Citation
If you use this dataset in your project, please cite it as follows:
```plaintext
Iqbal, Rashid. "House Prices Dataset". Available under the Creative Commons Attribution 4.0 International License.


