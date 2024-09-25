# Naval Vessel Engine Life Prediction

This project focuses on predicting the decay state of naval vessel engines using machine learning models. The dataset contains various engine parameters, and the goal is to use these features to forecast engine decay for predictive maintenance.

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies](#technologies)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Introduction
Naval vessels rely on complex engines that require timely maintenance to ensure optimal performance. This project implements a neural network model to predict the decay state coefficients of key engine components based on various sensor readings. The project also includes data cleaning, heatmap visualizations, and a PyTorch-based model to make predictions.

## Features
- **Data Cleaning**: Preprocessing of raw datasets, including feature selection and missing value handling.
- **Heatmap Visualization**: Correlation heatmap to analyze the relationship between engine parameters.
- **Machine Learning Model**: Predictive model using PyTorch for forecasting engine decay.
- **Hyperparameter Tuning**: Easily configurable parameters for experimentation.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Bekathunder215/naval-engine-lifecycle-prediction.git
    ```
   
2. Navigate to the project directory:
    ```bash
    cd naval-engine-lifecycle-prediction
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the main script**:
   After ensuring the dataset is in place (`datasets/data.csv`), you can run the project using the following command:
   ```bash
   python main.py



.	Modify Hyperparameters:
You can modify the hyperparameters such as batch_size, learning_rate, and seed directly in the script before running the model.

## Technologies

	•	Programming Language: Python 3.7+
	•	Libraries:
	•	PyTorch
	•	Pandas
	•	Seaborn
	•	Matplotlib
	•	NumPy
	•	Scikit-learn
	•	TQDM

## Project Structure
naval-engine-lifecycle-prediction/
│
├── datasets/                # Directory containing the dataset files
│   └── data.csv             # Main dataset
│
├── main.py                  # Main script containing the ML model and data preprocessing
├── requirements.txt         # Required dependencies for the project
├── README.md                # Project documentation
└── LICENSE                  # License file (if applicable)


## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request. Please ensure your code adheres to the following guidelines:

	•	Follow PEP8 for Python code style.
	•	Add meaningful comments and docstrings where necessary.
	•	Make sure your contributions are well-tested.

## License

This project is licensed under the MIT License. See the LICENSE file for more details

### Explanation of Sections:
- **Introduction**: Provides a concise overview of the project and its goals.
- **Features**: Highlights the main functionality of the project.
- **Installation**: Instructions for cloning the repository and installing dependencies.
- **Usage**: Shows how to run the project and modify hyperparameters.
- **Technologies**: Lists the key technologies used in the project.
- **Project Structure**: Explains the organization of the project files.
- **Contributing**: Guidelines for contributions (if you're open to collaboration).
- **License**: Details about the project’s licensing (you can choose MIT or any other license as appropriate).
