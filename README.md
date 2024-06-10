# Surat Housing Price Analysis

This project analyzes housing prices in Surat using a dataset of uncleaned housing data. The goal is to clean the data, perform exploratory data analysis (EDA), create new features, build a predictive model for housing prices, and evaluate its performance.

## Table of Contents

- [Introduction](#introduction)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The purpose of this project is to analyze and predict housing prices in Surat. By cleaning the dataset and performing various data analysis techniques, we aim to gain insights into the housing market and build a model to predict prices based on various features.

## Data

The dataset used in this project is `surat_uncleaned.csv`, which contains information about various properties in Surat, including:
- Property name
- Area type
- Square footage
- Transaction type
- Status
- Floor details
- Furnishing status
- Facing direction
- Description
- Price per square foot
- Total price

## Installation

To run this project, you need to have Python installed. We recommend using a virtual environment to manage dependencies. Follow these steps to set up the project:

1. Clone the repository:
    ```sh
    git clone https://github.com/FloZewi/Surat_Housing_Price_Analysis.git
    cd Surat_Housing_Price_Analysis
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    ```

3. Activate the virtual environment:
    - On Windows:
      ```sh
      venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```sh
      source venv/bin/activate
      ```

4. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

Run the following command to start the data analysis and model building:

```sh
python analysis.py
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request if you would like to contribute to this project.

## License
This project is licensed under the MIT License - see the LICENSE file for details.