
# Traffic Sign Detection Project

This project aims to develop a Convolutional Neural Network (CNN) model to detect and classify traffic signs. The notebook includes steps for data preprocessing, model training, evaluation, and improvement.

## Project Overview

- **Objective**: To create a model that accurately classifies traffic signs.
- **Data**: The dataset contains various classes of traffic signs with distribution.
- **Model**: Convolutional Neural Network (CNN) implemented using TensorFlow and Keras.

## Project Structure

- **Data Preprocessing**: Handling missing values, data augmentation, and splitting the dataset.
- **Model Training**: Building and training the CNN model with hyperparameter tuning.
- **Evaluation**: Assessing the model performance using accuracy, confusion matrix, and classification report.
- **Improvements**: Strategies for improving the model's accuracy and handling imbalanced classes.

## Repository Contents

- `traffic_sign_detection.ipynb`: Jupyter notebook containing the full implementation of the project.
- `README.md`: Project overview and instructions.
- `data/`: Directory containing the dataset (not included, see below for instructions on obtaining the dataset).

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/henryfan000418/adsp_ml.git
    ```
2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Download the dataset and place it in the `data/` directory.
2. Run the Jupyter Notebook:
    ```sh
    jupyter notebook traffic_sign_detection.ipynb
    ```

## Project Details




### Exploratory Data Analysis

- Distribution of classes
- Sample images from each class
- Data augmentation techniques

### Feature Engineering & Transformations

- Image resizing and normalization
- Data augmentation: rotation, zoom, shift, and flip

### Model Development

- Building a CNN model using Keras
- Adding convolutional layers, pooling layers, dropout layers, and dense layers
- Using Adam optimizer and categorical cross-entropy loss function

### Evaluation

- Splitting the data into training and validation sets
- Training the model and monitoring performance on the validation set
- Evaluating the model using accuracy, confusion matrix, and classification report

### Results and Learnings

- Achieved an accuracy of 97% on the validation set
- Identified the most challenging classes for the model
- Discussed potential improvements and future work

### Future Work

- Explore other data augmentation techniques
- Experiment with different model architectures
- Address class imbalance using techniques such as SMOTE or class weighting

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License.

---

Feel free to customize this README file further to match the specific details and results from your notebook. If you have any additional information or sections you'd like to include, let me know!
