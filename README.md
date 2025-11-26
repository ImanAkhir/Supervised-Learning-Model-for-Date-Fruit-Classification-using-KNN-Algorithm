# Supervised-Learning-Model-for-Date-Fruit-Classification-using-KNN-Algorithm

1.0 INTRODUCTION
------------------------------------------------------------
This project applies supervised machine learning to classify 
date fruit varieties using the K-Nearest Neighbors (KNN) 
algorithm. The model learns from a dataset containing 
geometric, color, and texture-based features extracted 
from images of date fruits.

The system automates the classification process, reducing 
human error and supporting efficient sorting in agriculture.

Key features of this project:
- Data preprocessing (encoding, scaling)
- KNN classification (k = 5)
- Model accuracy evaluation
- Prediction function for new samples


2.0 DATASET
------------------------------------------------------------
The dataset includes 898 labelled samples consisting of:
- Geometric features (area, perimeter, eccentricity, etc.)
- Texture features (entropy, skewness, kurtosis)
- Color features (RGB channels)
- Wavelet transform features
- Class label (fruit type)

Dataset must be named: Date_Fruit_Dataset.csv


3.0 PROJECT FILES
------------------------------------------------------------
- date_fruit_knn.py     → Main machine learning script
- README.txt            → Project overview & instructions
- requirements.txt      → Python dependencies
- Date_Fruit_Dataset.csv → Dataset (not included here)


4.0 INSTALLATION & SETUP
------------------------------------------------------------
1. Ensure you have Python 3.8+ installed.
2. Install dependencies using:

   pip install -r requirements.txt

3. Place the dataset in the same directory as the script.


5.0 RUNNING THE SCRIPT
------------------------------------------------------------
Run the training and evaluation script:

   python date_fruit_knn.py

The program will:
- Load the dataset
- Preprocess data
- Train the KNN model
- Display accuracy & classification report


6.0 PREDICTING NEW SAMPLES
------------------------------------------------------------
In the script, you may enable the prediction example:

   sample = X.iloc[0].tolist()
   print("Predicted class:", predict_single(sample))

You may also pass your own feature values into the function.


7.0 REQUIREMENTS
------------------------------
pandas, numpy, scikit-learn

8.0 LICENSE
------------------------------
This project is for educational use.

8.0 AUTHOR & CREDITS
------------------------------------------------------------
Iman A., developed as part of the Machine Learning Group Assignment together with Fitrah M.R. and Zullaikha Z.
