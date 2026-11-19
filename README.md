# StyleSense Forecasting Project

StyleSense is a fast-growing online women's clothing retailer. 
StyleSense is known for its trendy and affordable fashion. Many of its new
customers leave a review but do not say whether they *recommend* the product,
which leaves a backlog of reviews with a missing label.

This project builds a Machine Learning (ML) pipeline that predicts the label
(`Recommended IND`: 1 = recommends, 0 = not recommends) from variables such as the 
review text, the review title, the customer's age, the positive feedback count and the product
category. One scikit-learn Pipeline handles everything from raw data to
prediction including processings such as imputation, scaling, one-hot encoding, text processing (spaCy),
TF-IDF features and a logistic regression classifier. The pipeline has been tuned
using GridSearchCV and evaluated once on a held-out test set (10% of the data).

`Interpretation of Evaluation Held out Test set` : The tuned pipeline reaches about
0.90 accuracy and 0.84 macro F1 on unseen reviews,relatively close to its cross-validated macro F1 of 0.85, suggesting reasonable generalization.
Accuracy alone is only ~8% above the reference (0.816)

## Getting Started
You could follow the below instructions to run this project.

### Dependencies
* Python 3 (developed and tested with Python 3.12)
* Python packages (listed in `requirements.txt`):

```
numpy
pandas
scikit-learn
spacy
matplotlib
joblib
notebook
```

* For text processing, We used the spaCy English model `en_core_web_sm` for this exercise.


### Installation

1. Open a Command Prompt or PowerShell in the project folder
2. (Optional) create and activate a virtual environment
3. Install the requirements
4. Start Jupyter from the `starter` folder and open `starter.ipynb`.

```
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
cd starter
jupyter notebook starter.ipynb
```

* **Run time:** the grid search performs 1,440 model fits and it could
 take a few minutes or more dependent on your laptop system configuration (CPU|GPU).

## Testing

We have not created a separate automated test suite (no `pytest` files), though there
are some built-in checks in the notebook. 

### Break Down Tests

* **Feature check:** an assert confirms that every column of the data
  belongs to exactly one group (numerical, categorical or text)
* **Baseline vs. tuned comparison:** both scored with 5-fold stratified cross-validation on the
  training data
* **Held-out test evaluation:** the final pipeline, retrained on all training
  data, is scored once on 10 % test set 
* **Inference on raw data with missing values:** null values are handled without errors
  and also an example is provided in the notebook
* **Save and reload check:** the trained pipeline is saved with `joblib`and an assert
  confirms the reproducibility


## Project Instructions

**Task:** use the provided data (`starter/data/reviews.csv`) to build a machine
learning pipeline that predicts whether a customer recommends a product. The
data contains numerical, categorical and text features.

Project steps:

Preprocessing steps for the proper data types : numerical, categorical, and text data

Processing text data using `JoinTextColumns`, `SpacyLemmatizer` (tokenizing, lower-casing, lemmatizing, stop-word removal that keeps negations) 

Steps for creating features from text: character counts (`CountCharacter` for spaces, `!`, `?`), Proportion of Part-of-speech PoS (`SpacyPosRatios`), TF-IDF of lemmas 

Model pipeline combining preprocessing, features and prediction : `ColumnTransformer` + logistic regression in one `Pipeline` 

Fine-tune the pipeline's hyperparameters using `GridSearchCV`: 288 combinations x 5 CVfolds

Evaluate on the test data after training on all training data. We have used metrics such as accuracy, precision and recall to evaluate the trained model's performance.

We have also included Exploratory data analysis (incl plots), hyperparameter-effect plots, a
feature-importance plot (top words and features per class), predictions for new
raw reviews, and the saved trained pipeline.

`custom_transformers.py` contains custom transformers which are imported in the notebook.


**Limitations**
* The best `C` (32) and `min_df` (1) are at the edge of the search grid, so a
  wider range might do slightly better. For macro F1, the ten best candidates are very close to each other.
* Part of the gain from tuning comes from `class_weight="balanced"`, which
  trades some precision for recall on the minority class.
* NER was not implemented.

## Built With

* [scikit-learn](https://scikit-learn.org) - pipelines, preprocessing, TF-IDF, logistic regression and GridSearchCV
* [spaCy](https://spacy.io) - tokenizing, lemmatizing and part-of-speech tagging of the review text
* [pandas](https://pandas.pydata.org) - handling the data
* [NumPy](https://numpy.org) - handling numerical ops
* [Matplotlib](https://matplotlib.org) - plots
* [joblib](https://joblib.readthedocs.io) - caching and saving the trained pipeline
* [Jupyter Notebook](https://jupyter.org) - the notebook environment

## License
[License](LICENSE.txt)
