import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

from IPython.display import display

print("Imports OK. Numpy:", np.__version__)

# 1) Download dataset (Student Performance - Math)
url_mat = "https://raw.githubusercontent.com/sarthak-srivastava/Student-Performance-Dataset/master/student-mat.csv"

df = pd.read_csv(url_mat)   # <--- FIXED: no sep=';'
print("Loaded student-mat.csv — shape:", df.shape)
display(df.head())

# 2) Create a categorical target from final grade G3
#    low: 0-9, medium: 10-14, high: 15-20
def grade_to_level(g):
    if g <= 9:
        return 'low'
    elif g <= 14:
        return 'medium'
    else:
        return 'high'

df['performance'] = df['G3'].apply(grade_to_level)
print("\nTarget distribution:")
print(df['performance'].value_counts())

# 3) Feature selection
numeric_features = ['age', 'absences', 'G1', 'G2', 'studytime', 'failures', 'traveltime']
exclude = set(numeric_features + ['G3', 'performance'])
categorical_features = [c for c in df.columns if c not in exclude]

print("\nNumeric features:", numeric_features)
print("Number of categorical features:", len(categorical_features))

# 4) Preprocessing pipelines
numeric_transformer = Pipeline([('scaler', StandardScaler())])
categorical_transformer = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# 5) Train/test split (stratify by target)
X = df[numeric_features + categorical_features].copy()
y = df['performance'].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("\nTrain/test shapes:", X_train.shape, X_test.shape)
print("Train class distribution:\n", y_train.value_counts(normalize=True))

# 6) Decision Tree pipeline
dt_pipeline = Pipeline([
    ('pre', preprocessor),
    ('clf', DecisionTreeClassifier(max_depth=6, random_state=42))
])

dt_pipeline.fit(X_train, y_train)
y_pred_dt = dt_pipeline.predict(X_test)

print("\n=== Decision Tree Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

# 7) Gaussian Naive Bayes
#    GaussianNB expects dense numeric arrays

# Use a separate preprocessor fit for Naive Bayes
pre_nb = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

Xtr_trans = pre_nb.fit_transform(X_train)
Xte_trans = pre_nb.transform(X_test)

if hasattr(Xtr_trans, "toarray"):
    Xtr_dense = Xtr_trans.toarray()
    Xte_dense = Xte_trans.toarray()
else:
    Xtr_dense = Xtr_trans
    Xte_dense = Xte_trans

gnb = GaussianNB()
gnb.fit(Xtr_dense, y_train)
y_pred_gnb = gnb.predict(Xte_dense)

print("\n=== Gaussian Naive Bayes Results ===")
print("Accuracy:", accuracy_score(y_test, y_pred_gnb))
print(classification_report(y_test, y_pred_gnb))

# 8) Cross-validated comparison (5-fold)
print("\n=== 5-fold Cross Validation ===")
print("Decision Tree (with preprocessing pipeline):")
scores_dt = cross_val_score(dt_pipeline, X, y, cv=5, scoring='accuracy')
print("DT mean {:.3f} ± {:.3f}".format(scores_dt.mean(), scores_dt.std()))

print("\nGaussianNB (on preprocessed dense features):")
X_all_trans = pre_nb.fit_transform(X)
if hasattr(X_all_trans, "toarray"):
    X_all_dense = X_all_trans.toarray()
else:
    X_all_dense = X_all_trans

scores_gnb = cross_val_score(GaussianNB(), X_all_dense, y, cv=5, scoring='accuracy')
print("GNB mean {:.3f} ± {:.3f}".format(scores_gnb.mean(), scores_gnb.std()))

# 9) Confusion matrices (matplotlib-only)
# Confusion Matrix is a table used to evaluate the performance of a classification model.
def plot_confusion(cm, classes, title='Confusion matrix'):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha='center', va='center',
                color='white' if cm[i, j] > cm.max()/2 else 'black'
            )
    plt.tight_layout()
    plt.show()

classes = ['low', 'medium', 'high']

cm_dt = confusion_matrix(y_test, y_pred_dt, labels=classes)
plot_confusion(cm_dt, classes, title='Decision Tree')

cm_gnb = confusion_matrix(y_test, y_pred_gnb, labels=classes)
plot_confusion(cm_gnb, classes, title='GaussianNB')

# 10) Plot top levels of the decision tree (for interpretability)
trained_tree = dt_pipeline.named_steps['clf']

plt.figure(figsize=(20, 8))
plot_tree(
    trained_tree,
    class_names=classes,
    filled=True,
    rounded=True,
    max_depth=3   # just top levels so it stays readable
)
plt.title('Decision Tree (top levels)')
plt.show()

# 11) Live prediction / example (edit during demo)
example = {
    'age': 17, 'absences': 4, 'G1': 12, 'G2': 13, 'studytime': 2,
    'failures': 0, 'traveltime': 1,
    # categorical fields - pick realistic values from dataset
    'school': 'GP', 'sex': 'F', 'address': 'U', 'famsize': 'GT3', 'Pstatus': 'T',
    'Medu': 2, 'Fedu': 2, 'Mjob': 'at_home', 'Fjob': 'teacher', 'reason': 'course',
    'guardian': 'mother', 'schoolsup': 'no', 'famsup': 'no', 'paid': 'no',
    'activities': 'no', 'nursery': 'yes', 'higher': 'yes', 'internet': 'yes',
    'romantic': 'no', 'famrel': 4, 'freetime': 3, 'goout': 3,
    'Dalc': 1, 'Walc': 1, 'health': 3
}

def make_example_df(example_dict, X_template):
    ex = {}
    for c in X_template.columns:
        if c in example_dict:
            ex[c] = example_dict[c]
        else:
            if c in numeric_features:
                ex[c] = X_template[c].median()
            else:
                ex[c] = X_template[c].mode()[0]
    return pd.DataFrame([ex])

ex_df = make_example_df(example, X_train.reset_index(drop=True))
print("\nExample student row:")
display(ex_df.head())

# Predict with Decision Tree pipeline
pred_dt = dt_pipeline.predict(ex_df)
pred_proba_dt = dt_pipeline.predict_proba(ex_df)

# Predict with GaussianNB (using the NB preprocessor)
ex_trans_nb = pre_nb.transform(ex_df)
if hasattr(ex_trans_nb, "toarray"):
    ex_dense_nb = ex_trans_nb.toarray()
else:
    ex_dense_nb = ex_trans_nb

pred_gnb = gnb.predict(ex_dense_nb)
pred_proba_gnb = gnb.predict_proba(ex_dense_nb)

print("\n=== Live prediction ===")
print("Decision Tree ->", pred_dt[0], " probs:", pred_proba_dt[0])
print("GaussianNB   ->", pred_gnb[0], " probs:", pred_proba_gnb[0])

print("\nDemo complete.")
