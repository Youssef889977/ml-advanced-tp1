California Housing Price Prediction

Imagine you just joined Machine Learning Housing Corp. Your mission is to predict the median housing price for each census block group (district) in California so that an investment engine can decide where to buy land.
1- Business framing



What type of Machine‑Learning task is this (supervised, unsupervised, reinforcement)?
Is it a regression or classification problem?

Performance metric
Which error metric would you propose first and why?  
When might MAE be preferable?

1 | Getting the Data
1.1 Set‑up 
Create a Python module (or a cell in your notebook) containing the helper below. Complete the missing pieces.
import os, tarfile, urllib.request
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL  = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url: str = HOUSING_URL, housing_path: str = HOUSING_PATH):
    """Download & uncompress the California housing dataset."""
    #  ensure local directory exists
    os.makedirs(housing_path, exist_ok=True)

    #  download the archive only if it isn’t already present
    tgz_path = os.path.join(housing_path, "housing.tgz")
    if not os.path.isfile(tgz_path):
        print(" Downloading …")
        urllib.request.urlretrieve(housing_url, tgz_path)

    #  extract csv
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=housing_path)
    print(" Dataset ready at", housing_path)
Run fetch_housing_data() once. You should get a directory structure like:
├─ datasets/
│  └─ housing/
│     ├─ housing.tgz
│     └─ housing.csv
1.2 Load into pandas
import pandas as pd
def load_housing_data(housing_path: str = HOUSING_PATH) -> pd.DataFrame:
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
housing = load_housing_data()
Print the first five rows (head()), the shape, and the list of columns. What looks like the target variable?
2 | Quick Data Inspection
2.1 Structure & basic stats
Which numeric attribute has the largest standard deviation? Any missing values?
How many instances/observations in the dataset ? 
Use info and describe python method 
You can find out what categories exist and how many districts belong to each category by using the value_counts() method
Distribution visualisation
Plot an histogram for every feature.
%matplotlib inline
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(12,8))
Comment on any skewed distributions you observe. Which features might require scaling or log‑transform?
Creating a Reliable Test Set
Objective: build a train / test (and later validation) split that stays stable over time and properly represents key strata in the data.
Why Separate ? 
In your own words, explain what could go wrong if you tune hyper‑parameters on a test set you already peeked at.
Simple Random Hold‑Out
import numpy as np

def split_train_test(data, test_ratio=0.2, seed=42):
    """Return train_df, test_df using a random permutation."""
    np.random.seed(seed)                # 🔒 repeatability
    shuffled_idx   = np.random.permutation(len(data))
    test_set_size  = int(len(data) * test_ratio)
    test_idx       = shuffled_idx[:test_set_size]
    train_idx      = shuffled_idx[test_set_size:]
    return data.iloc[train_idx], data.iloc[test_idx]

train_set, test_set = split_train_test(housing, test_ratio=0.2)
print(len(train_set), len(test_set))

What does “seed”?
Quick Convenience: train_test_split()
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
Pros: one‑liner, consistent shuffling across multiple arrays.
Cons: still pure random → potential sampling bias.
How to sample correctly : 

When a survey company decides to call 1,000 people to ask them a few questions, they don’t just pick 1,000 people randomly in a phone book. They try to ensure that these 1,000 people are representative of the whole population. For example, the US population is 51.3%
females and 48.7% males, so a well-conducted survey in the US would try to maintain this ratio in the sample: 513 female and 487 male. This is called stratified sampling: the population is divided into homogeneous subgroups called strata, and the right number of instances are sampled from each stratum to guarantee that the test set is representative of the overall population. If the people running the survey used purely random sampling, there would be about a 12% chance of sampling a skewed test set that was either less than 49% female or more than 54% female. Either way, the survey results would be significantly biased.

Stratified Sampling by Income Category
Experts say median_income is the strongest driver of house prices → we’ll keep its distribution intact in both sets.
housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_idx]
    strat_test_set  = housing.loc[test_idx]
Check this : 
strat_test_set["income_cat"].value_counts() / len(strat_test_set)
Compute the proportion of each income_cat in the overall dataset vs. strat_test_set vs. a purely random test split. Paste a small table of the three percentages side by side.
Stratified split matches the global proportions within ±0.1 pp; random split often deviates by >1 pp, esp. for rare categories 1 & 5.


Dose test set generated using stratified sampling has income category proportions almost identical to those in the full dataset ? 
Clean‑up 
for _set in (strat_train_set, strat_test_set):
    _set.drop("income_cat", axis=1, inplace=True)

Discover & Visualize the Training Data
Goal: use plots & correlations to surface patterns that will later guide feature engineering
Isolate an Exploration Copy
# Always explore *only* the training fold
housing = strat_train_set.copy()
(Keeps original split intact; accidental filtering or new columns won’t propagate to the data used later for modelling.)

Geographic Scatterplot
housing.plot(kind="scatter", x=xxx, y=xxx, figsize=(10,7))
 Describe what this picture shows?

Add transparency to reveal density:
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1, figsize=(10,7))
Any surprises ? Which areas exhibit the highest data‑point density now?



Housing Prices on the Map
import matplotlib.pyplot as plt

housing.plot(
    kind="scatter", x="longitude", y="latitude", alpha=0.4, figsize=(10,7),
    s=housing["population"] / 100,              # bubble size
    label="population",
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()
What two variables seem most linked to expensive areas?
Correlation Matrix
corr_matrix = housing.corr(numeric_only=True)
print(corr_matrix["xxx"].sort_values(ascending=False))
corr_matrix.style.background_gradient(cmap='coolwarm',).format(precision=2)
Aside from the target itself, which attribute has the strongest positive correlation with price? Which one is most negatively correlated?

5.5 Scatter‑Matrix Focus
from pandas.plotting import scatter_matrix
sel_attrs = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[sel_attrs], figsize=(12,8))
Zoom on the most promising pair:
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
What artefacts (horizontal lines) do you see? Suggest a rule to drop these capped districts.
Later we can remove these outliers now or log them for later treatment.

Feature Engineering — Attribute Combinations
Goal: craft new, more informative features and verify their correlation with the target value.
6.1 Why Create Ratios?
Raw totals (rooms, bedrooms, population) ignore scale.
Ratios such as rooms / household capture size of dwellings; bedrooms / room hints at luxury vs. cramped.

housing_fe = housing.copy()  # fresh working copy
housing_fe["rooms_per_household"]      = housing_fe["total_rooms"] / housing_fe["households"]
housing_fe["bedrooms_per_room"]        = housing_fe["total_bedrooms"] / housing_fe["total_rooms"]
housing_fe["population_per_household"] = housing_fe["population"]    / housing_fe["households"]
Re‑compute the correlation matrix and list the top 5 attributes most correlated with median_house_value.
Interpret the sign of bedrooms_per_room correlation. What kind of houses does a low ratio indicate?
Data Preparation Pipeline
We now build reusable transformations so future data (validation, production) follows the same rules.
7.1 Separate Labels from Features
housing_prep = strat_train_set.drop("fxxx", axis=1).copy()
housing_labels = strat_train_set["median_house_value"].copy()
7.2 Handling Missing Values (Numerical)

Most Machine Learning algorithms cannot work with missing features, so let’s create
a few functions to take care of them.
We saw earlier that the total_bedrooms attribute has some missing values, so let’s fix this. You have three options:
Drop rows
Drop column
Impute value (median recommended)f

- df.dropna(subset=["total_bedrooms"]) # option 1
- df.drop("total_bedrooms", axis=1) # option 2
- median = df["total_bedrooms"].median() # option 3
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
housing_num = housing_prep.drop(["ocean_proximity"], axis=1)
num_attribs = list(housing_num)
imputer.fit(housing_num)
imputer.statistics_
Why is it important to fit the SimpleImputer only on the training data?

7.3 Full Pre‑processing (Numerical + Categorical)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
cat_attribs = ["xxx"]
oneHot = ColumnTransformer([
	("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs)
])
housing_oneHot = oneHot.fit_transform(housing_prep)
Inspect housing_prepared.shape. How many new columns were created by one‑hot encoding?op

Encoder
Output
When to use
Caveat
OrdinalEncoder
Single integer per row
Ordered categories (e.g. 'low', 'med', 'high')
Implies distance between integers
OneHotEncoder
Sparse binary vector
Unordered categories, small cardinality
High‑dim matrices when many categories

Feature Scaling Strategies
Two popular scalers:
Min‑Max (Normalization) → MinMaxScaler(feature_range=(0,1))
Standardization (Z‑Score) → StandardScaler()
All together (numeric missing values, adding variable and  scaling )
Custom Transformers (Class-Based)
When built-in transformers aren't enough, write your own that plugs into scikit-learn.
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing impor StandardScaler
from sklearn.pipeline import Pipeline
# column indices within the numeric array (after ColumnTransformer selects num_attribs)
rooms_ix, bedrooms_ix, population_ix, households_ix = ( num_attribs.index("total_rooms"),num_attribs.index("total_bedrooms"),
num_attribs.index("population"), num_attribs.index("households"))
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room: bool = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix]/ X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        return np.c_[X, rooms_per_household, population_per_household]
Integrate it into the numeric pipeline (replace the FunctionTransformer step):
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("attribs_adder", CombinedAttributesAdder(add_bedrooms_per_room=True)),
    ("scaler", StandardScaler()),
])


Full Pre‑processing (Numerical + Categorical)
from sklearn.preprocessing import OneHotEncoder
full_pipeline = ColumnTransformer([
	 ("num", num_pipeline, num_attribs),
	("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs)
])
housing_prepared = full_pipeline.fit_transform(housing_prep)
full_pipeline
Sanity Baseline: Constant Predictor
import numpy as np
from sklearn.metrics import mean_squared_error
X_train = housing_prepared            # from Section 7.3
y_train = housing_labels
# baseline that always predicts the training median
baseline_pred = np.full_like(y_train, fill_value=y_train.median(), dtype=float)
baseline_rmse = np.sqrt(mean_squared_error(y_train, baseline_pred))
print(f"Baseline (median) RMSE on training: {baseline_rmse:,.0f}")
Compare baseline_rmse with y_train.std(). What does this tell you?
Predicting a constant yields RMSE ≈ target std. Any trained model should beat this by a clear margin.

8.2 Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
some_data = housing_prep.iloc[:5]
some_labels = y_train.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))
lin_rmse = np.sqrt(mean_squared_error(y_train, lin_reg.predict(X_train)))
print(f"Linear Regression RMSE (train): {lin_rmse:,.0f}")
Is the linear model underfitting? Justify using the scale of median_house_value.
Typical train RMSE around 65–75k vs prices 120–500k → noticeable underfit; expect improvement with nonlinear models and better features.
Decision Tree (Watch for Overfitting)
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)
tree_rmse_train = np.sqrt(mean_squared_error(y_train, tree_reg.predict(X_train)))
print(f"DecisionTree RMSE (train): {tree_rmse_train:,.0f}")
If training RMSE ≈ 0, is the model perfect? What’s your hypothesis?
Cross‑Validation Utility
from sklearn.model_selection import cross_val_score
def rmse_cv(estimator, X, y, cv=10):
    neg_mse = cross_val_score(estimator, X, y,                              scoring="neg_mean_squared_error", cv=cv)
    return np.sqrt(-neg_mse)  # array of RMSEs
# Compare Linear vs Tree
lin_rmse_cv  = rmse_cv(lin_reg,  X_train, y_train)
tree_rmse_cv = rmse_cv(tree_reg, X_train, y_train)
def display_scores(name, scores):
    print(f"{name} CV RMSEs: {np.round(scores,0)}")
    print(f"Mean: {scores.mean():,.0f}  |  Std: {scores.std():,.0f}
")
display_scores("Linear",  lin_rmse_cv)
display_scores("Tree",    tree_rmse_cv)
Task 8‑D. Which model wins on CV? How does CV mean compare to training RMSE for each model, and what does that say about bias/variance?
Random Forest (Ensemble)
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
forest_reg.fit(X_train, y_train)
forest_rmse_cv = rmse_cv(forest_reg, X_train, y_train)
display_scores("RandomForest", forest_rmse_cv)
Is the forest clearly better than the linear and the single tree on CV mean? What about its standard deviation?

Optional) Try a Couple More Models
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
svr_rbf = SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.2)
mlp = MLPRegressor(hidden_layer_sizes=(128,64), activation="relu",
                   learning_rate_init=1e-3, random_state=42, max_iter=500)
for name, est in [("SVR-RBF", svr_rbf), ("MLP", mlp)]:
    scores = rmse_cv(est, X_train, y_train, cv=5)  # 5-fold to save time
    display_scores(name, scores)
Shortlist your top 2 models based on CV mean and stability (std). Justify in 2–3 sentences.

Save Your Work (Models & Metadata)
import joblib
results = {
    "baseline_rmse": float(baseline_rmse),
    "lin_cv":  (float(lin_rmse_cv.mean()),  float(lin_rmse_cv.std())),
    "tree_cv": (float(tree_rmse_cv.mean()), float(tree_rmse_cv.std())),
    "forest_cv": (float(forest_rmse_cv.mean()), float(forest_rmse_cv.std())),
}
joblib.dump({
    "pipeline": full_pipeline,
    "lin_reg": lin_reg,
    "tree_reg": tree_reg,
    "forest_reg": forest_reg,
    "cv_results": results,
}, "models/first_pass_models.joblib")
print("✓ Saved models to models/first_pass_models.joblib")
Why do we save both the pipeline and the estimator together? What can go wrong if we only save the estimator?
Fine‑Tune Your Model
Goal: systematically search hyperparameters, avoid leakage by tuning the full pipeline + model, and extract insights from the best estimator.
10.1 Grid Search (on the Full Pipeline)
Wrap preprocessing and a model together so CV includes all steps:
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

full_model = Pipeline([
    ("preprocess", full_pipeline),       # from Section 7.3 (ColumnTransformer)
    ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
])
param_grid = [
    {
        "model__n_estimators": [50, 100, 200],
        "model__max_features": ["sqrt", 4, 6, 8],
        "model__max_depth": [None, 20, 30],
        # Try toggling a preprocessing hyperparameter too (if using CombinedAttributesAdder)
        "preprocess__num__attribs_adder__add_bedrooms_per_room": [True,
False],
    },
    {
        "model__bootstrap": [False],
        "model__n_estimators": [50, 100],
        "model__max_features": [4, 6, 8],
    },
]
grid_search = GridSearchCV(
    estimator=full_model,
    param_grid=param_grid,
    scoring="neg_mean_squared_error",   # higher is better → negative MSE
    cv=5,
    n_jobs=-1,
    return_train_score=True,
)
grid_search.fit(housing_prep, housing_labels)
Print grid_search.best_params_ and grid_search.best_score_ (convert to RMSE). Are any best values on the edge of your grid?
best_rmse = (-grid_search.best_score_) ** 0.5
print("Best params:", grid_search.best_params_)
print(f"Best CV RMSE: {best_rmse:,.0f}")
If best uses the largest n_estimators or max_features, expand the grid upward and rerun. Default refit=True retrains on full training data.
Build a tidy DataFrame from grid_search.cv_results_ with columns mean_test_rmse, std_test_rmse, and key params. Sort ascending by mean_test_rmse and show top 10.
import pandas as pd
cvres = pd.DataFrame(grid_search.cv_results_)
cvres["mean_test_rmse"] = (-cvres["mean_test_score"]) ** 0.5
cvres["std_test_rmse"]  = cvres["std_test_score"] ** 0.5
cols = [c for c in cvres.columns if c.startswith("param_")] +
["mean_test_rmse","std_test_rmse"]
cvres.sort_values("mean_test_rmse")[cols].head(10)

10.2 Randomized Search (bigger spaces, fixed budget)
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
rand_param_dist = {
    "model__n_estimators": randint(100, 800),
    "model__max_depth": randint(10, 60),
    "model__max_features": ["sqrt", "log2", None, 4, 6, 8],
    "model__min_samples_split": randint(2, 20),
    "model__min_samples_leaf": randint(1, 10),
}
rand_search = RandomizedSearchCV(
    estimator=full_model,
    param_distributions=rand_param_dist,
    n_iter=40,  # adjust to time budget
    scoring="neg_mean_squared_error",
    cv=5,
    random_state=42,
    n_jobs=-1,
)
rand_search.fit(housing_prep, housing_labels)
rand_rmse = (-rand_search.best_score_) ** 0.5
print("Randomized best params:", rand_search.best_params_)
print(f"Randomized best CV RMSE: {rand_rmse:,.0f}")
Which search (grid vs randomized) gave a better CV RMSE per minute of runtime? Briefly justify.

10.3 (Optional) Try a Simple Ensemble
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
stack = StackingRegressor(
    estimators=[
        ("lin", Pipeline([("pre", full_pipeline), ("m",
LinearRegression())])),
        ("svr", Pipeline([("pre", full_pipeline), ("m", SVR(C=10,
epsilon=0.2))])),
    ],
    final_estimator=RandomForestRegressor(n_estimators=200,
random_state=42),
    n_jobs=-1,
)
# or a simple VotingRegressor over pre‑tuned base models
Does stacking improve CV RMSE vs your best single model? If not, speculate why (e.g., base learners too correlated).

10.4 Inspect Feature Importances (from the Best Forest)
def get_col_names(pre):
    col_names=[]
    for name, trans, *_ in pre._iter(
                fitted=True,
                column_as_labels=False,
                skip_empty_columns=True,
                skip_drop=True,
            ):
        if "pipeline" in str(type(trans)):
            for step in  trans.steps:
                name, cols= step[0], step[1].get_feature_names_out()
                print(step,type(step))
                if name=="scaler":continue
                col_names+=list(cols)
        else : col_names+=list(trans.get_feature_names_out())
    return col_names
           
best_model = grid_search.best_estimator_  # or rand_search.best_estimator_
rf = best_model.named_steps["model"]
importances = rf.feature_importances_
# Get feature names after preprocessing
pre = best_model.named_steps["preprocess"]
feat_names =get_col_names(pre)# pre.get_feature_names_out() if there was no ColumnTransformer
feat_imp = pd.DataFrame({"feature": feat_names, "importance": importances})
\
           .sort_values("importance", ascending=False)
feat_imp.head(15)
List the top 10 features. Are engineered ratios (e.g., bedrooms_per_room) among them? Any surprising categorical dummies?

10.5 Error Analysis
Use out‑of‑fold predictions to study residuals without peeking at the test set:
from sklearn.model_selection import cross_val_predict
best = grid_search.best_estimator_
# Predictions from 5‑fold CV on training data
oof_pred = cross_val_predict(best, housing_prep, housing_labels, cv=5,
n_jobs=-1)
residuals = housing_labels - oof_pred
pd.Series(residuals).describe()
# Optional plots
import matplotlib.pyplot as plt
plt.figure(figsize=(5,4)); plt.hist(residuals, bins=50); plt.title("Residuals (OOF)"); plt.show()
plt.figure(figsize=(5,4)); plt.scatter(oof_pred, residuals, alpha=0.2); plt.axhline(0,color='k'); plt.xlabel("Predicted"); plt.ylabel("Residual"); plt.show()
Describe two systematic error patterns you see (e.g., underestimation at very high prices). Propose a feature you’d add to address one pattern.
Save the Tuned Pipeline
import joblib
joblib.dump({
    "best_model": best_model,
    "cv":"grid",
    "best_params": grid_search.best_params_,
    "best_cv_rmse": float(best_rmse),
}, "models/tuned_random_forest.joblib")
print("✓ Saved tuned model → models/tuned_random_forest.joblib")

1 | Final Test‑Set Evaluation
One shot only. Evaluate exactly once on the held‑out test set. If results disappoint, go back to training/CV; do not tweak on the test set.
11.1 Prepare X_test / y_test
# Separate features and labels from the *held‑out* test fold
X_test = strat_test_set.drop("median_house_value", axis=1).copy()
y_test = strat_test_set["median_house_value"].copy()
11.2 Predict with the Tuned Model
Two cases — pick the one matching your setup:
A. Tuned model includes preprocessing (we used a Pipeline in §10.1):
final_model = grid_search.best_estimator_   # or rand_search.best_estimator_
final_predictions = final_model.predict(X_test)  # no manual transform
B. Tuned model is just the estimator (no preprocess inside):
final_model = forest_reg  # example: plain estimator
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
Print final_model and confirm whether it contains a preprocess step. Choose A or B accordingly.
11.3 Report Test Metrics (RMSE, MAE)
from sklearn.metrics import mean_squared_error, mean_absolute_error
final_rmse = mean_squared_error(y_test, final_predictions)
final_mae  = mean_absolute_error(y_test, final_predictions)
print(f"Test RMSE: {final_rmse:,.0f} | Test MAE: {final_mae:,.0f}")
Compare test RMSE to your best CV RMSE. Is there optimism bias (test worse than CV)? Briefly explain why that happens after tuning.
11.4 95% Confidence Interval for RMSE
Compute a CI on RMSE by applying the interval to MSE, then taking the square root.
import numpy as np
from scipy import stats
errors = final_predictions - y_test
se2 = errors ** 2
confidence = 0.95
m = len(se2)
ci_mse = stats.t.interval(confidence, df=m-1, loc=se2.mean(), scale=stats.sem(se2))
ci_rmse = np.sqrt(ci_mse)
print("95% CI for RMSE:", f"[{ci_rmse[0]:,.0f}, {ci_rmse[1]:,.0f}]")
Your previous production model has RMSE = 48k. Does your CI exclude 48k? If not, you cannot claim a significant improvement yet.
11.5 Sanity Checks Before Ship
Feature mismatch: ensure train & test preprocessing use the same fitted pipeline.
Distribution shift: optional — compare key feature histograms between train/test; large shifts may explain errors.
Error slices: compute RMSE by geography (coastal vs inland) or income quintiles.
import pandas as pd
test_df = X_test.copy()
test_df["y_true"] = y_test
test_df["y_pred"] = final_predictions
# Example slice: inland vs others
is_inland = test_df.get("ocean_proximity").eq("INLAND") if "ocean_proximity" in test_df else None
if is_inland is not None:
    rmse_inland = mean_squared_error(test_df.loc[is_inland, "y_true"], test_df.loc[is_inland, "y_pred"], squared=False)
    rmse_other  = mean_squared_error(test_df.loc[~is_inland, "y_true"], test_df.loc[~is_inland, "y_pred"], squared=False)
    print(f"Slice RMSE — INLAND: {rmse_inland:,.0f} | non‑INLAND: {rmse_other:,.0f}")
Identify one segment with notably worse error. Suggest a feature or data source to mitigate it.
11.6 Save Final Artefacts
import joblib, json, sys, sklearn
joblib.dump({
    "final_model": final_model,
    "cv_best_rmse": float(best_rmse) if 'best_rmse' in globals() else None,
    "test_rmse": float(final_rmse),
    "test_mae": float(final_mae),
    "ci_rmse": [float(ci_rmse[0]), float(ci_rmse[1])],
}, "models/final_release.joblib")

meta = {
    "sklearn_version": sklearn.__version__,
    "python_version": sys.version,
    "random_state": 42,
}
with open("models/final_release_meta.json", "w") as f:
    json.dump(meta, f, indent=2)
print("✓ Final model + metadata saved to models/")
Write a 5‑sentence handover note: what works, what doesn’t, assumptions, monitoring ideas, and a rollback plan.
