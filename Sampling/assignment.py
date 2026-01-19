import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.combine import SMOTETomek


# load data
url = "https://raw.githubusercontent.com/AnjulaMehto/Sampling_Assignment/main/Creditcard_data.csv"
df = pd.read_csv(url)

X = df.drop("Class", axis=1)
y = df["Class"]


# balancing dataset (class 0 and class 1 equal)
ros = RandomOverSampler(random_state=42)
X_bal, y_bal = ros.fit_resample(X, y)

data = pd.DataFrame(X_bal, columns=X.columns)
data["Class"] = y_bal


# create 5 samples
samples = []
n = int(len(data) * 0.2)

for i in range(5):
    samples.append(data.sample(n=n, random_state=10 + i))


# sampling techniques
samplers = [
    ("Sampling1", RandomOverSampler(random_state=42)),
    ("Sampling2", RandomUnderSampler(random_state=42)),
    ("Sampling3", SMOTE(random_state=42)),
    ("Sampling4", NearMiss()),
    ("Sampling5", SMOTETomek(random_state=42))
]


# models M1-M5
models = [
    ("M1", LogisticRegression(max_iter=2000)),
    ("M2", DecisionTreeClassifier(random_state=42)),
    ("M3", RandomForestClassifier(n_estimators=100, random_state=42)),
    ("M4", GaussianNB()),
    ("M5", SVC())
]


# function for scaling
def scale(X_train, X_test):
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test


# store accuracy
acc_table = {}
for m, _ in models:
    acc_table[m] = {s: 0 for s, _ in samplers}


# run for 5 samples
for samp_data in samples:

    Xs = samp_data.drop("Class", axis=1)
    ys = samp_data["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        Xs, ys, test_size=0.25, random_state=42, stratify=ys
    )

    for samp_name, sampler in samplers:

        X_res, y_res = sampler.fit_resample(X_train, y_train)

        for model_name, model in models:

            if model_name in ["M1", "M5"]:   # Logistic Regression and SVM
                X_res_scaled, X_test_scaled = scale(X_res, X_test)
                model.fit(X_res_scaled, y_res)
                pred = model.predict(X_test_scaled)
            else:
                model.fit(X_res, y_res)
                pred = model.predict(X_test)

            acc = accuracy_score(y_test, pred)
            acc_table[model_name][samp_name] += acc


# average accuracy
for m in acc_table:
    for s in acc_table[m]:
        acc_table[m][s] = round(acc_table[m][s] / 5, 4)


accuracy_df = pd.DataFrame(acc_table).T
accuracy_df.to_csv("accuracy_table.csv")


# best sampling technique for each model
best_list = []
for m in accuracy_df.index:
    best_sampling = accuracy_df.loc[m].idxmax()
    best_acc = accuracy_df.loc[m].max()
    best_list.append([m, best_sampling, best_acc])

best_df = pd.DataFrame(best_list, columns=["Model", "BestSampling", "Accuracy"])
best_df.to_csv("best_sampling_per_model.csv", index=False)
