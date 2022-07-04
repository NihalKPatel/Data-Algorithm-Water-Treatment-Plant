import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import preprocessing, metrics
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
# ------------------------------- preprocessing of dataset WaterTreatment -------------------------------#
from yellowbrick.cluster import SilhouetteVisualizer

data = pd.read_csv("water-treatment.csv", na_values='?')
data.info()
headerList = ["Day", "Q.E", "ZN.E", "PH.E", "DBO.E", "DQO.E", "SS.E", "SSV.E", "SED.E", "COND.E",
              "PH.P", "DBO.P", "SS.P", "SSV.P", "SED.P", "COND.P", "PH.D", "DBO.D", "DQO.D",
              "SS.D", "SSV.D", "SED.D", "COND.D", "PH.S", "DBO.S", "DQO.S", "SS.S", "SSV.S",
              "SED.S", "COND.S", "RD.DBO.P", "RD.SS.P", "RD.SED.P", "RD.DBO.S", "RD.DQO.S",
              "RD.DBO.G", "RD.DQO.G", "RD.SS.G", "RD.SED.G"]

data.to_csv("water-treatment_updated.csv", header=headerList, index=False)

file = 'water-treatment_updated.csv'
with open(file, 'r'):
    data = pd.read_csv(file)
data.info()
# Verify
data.count()
data.head()
print('data shape: ', data.shape)
data.count()  # confirmed
# Null values
print("NULL")
print(data[data.isnull().any(axis=1)])  # no null values
# NaN values
print("NAN")
print(data[data.isna().any(axis=1)])  # no nan values
# Duplicates
print(data.duplicated().any())  # No duplicated rows
# Single value columns
print(data.nunique())  # No columns with single same value

data.head(10)

# Export to csv and save cleaned dataset


# ------------------------------- exploration of dataset WaterTreatment -------------------------------#
data.isnull().sum()
data.isna().sum()
data.duplicated().sum()

file = 'water-treatment_updated.csv'
data.to_csv(file, index=False)

file = 'water-treatment_updated.csv'
with open(file, 'r'):
    data = pd.read_csv(file)

plt.figure(figsize=(30, 20))
data = data[["Q.E", "ZN.E", "PH.E", "DBO.E", "DQO.E", "SS.E", "SSV.E", "SED.E", "COND.E",
             "PH.P", "DBO.P", "SS.P", "SSV.P", "SED.P", "COND.P", "PH.D", "DBO.D", "DQO.D",
             "SS.D", "SSV.D", "SED.D", "COND.D", "PH.S", "DBO.S", "DQO.S", "SS.S", "SSV.S",
             "SED.S", "COND.S", "RD.DBO.P", "RD.SS.P", "RD.SED.P", "RD.DBO.S", "RD.DQO.S",
             "RD.DBO.G", "RD.DQO.G", "RD.SS.G", "RD.SED.G"]]

mask = np.triu(np.ones_like(data.corr(), dtype=bool))
heatmap = sns.heatmap(data.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Triangle Correlation Heatmap')
plt.show()

# ZN-E, PH, COND-E

# data.drop('ZN.E', axis=1, inplace=True)
# data.drop('PH.E', axis=1, inplace=True)
# data.drop('COND.E', axis=1, inplace=True)

file = 'water-treatment_updated.csv'
data.to_csv(file, index=False)

# -------------------------------Feature Selection Water Treatment -------------------------------#
file = 'water-treatment_updated.csv'
with open(file, 'r'):
    data = pd.read_csv(file)

le = preprocessing.LabelEncoder()
data = data.apply(le.fit_transform)

# ANOVA FEATURE SELECTION
X = data.iloc[:, 1:]
y = data.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
fs = SelectKBest(score_func=f_classif, k=10)
fs.fit(X_train, y_train)
# transform train input data
X_train_fs = fs.transform(X_train)
# transform test input data
X_test_fs = fs.transform(X_test)
fs_score_df = pd.DataFrame()
feature_no = 0
feature_score = 0

for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))
    fs_score_df = fs_score_df.append({feature_no: i,
                                      feature_score: fs.scores_[i]}, ignore_index=True)
# plot the scores
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
fs_score_df['Feature Name'] = data.columns[1:]
fs_score_df.rename(columns={0: 'Score'}, inplace=True)
fs_score_df = fs_score_df.sort_values(by=['Score'], ascending=False)
plt.xlabel('Water Treatment Features')
plt.ylabel('Feature Selection Scores')
plt.show()

# %% Reduced Feature Set - by removing highly correlated features
# Load file
file = 'water-treatment_updated.csv'
with open(file, 'r'):
    data = pd.read_csv(file)

data.drop(['ZN.E', 'PH.E', 'COND.E'], axis=1, inplace=True)
le = preprocessing.LabelEncoder()
data = data.apply(le.fit_transform)

# ANOVA Refined No High correlation


X = data.iloc[:, 1:]
y = data.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
fs = SelectKBest(score_func=f_classif, k=6)
fs.fit(X_train, y_train)
# transform train input data
X_train_fs = fs.transform(X_train)
# transform test input data
X_test_fs = fs.transform(X_test)
fs_score_df = pd.DataFrame()
feature_no = 0
feature_score = 0
print("Completed water_treatment Feature Selection")
for i in range(len(fs.scores_)):
    print('Feature %d: %f' % (i, fs.scores_[i]))
    fs_score_df = fs_score_df.append({feature_no: i,
                                      feature_score: fs.scores_[i]}, ignore_index=True)
# plot the scores
plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
fs_score_df['Feature Name'] = data.columns[1:]
fs_score_df.rename(columns={0: 'Score'}, inplace=True)
fs_score_df = fs_score_df.sort_values(by=['Score'], ascending=False)
plt.xlabel('Water Treatment Features')
plt.ylabel('Feature Selection Scores')
plt.show()
# ------------------------------- K Means water_treatment  ------------------------------- #
file = 'water-treatment_updated.csv'
with open(file, 'r'):
    data = pd.read_csv(file, encoding='unicode_escape')

data = data[["Q.E", "DBO.E", "DQO.E", "SS.E", "SSV.E", "SED.E", "PH.P", "DBO.P", "SS.P", "SSV.P",
             "SED.P", "COND.P", "PH.D", "DBO.D", "DQO.D", "SS.D", "SSV.D", "SED.D", "COND.D",
             "PH.S", "DBO.S", "DQO.S", "SS.S", "SSV.S", "SED.S", "COND.S", "RD.DBO.P", "RD.SS.P",
             "RD.SED.P", "RD.DBO.S", "RD.DQO.S", "RD.DBO.G", "RD.DQO.G", "RD.SS.G", "RD.SED.G"]]

X = data.loc[:, ['Q.E', 'RD.DQO.S']]

# image size
plt.figure(figsize=(10, 5))

# ploting scatered graph
plt.scatter(x=X['Q.E'], y=X['RD.DQO.S'])
plt.xlabel('Q.E')
plt.ylabel('RD.DQO.S')
plt.show()

# ------------------------------- K VALUE water_treatment  ------------------------------- #
wcss = []

# for loop
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imp.fit_transform(X)

for i in range(1, 11):
    # k-mean cluster model for different k values
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)

    # inertia method returns wcss for that model
    wcss.append(kmeans.inertia_)

# figure size
plt.figure(figsize=(10, 5))
sns.lineplot(range(1, 11), wcss, marker='o', color='green')
# labeling
plt.title('The Elbow Method Water')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
Kmean_n_clusters_Water = 3
# ------------------------------- Kmeans water_treatment with clusters ------------------------------- #
# Elbow Method Shows 3 is the optimal number of clusters
start = time.time()
file = 'water-treatment_updated.csv'
with open(file, 'r'):
    data = pd.read_csv(file, encoding='unicode_escape')

data = data[['Q.E', 'RD.DQO.S']]

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
data = imp.fit_transform(data)

scaler = StandardScaler()
data = scaler.fit_transform(data)

km = KMeans(Kmean_n_clusters_Water)
km.fit(data)

plt.figure(figsize=(10, 5))
scatter = plt.scatter(x=data[:, 0], y=data[:, 1], c=km.labels_, cmap="Set2")
plt.xlabel('Q.E')
plt.ylabel('RD.DQO.S')
plt.legend(handles=scatter.legend_elements()[0], labels=[0, 1, 2, 3])
plt.show()

finish = time.time()
# ------------------------------- Kmeans with Time Taken water_treatment ------------------------------- #

Kmean_time_taken = {finish - start}

# ------------------------------- Kmeans water_treatment with Davis-Bouldin score  ------------------------------- #

file = 'water-treatment_updated.csv'
with open(file, 'r'):
    data = pd.read_csv(file, encoding='unicode_escape')

X = data['Q.E']

scaler = StandardScaler()

X = scaler.fit_transform(pd.DataFrame(X))

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imp.fit_transform(X)

scaler = StandardScaler()
X = scaler.fit_transform(X)

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=30).fit(X)
labels = kmeans.fit_predict(X)

kmeans_Davis_Bouldin_score = davies_bouldin_score(X, labels)

# ------------------------------- Kmeans water_treatment CSM ------------------------------- #
file = 'water-treatment_updated.csv'
with open(file, 'r'):
    data = pd.read_csv(file, encoding='unicode_escape')

y = data['RD.DQO.S']
x = data[['Q.E']]

scaler = StandardScaler()

X = scaler.fit_transform(pd.DataFrame(X))

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imp.fit_transform(X)

scaler = StandardScaler()
x = scaler.fit_transform(X)

km = KMeans(n_clusters=3, random_state=42)

km.fit_predict(X)

kmeans_silhouette_avg = silhouette_score(X, km.labels_, metric='euclidean')

fig, ax = plt.subplots(2, 2, figsize=(15, 8))
for i in [2, 3, 4, 5]:
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
    plt.title("$k={}$".format(i))
    if i in (2, 3):
        plt.ylabel("Cluster")

    if i in (4, 5):
        plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.xlabel("Silhouette Coefficient")
    q, mod = divmod(i, 2)

    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q - 1][mod])
    visualizer.fit(x)
plt.show()
# ------------------------------- Agglomerative water_treatment with clusters ------------------------------- #
start = time.time()

file = 'water-treatment_updated.csv'
with open(file, 'r'):
    data = pd.read_csv(file, encoding='unicode_escape')

data = data[['Q.E', 'RD.DQO.S']]

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
data = imp.fit_transform(data)

scaler = StandardScaler()
data = scaler.fit_transform(data)

aggloclust = AgglomerativeClustering(n_clusters=3).fit(data)
plt.figure(figsize=(10, 5))
labels = aggloclust.labels_
plt.scatter(x=data[:, 0], y=data[:, 1], c=labels, cmap="Set2")
plt.xlabel('shares')
plt.ylabel('comments')
plt.title("2 Cluster Agglomerative ")
plt.show()
finish = time.time()
# ------------------------------- Agglomerative water_treatment with Time Taken ------------------------------- #

Agglomerative_time_taken = finish - start

# ------------------------------- Agglomerative water with Davis-Bouldin score  ------------------------------- #
file = 'water-treatment_updated.csv'
with open(file, 'r'):
    data = pd.read_csv(file, encoding='unicode_escape')

X = data['Q.E']

scaler = StandardScaler()

X = scaler.fit_transform(pd.DataFrame(X))

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imp.fit_transform(X)

scaler = StandardScaler()
x = scaler.fit_transform(X)

n_clusters = 3
model = AgglomerativeClustering(n_clusters=3)
# fit model and predict clusters
yhat_2 = model.fit_predict(x)

Agglomerative_Davis_Bouldin_score = davies_bouldin_score(x, yhat_2)

# ------------------------------- Agglomerative with CSM ------------------------------- #
file = 'water-treatment_updated.csv'
with open(file, 'r'):
    data = pd.read_csv(file, encoding='unicode_escape')

y = data['RD.DQO.S']
x = data[['Q.E']]

scaler = StandardScaler()

X = scaler.fit_transform(pd.DataFrame(X))

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imp.fit_transform(X)

scaler = StandardScaler()
x = scaler.fit_transform(X)

agg_avg = AgglomerativeClustering(linkage='average', n_clusters=3)
as_avg = agg_avg.fit(x)
Agglomerative_silhouette_avg = silhouette_score(x, as_avg.labels_, metric='euclidean')

# ------------------------------- DBSCAN water_treatment with clusters ------------------------------- #
start = time.time()

file = 'water-treatment_updated.csv'
with open(file, 'r'):
    data = pd.read_csv(file, encoding='unicode_escape')

data = data[['Q.E', 'RD.DQO.S']]

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
data = imp.fit_transform(data)

scaler = StandardScaler()
data = scaler.fit_transform(data)

X = np.nan_to_num(data)
X = np.array(X, dtype=np.float64)
X = StandardScaler().fit_transform(X)

db = DBSCAN(eps=0.4, min_samples=5).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

realClusterNum = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
clusterNum = len(set(db.labels_))

plt.figure(figsize=(15, 10))
unique_labels = set(db.labels_)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]
class_member_mask = (db.labels_ == k)
xy = X[class_member_mask & core_samples_mask]
plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
         markeredgecolor='k', markersize=14)
xy = X[class_member_mask & ~core_samples_mask]
plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
         markeredgecolor='k', markersize=6)
plt.title('Estimated Number of Clusters: %d' % realClusterNum, fontweight='bold', fontsize=20)
plt.legend(fontsize=20)
n_noise_ = list(db.labels_).count(-1)
print('number of noise(s): ', n_noise_)
plt.xlabel('Q.E')
plt.ylabel('RD.DQO.S')
plt.show()
finish = time.time()
# ------------------------------- DBSCAN water_treatment with Time Taken ------------------------------- #

DBSCAN_time_taken = finish - start

# ------------------------------- DBSCAN water_treatment with Davis-Bouldin score  ------------------------------- #
file = 'water-treatment_updated.csv'
with open(file, 'r'):
    data = pd.read_csv(file, encoding='unicode_escape')

x = data['Q.E']

scaler = StandardScaler()

x = scaler.fit_transform(pd.DataFrame(x))

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
x = imp.fit_transform(x)

scaler = StandardScaler()
x = scaler.fit_transform(x)

db = DBSCAN(eps=0.3, min_samples=10).fit(x)

DBSCAN_Davis_Bouldin_score = davies_bouldin_score(x, db.labels_)

# ------------------------------- DBSCAN with CSM ------------------------------- #
file = 'water-treatment_updated.csv'
with open(file, 'r'):
    data = pd.read_csv(file, encoding='unicode_escape')
y = data['RD.DQO.S']
x = data[['Q.E']]

scaler = StandardScaler()

X = scaler.fit_transform(pd.DataFrame(X))

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
X = imp.fit_transform(X)

scaler = StandardScaler()
x = scaler.fit_transform(X)

db = DBSCAN(eps=0.3, min_samples=10).fit(x)

DBSCAN_silhouette_avg = metrics.silhouette_score(x, db.labels_)

# ------------------------------- Kmeans in Tabulate Form water_treatment ------------------------------- #

all_data = [["CSM", "Davis Bouldin Score", "TimeTaken"],
            [("No.Cluster", Kmean_n_clusters_Water, "Avg", kmeans_silhouette_avg), kmeans_Davis_Bouldin_score,
             (Kmean_time_taken, "Secs in Kmean")],
            [("No.Cluster", Kmean_n_clusters_Water, "Avg", Agglomerative_silhouette_avg),
             Agglomerative_Davis_Bouldin_score,
             (Agglomerative_time_taken, "Secs in Agglomerative")],
            [("No.Cluster", Kmean_n_clusters_Water, "Avg", DBSCAN_silhouette_avg),
             DBSCAN_Davis_Bouldin_score,
             (DBSCAN_time_taken, "Secs in DBscan")]
            ]
print(tabulate(all_data, headers='firstrow', tablefmt='grid'))
