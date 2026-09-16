import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("placement_predict_50k Dataset (3).csv")

print(data.columns.tolist())

X = data[
    [
        "CGPA",
        "AptitudeTestScore"
    ]
]

print("\nSelected Features:")
print(X.head())


wcss = []

for k in range(1, 11):
    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker="o")

plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")

plt.show()

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

data["Cluster"] = kmeans.fit_predict(X)

print("\nCluster Results:")
print(
    data[
        [
            "CGPA",
            "AptitudeTestScore",
            "Cluster"
        ]
    ].head(10)
)

print("\nCluster Centers:")
print(kmeans.cluster_centers_)

print("\nNumber of Students in Each Cluster:")
print(data["Cluster"].value_counts().sort_index())



plt.scatter(
    data["CGPA"],
    data["AptitudeTestScore"],
    c=data["Cluster"]
)

plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    marker="X",
    s=200
)

plt.xlabel("CGPA")
plt.ylabel("Aptitude Test Score")
plt.title("K-Means Clustering")

plt.show()