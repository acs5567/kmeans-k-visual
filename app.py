import io
from flask import Flask, render_template, request

import random
from flask import Response
# from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib
matplotlib.use('Agg')
from io import BytesIO
import base64

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances_argmin

app = Flask(__name__, template_folder='.')

@app.route('/',methods=['GET','POST'])
def plot_png():
    
    k = 1
    
    if request.method == 'POST':
        k = request.form['k']
        k = int(k)
        
    img = BytesIO()
    fig = create_figure(k)
    fig.savefig(img, format='png')
#     fig.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template('index.html', plot_url=plot_url)
    
def generate_data():
    # Generate sample data
    np.random.seed(10)
    # create raw data
    batch_size = 45
    X, labels_true = make_blobs(n_samples=3000, centers=5, cluster_std=0.7)

    # do kmeans
    k_range = np.arange(2,15,1)
    elbow_data = []
    sil_data = []

    for k in k_range:

        clusterer = KMeans(n_clusters=k)
        cluster_labels = clusterer.fit_predict(X)

        elbow_data.append(clusterer.inertia_)
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        sil_data.append(silhouette_avg)
        
    return X, k_range, elbow_data, sil_data
  
def create_figure(k_chosen=5):

    fig = plt.figure(figsize=(15,10))

    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1)
    ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1)
    ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

    colors = [np.random.rand(3,) for c in range(k_chosen)]

    # do chosen_kmeans
    clusterer = KMeans(n_clusters=k_chosen)
    clusterer.fit(X)
    k_means_cluster_centers = clusterer.cluster_centers_
    k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)

    # Raw data plo

    for k, col in zip(range(k_chosen), colors):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        ax3.plot(X[my_members, 0], X[my_members, 1], 'w',
                markerfacecolor=col, marker='o')
        ax3.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=10)

    ax3.set_title(f'kMeans Results with k={k_chosen}')
    ax3.set_xticks(())
    ax3.set_yticks(())

    # elbow curve
    ax1.plot(k_range, elbow_data, 'x-', c='black')
    ax1.set_title(f'kMeans Elbow Curve')

    # silhoutte curve
    ax2.plot(k_range, sil_data, 'x-', c='black')
    ax2.set_title(f'kMeans Silhouette Curve')

    return fig

if __name__ == "__main__":
    
    X, k_range, elbow_data, sil_data = generate_data()
    
    app.run(port=5000, debug=True)