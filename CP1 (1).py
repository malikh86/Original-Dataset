# %%
import os
import sys
import logging
from sklearn.base import clone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        # logging.FileHandler("fig_full_data/debug.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info("Importing the required libraries")

import pandas as pd
import numpy as np


session_id = 1000
logging.info(f'set the session id: {session_id}')

# %%
from pycaret.clustering import ClusteringExperiment

# Get the current working directory
current_directory = os.getcwd()
# Get the parent directory
parent_directory = os.path.dirname(current_directory)
# Get the name of the parent directory
anomaly_method = os.path.basename(parent_directory).split('_')[1]

# Prepare the first pipeline and
logging.info(f"load the the anomaly tagged dataset - {anomaly_method} method")
df = pd.read_csv('../anomaly_tagged_df_for_clustering.csv', index_col=0)

# Assuming your DataFrame is named 'df'
# df = df[df['Anomaly'] == 1]

# df.drop('Anomaly', inplace=True, axis=1)

ignore_features = ['INT_NO', 'RUE_1', 'RUE_2', 'NO_ARRONDISSEMENT', 'ARRONDISSEMENT', 'PERMANENT_OU_TEMPORAIRE','window_start', 'window_end', 'LOC_X', 'LOC_Y', 'Longitude', 'Latitude', 'Anomaly']
numeric_features = ['x1', 'x2', 'x3', 'x4', 'x5',
                    'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'y1']

SP = ClusteringExperiment()
SP.setup(data=df.sample(n=10000, random_state=session_id),
         numeric_features=numeric_features,
         ignore_features =ignore_features,

         # Remove zero variance and perfect remove col-linearity
         low_variance_threshold=0,  # keep all features with non-zero variance,
         # remove_multicollinearity=True,
         # # Minimum absolute Pearson correlation to identify correlated features. The default value removes equal columns.
         # multicollinearity_threshold=0.99,

         pca=True,
         pca_components=0.99,
         pca_method='linear',

         # Normalize and transform the data
         normalize=True,
         # transformation=True,
         normalize_method='minmax',
         # transformation_method='yeo-johnson',

         session_id=session_id,
         experiment_name=f"Clustering Traffic Anomaly Pattern - {anomaly_method}",
         system_log=True,
         # silent=True,
         html=False
         )

# Create the directory if it does not exist
os.makedirs('out', exist_ok=True)

from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer, InterclusterDistance
from sklearn.cluster import AgglomerativeClustering, KMeans

SP_kmeans_plot = KMeans(random_state=session_id)
SP_hclust_plot = clustering = AgglomerativeClustering(linkage='ward', metric='euclidean')
SP_kmeans_num_clusters = 4
SP_hclust_num_clusters = 4

logging.info("Determine the best k for Kmeans")
try:
    visualizer1 = KElbowVisualizer(clone(SP_kmeans_plot), k=(2, 19), metric="distortion", random_state=session_id,
                                   title=f"Distortion Score Elbow for KMeans Clustering ({anomaly_method})", size=(650, 380),
                                   fontdict={'fontsize': 600}
                                   )
    visualizer1.fit(SP.get_config('X_transformed'))
    SP_kmeans_num_clusters = visualizer1.elbow_value_
    visualizer1.show(outpath=f'./out/{anomaly_method.replace(" ", "")}_kmeans_KElbow_distortion.png',
                     clear_figure=True, dpi = 300)
except:
    print("ERROR:KElbowVisualizer for KMeans")

logging.info("Determine the best k for Agglomerative")
try:
    visualizer2 = KElbowVisualizer(clone(SP_hclust_plot), k=(2, 19), metric="distortion", random_state=session_id,
                                   title=f"Distortion Score Elbow for Agglomerative Clustering ({anomaly_method})",
                                   size=(650, 380),
                                   fontdict={'fontsize': 600}
                                   )
    visualizer2.fit(SP.get_config('X_transformed'))
    SP_hclust_num_clusters = visualizer2.elbow_value_
    visualizer2.show(outpath=f'./out/{anomaly_method.replace(" ", "")}_hclust_KElbow_distortion.png',
                     clear_figure=True, dpi = 300)
except:
    print("ERROR:KElbowVisualizer for hclust")

SP.remove_metric('hs')
SP.remove_metric('ari')
SP.remove_metric('cs')

res_list = []
# Compare clustering models
logging.info("Create Kmeans Model")
SP_kmeans = SP.create_model('kmeans', num_clusters=SP_kmeans_num_clusters, round=4)
kmeans_res = SP.pull().T
kmeans_res.columns = ['KMeans']
res_list.append(kmeans_res)
logging.info(f"Model Param: {SP_kmeans.get_params(deep=True)}")

df_p = df.copy()
df_p['Cluster'] = SP.predict_model(SP_kmeans, data=df)['Cluster']
df_p.to_csv('./out/kmeans_df.csv')

logging.info("Create Agglomerative Model")
SP_hclust = SP.create_model('hclust', linkage='ward', metric='euclidean', num_clusters=SP_hclust_num_clusters, round=4)
hclust_res = SP.pull().T
hclust_res.columns = ['Agglomerative']
res_list.append(hclust_res)
logging.info(f"Model Param: {SP_hclust.get_params(deep=True)}")

df_p = df.copy()
df_p['Cluster'] = SP.assign_model(SP_hclust)['Cluster']
df_p.to_csv('./out/hclust_df.csv')

logging.info("Create MeanShift Model")
SP_meanshift = SP.create_model('meanshift', round=4)
meanshift_res = SP.pull().T
meanshift_res.columns = ['MeanShift']
res_list.append(meanshift_res)
logging.info(f"Model Param: {SP_meanshift.get_params(deep=True)}")

df_p = df.copy()
df_p['Cluster'] = SP.predict_model(SP_meanshift, data=df)['Cluster']
df_p.to_csv('./out/meanshift_df.csv')


logging.info('Export the results')
SP_results = pd.concat(res_list, axis=1)
SP_results.to_latex(buf=f'out/{anomaly_method.replace(" ", "")}_Results.tex', caption=f'Comparison of Clustering Methods ({anomaly_method})',
                    label=f'tbl:Comparison_{anomaly_method.replace(" ", "")}', position='t')

logging.info("Plot the InterclusterDistance for KMeans")
try:
    visualizer1 = InterclusterDistance(clone(SP_kmeans), legend=False, random_state=session_id,
                                       title=f"KMeans Intercluster Distance Map (via MDS, {anomaly_method})", size=(550, 450),
                                       fontdict={'fontsize': 600}
                                       )
    visualizer1.fit(SP.get_config('X_transformed'))
    # print(visualizer1.scores_)
    visualizer1.show(outpath=f'./out/{anomaly_method.replace(" ", "")}_kmeans_InterclusterDistance.png',
                     clear_figure=True, dpi = 300)
except:
    print("ERROR:InterclusterDistance for KMeans")

logging.info("Plot InterclusterDistance for Meanshift")
try:
    visualizer1 = InterclusterDistance(clone(SP_meanshift), legend=False, random_state=session_id,
                                       title=f"MeanShift Intercluster Distance Map (via MDS, {anomaly_method})", size=(550, 450),
                                       fontdict={'fontsize': 600}
                                       )
    visualizer1.fit(SP.get_config('X_transformed'))
    visualizer1.show(outpath=f'./out/{anomaly_method.replace(" ", "")}_meanshift_InterclusterDistance.png',
                     clear_figure=True, dpi = 300)
except:
    print("ERROR:InterclusterDistance for Meanshift")

logging.info("Plot the Silhouette analysis for Kmeans")
try:
    visualizer1 = SilhouetteVisualizer(clone(SP_kmeans), colors='yellowbrick', random_state=session_id,
                                       title=f"Silhouette Plot of KMeans Clustering ({anomaly_method})", size=(550, 450),
                                       fontdict={'fontsize': 600}
                                       )
    visualizer1.fit(SP.get_config('X_transformed'))
    visualizer1.show(outpath=f'./out/{anomaly_method.replace(" ", "")}_kmeans_Silhouette.png',
                     clear_figure=True, dpi = 300)
except:
    print("ERROR:SilhouetteVisualizer for KMeans")

print(f'Done ({anomaly_method})')
