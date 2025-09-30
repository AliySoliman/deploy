import streamlit as st
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

def hierarchy_script(df, features, n_clusters, linkage, metric, compute_full_tree, distance_threshold, edit):
    # Prepare data
    X = df[features].values
    
    # Standardize the data (important for clustering)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create and fit the model
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric=metric,
        linkage=linkage,
        compute_full_tree=compute_full_tree,
        distance_threshold=distance_threshold
    )
    
    # Make predictions (cluster labels)
    cluster_labels = model.fit_predict(X_scaled)
    
    # Calculate clustering metrics
    try:
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    except:
        silhouette_avg = None
    
    try:
        calinski_score = calinski_harabasz_score(X_scaled, cluster_labels)
    except:
        calinski_score = None
        
    try:
        davies_score = davies_bouldin_score(X_scaled, cluster_labels)
    except:
        davies_score = None
    
    # Store results
    model_results = {
        'model': model,
        'scaler': scaler,
        'metrics': {
            'Silhouette Score': silhouette_avg,
            'Calinski-Harabasz Score': calinski_score,
            'Davies-Bouldin Score': davies_score,
            'Number of Clusters': n_clusters,
            'Cluster Sizes': [sum(cluster_labels == i) for i in range(n_clusters)]
        },
        'cluster_labels': cluster_labels,
        'features': features,
        'X_scaled': X_scaled  # Store scaled data for visualization
    }
    
    st.success("Hierarchical Clustering model created successfully!")
    return model_results

def hierarchy_validate_model(params):
    if len(params['features']) < 2:
        st.error("Please select at least two feature columns for clustering")
        return False
    
    # Check if Ward linkage is used with Euclidean distance
    if params['linkage'] == 'ward' and params['metric'] != 'euclidean':
        st.error("Ward linkage can only be used with Euclidean distance metric")
        return False
    
    # Validate number of clusters
    if params['n_clusters'] < 2:
        st.error("Number of clusters must be at least 2")
        return False
    
    if params['n_clusters'] > len(params['df']):
        st.error("Number of clusters cannot exceed the number of data points")
        return False
    
    return True