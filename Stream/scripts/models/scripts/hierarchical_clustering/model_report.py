import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

def hierarchy_report():
    results = st.session_state.model_results
    
    # Main report section
    st.markdown("""
                        <div class="report-container">
                            <h3>Clustering Performance Report</h3>
                            <div class="metric-card">
                                <strong>Features Used:</strong> {features}<br>
                                <strong>Number of Clusters:</strong> {n_clusters}
                            </div>
                            <div class="metric-card">
                                <strong>Silhouette Score:</strong> {silhouette:.4f}<br>
                                <strong>Calinski-Harabasz Score:</strong> {calinski:.4f}<br>
                                <strong>Davies-Bouldin Score:</strong> {davies:.4f}
                            </div>
                            <div class="metric-card">
                                <strong>Cluster Sizes:</strong><br>
                                {cluster_sizes}
                            </div>
                            <div class="metric-card">
                                <strong>Algorithm Parameters:</strong><br>
                                <strong>Linkage Method:</strong> {linkage}<br>
                                <strong>Distance Metric:</strong> {metric}<br>
                                <strong>Compute Full Tree:</strong> {full_tree}
                            </div>
                        </div>
                        """.format(
                            features=", ".join(results['features']),
                            n_clusters=results['metrics']['Number of Clusters'],
                            silhouette=results['metrics']['Silhouette Score'] if results['metrics']['Silhouette Score'] is not None else 0,
                            calinski=results['metrics']['Calinski-Harabasz Score'] if results['metrics']['Calinski-Harabasz Score'] is not None else 0,
                            davies=results['metrics']['Davies-Bouldin Score'] if results['metrics']['Davies-Bouldin Score'] is not None else 0,
                            cluster_sizes="<br>".join([f"&nbsp;&nbsp;Cluster {i}: {size} points" 
                                                    for i, size in enumerate(results['metrics']['Cluster Sizes'])]),
                            linkage=results['model'].linkage,
                            metric=results['model'].metric,
                            full_tree=results['model'].compute_full_tree
                        ), unsafe_allow_html=True)
    
    # Visualization Section
    st.markdown("---")
    st.subheader("Cluster Visualization")
    
    # Get the data and cluster labels
    X = results['X_scaled']
    cluster_labels = results['cluster_labels']
    n_clusters = results['metrics']['Number of Clusters']
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Dendrogram", "2D PCA Projection", "Cluster Distribution", "Silhouette Analysis"])
    
    with tab1:
        # Dendrogram - Shows the merging process
        st.write("**Dendrogram - Cluster Merging Hierarchy**")
        
        # Calculate linkage matrix for dendrogram
        try:
            # Use the same parameters as the model
            Z = linkage(X, method=results['model'].linkage, metric=results['model'].metric)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create dendrogram
            dendrogram(
                Z,
                truncate_mode='lastp',  # Show only the last p merged clusters
                p=n_clusters,
                show_leaf_counts=True,
                leaf_rotation=90.,
                leaf_font_size=8.,
                show_contracted=True,
                ax=ax
            )
            
            # Add cutoff line for current number of clusters
            if n_clusters > 1:
                # Find the height where we get n_clusters
                last_merge_height = Z[-n_clusters, 2]
                ax.axhline(y=last_merge_height, color='r', linestyle='--', 
                          label=f'Cutoff for {n_clusters} clusters')
            
            ax.set_title(f'Dendrogram ({results["model"].linkage} linkage)')
            ax.set_xlabel('Data points or cluster size')
            ax.set_ylabel('Distance')
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            
            # Dendrogram interpretation
            st.markdown("""
            **How to read the dendrogram:**
            - **Vertical lines**: Represent clusters being merged
            - **Height**: Distance between clusters when merged (higher = less similar)
            - **Horizontal lines**: Show which clusters/points are connected
            - **Cutoff line**: Red dashed line shows where clusters are cut to get your chosen number
            """)
            
        except Exception as e:
            st.error(f"Could not create dendrogram: {str(e)}")
            st.info("Try using Euclidean distance with Ward linkage for best results")
    
    with tab2:
        # 2D PCA Projection
        st.write("**2D PCA Projection of Clusters**")
        if X.shape[1] >= 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.7, s=50)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            ax.set_title('Cluster Visualization (PCA Projection)')
            plt.colorbar(scatter, ax=ax, label='Cluster')
            st.pyplot(fig)
        else:
            st.info("Need at least 2 features for 2D visualization")
    
    with tab3:
        # Cluster Distribution
        st.write("**Cluster Size Distribution**")
        cluster_sizes = results['metrics']['Cluster Sizes']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Pie chart
        colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
        wedges, texts, autotexts = ax1.pie(cluster_sizes, labels=[f'Cluster {i}' for i in range(n_clusters)], 
                                         autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Cluster Distribution')
        
        # Bar chart
        bars = ax2.bar(range(n_clusters), cluster_sizes, color=plt.cm.viridis(np.linspace(0, 1, n_clusters)))
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Number of Points')
        ax2.set_title('Cluster Sizes')
        ax2.set_xticks(range(n_clusters))
        ax2.set_xticklabels([f'Cluster {i}' for i in range(n_clusters)])
        
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with tab4:
        # Silhouette Analysis
        st.write("**Silhouette Analysis**")
        if results['metrics']['Silhouette Score'] is not None:
            from sklearn.metrics import silhouette_samples
            silhouette_vals = silhouette_samples(X, cluster_labels)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            y_ticks = []
            y_lower = y_upper = 0
            
            for i in range(n_clusters):
                cluster_silhouette_vals = silhouette_vals[cluster_labels == i]
                cluster_silhouette_vals.sort()
                y_upper += len(cluster_silhouette_vals)
                color = plt.cm.viridis(i / n_clusters)
                ax.barh(range(y_lower, y_upper), cluster_silhouette_vals, height=1.0, 
                       edgecolor='none', color=color)
                y_ticks.append((y_lower + y_upper) / 2)
                y_lower += len(cluster_silhouette_vals)
            
            ax.axvline(results['metrics']['Silhouette Score'], color="red", linestyle="--", 
                      label=f'Average: {results["metrics"]["Silhouette Score"]:.3f}')
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([f'Cluster {i}' for i in range(n_clusters)])
            ax.set_xlabel('Silhouette Coefficient')
            ax.set_ylabel('Cluster')
            ax.set_title('Silhouette Plot for Clusters')
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("Silhouette score not available for this configuration")

# Optional: Enhanced dendrogram with interactive features
def create_interactive_dendrogram(X, linkage_method, metric, n_clusters):
    """Create a more detailed dendrogram"""
    Z = linkage(X, method=linkage_method, metric=metric)
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Create detailed dendrogram
    ddata = dendrogram(
        Z,
        truncate_mode='lastp',
        p=min(20, n_clusters * 3),  # Show more detail
        show_leaf_counts=True,
        leaf_rotation=90.,
        leaf_font_size=10,
        show_contracted=True,
        ax=ax
    )
    
    # Add cluster cutoff lines for different k values
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, k in enumerate([n_clusters, n_clusters+1, n_clusters-1]):
        if k > 1 and k <= len(X):
            try:
                cutoff_height = Z[-k, 2]
                ax.axhline(y=cutoff_height, color=colors[i], linestyle='--', 
                          alpha=0.7, label=f'k={k}')
            except:
                continue
    
    ax.set_title(f'Dendrogram - {linkage_method} linkage ({metric} distance)')
    ax.set_xlabel('Data Points (Clustered)')
    ax.set_ylabel('Merge Distance')
    ax.legend()
    
    return fig

# Add CSS for better styling
st.markdown("""
<style>
.report-container {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.metric-card {
    background-color: white;
    padding: 15px;
    margin: 10px 0;
    border-radius: 5px;
    border-left: 4px solid #4CAF50;
}
</style>
""", unsafe_allow_html=True)