import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import warnings
warnings.filterwarnings('ignore')

# --- Page Configuration and Custom CSS ---
st.set_page_config(
    page_title="Clusterisasi Harga Rumah Bandung",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with ngrok-friendly styling
st.markdown("""
<style>
/* Base font and background */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background-color: #f7fafc;
}

/* Main header styling */
.main-header {
    text-align: center;
    padding: 2rem 1rem;
    background-color: #004161;
    color: #ffffff;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    margin-bottom: 2rem;
}
.main-header h1 {
    margin: 0;
    font-size: 2.5rem;
}
.main-header p {
    margin: 0.5rem 0 0;
    font-size: 1.1rem;
    opacity: 0.9;
}

/* Metric cards */
.metric-card {
    background-color: #ffffff;
    padding: 1.25rem;
    border-radius: 0.75rem;
    border-left: 5px solid #1f77b4;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    margin-bottom: 1rem;
}
.metric-card h3 {
    color: #1f2937; /* Darker text for headings */
    font-size: 1.2rem;
    margin: 0;
}

.metric-card p {
    color: #0f172a; /* Darker text for values */
    font-size: 1.6rem;
    font-weight: 600;
    margin: 0.25rem 0 0;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
}
.metric-card .title {
    font-size: 1rem;
    color: #555;
    margin-bottom: 0.5rem;
}
.metric-card .value {
    font-size: 1.75rem;
    font-weight: 600;
    color: #1f77b4;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    margin-bottom: 1rem;
}
.stTabs [role="tab"] {
    padding: 0.5rem 1rem;
    border-radius: 0.5rem 0.5rem 0 0;
    background-color: #e2e8f0;
    color: #334155;
    font-weight: 500;
    transition: background-color 0.2s ease;
}
.stTabs [role="tab"][aria-selected="true"] {
    background-color: #ffffff;
    color: #1f77b4;
    box-shadow: 0 -2px 6px rgba(0, 0, 0, 0.05);
}

/* Ngrok info box */
.ngrok-info {
    background-color: #e6f7ff;
    padding: 1rem;
    border-radius: 0.75rem;
    border-left: 5px solid #28a745;
    margin-bottom: 1.5rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}
.ngrok-info strong {
    font-size: 1rem;
    color: #19692c;
}
</style>
""", unsafe_allow_html=True)

# Render header
st.markdown("""
<div class="main-header">
    <h1>🏠 Clusterisasi Rumah Bandung</h1>
    <p>Powered by Streamlit & Ngrok | Made by Kelompok 3</p>
</div>
""", unsafe_allow_html=True)


# --- MODIFIED SECTION START ---
# Moved load_data outside the class to make it a standalone, cacheable function.
@st.cache_data
def load_data():
    """Loads data directly from a CSV file and returns a DataFrame."""
    try:
        # Define possible paths for the dataset for flexibility
        possible_paths = [
            'sample_data/dataset_rumah.csv',
            'data/dataset_rumah.csv',
            'dataset_rumah.csv'
        ]

        df = None
        loaded_path = None
        for path in possible_paths:
            try:
                df = pd.read_csv(path)
                loaded_path = path
                break
            except FileNotFoundError:
                continue

        if df is None:
            st.error("Error: `dataset_rumah.csv` not found. Please ensure the dataset is in the correct path.")
            return None

        st.success(f"Data loaded successfully from `{loaded_path}`!")
        return df

    except Exception as e:
        st.error(f"An error occurred while loading the data: {str(e)}")
        return None
# --- MODIFIED SECTION END ---

# --- Core Application Class ---
class HouseClusteringDashboard:
    """
    This class encapsulates all the data processing, clustering,
    and prediction logic for the dashboard.
    """
    def __init__(self, dataframe):
        """Initializes the dashboard state with a given dataframe."""
        self.df = dataframe
        self.X_cluster = None
        self.X_scaled = None
        self.scaler = StandardScaler()
        self.kmeans = None
        self.pca = PCA(n_components=2)

    def data_exploration(self):
        """Displays widgets and plots for data exploration."""
        st.subheader("📊 Data Overview")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Dataset Shape</h3>
                <p>{self.df.shape[0]} × {self.df.shape[1]}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Missing Values</h3>
                <p>{self.df.isnull().sum().sum()}</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Numeric Features</h3>
                <p>{len(self.df.select_dtypes(include=np.number).columns)}</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Categorical Features</h3>
                <p>{len(self.df.select_dtypes(include='object').columns)}</p>
            </div>
            """, unsafe_allow_html=True)

        st.write("") # Add some space

        # Data preview and summary
        st.subheader("📋 Data Preview & Summary")
        with st.expander("Click to see Data Preview", expanded=False):
            st.dataframe(self.df.head(10), use_container_width=True)
        with st.expander("Click to see Statistical Summary", expanded=False):
            st.dataframe(self.df.describe(), use_container_width=True)

        # Visualizations
        st.subheader("📈 Visualizations")
        numeric_columns = self.df.select_dtypes(include=np.number).columns.tolist()

        # Data Distribution
        st.markdown("**Distribution of Numeric Features**")
        if len(numeric_columns) > 0:
            selected_cols = st.multiselect("Select columns to visualize distribution:", numeric_columns, default=numeric_columns[:min(4, len(numeric_columns))])
            if selected_cols:
                fig, axes = plt.subplots((len(selected_cols) + 1) // 2, 2, figsize=(14, 4 * ((len(selected_cols) + 1) // 2)))
                axes = axes.flatten()
                for i, col in enumerate(selected_cols):
                    sns.histplot(self.df[col], kde=True, ax=axes[i], color='skyblue')
                    axes[i].set_title(f'Distribution of {col}', fontsize=12)
                    axes[i].grid(True, alpha=0.3)
                # Hide unused subplots
                for j in range(i + 1, len(axes)):
                    fig.delaxes(axes[j])
                plt.tight_layout()
                st.pyplot(fig)

        # Correlation Heatmap
        st.markdown("**Feature Correlation Heatmap**")
        if len(numeric_columns) > 1:
            correlation_matrix = self.df[numeric_columns].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax, fmt='.2f', annot_kws={"size": 8})
            plt.title('Feature Correlation Matrix', fontsize=14)
            st.pyplot(fig)

    def prepare_clustering_features(self, selected_features):
        """Prepares and scales features for the clustering algorithm."""
        try:
            self.X_cluster = self.df[selected_features].fillna(self.df[selected_features].median())
            self.X_scaled = self.scaler.fit_transform(self.X_cluster)
            return True
        except Exception as e:
            st.error(f"Error preparing features: {str(e)}")
            return False

    def find_optimal_clusters(self, max_k=10):
        """Calculates metrics to help find the optimal number of clusters."""
        k_range = range(2, max_k + 1) # k=1 is not useful for silhouette/calinski
        inertia = []
        silhouette_scores = []
        calinski_scores = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, k in enumerate(k_range):
            status_text.text(f'Evaluating k={k}...')
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.X_scaled)
            inertia.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.X_scaled, labels))
            calinski_scores.append(calinski_harabasz_score(self.X_scaled, labels))
            progress_bar.progress((i + 1) / len(k_range))

        progress_bar.empty()
        status_text.empty()

        # Get inertia for k=1 for elbow plot
        kmeans_k1 = KMeans(n_clusters=1, random_state=42, n_init=10).fit(self.X_scaled)

        return {
            'k_range': list(range(1, max_k + 1)),
            'inertia': [kmeans_k1.inertia_] + inertia,
            'k_range_scores': list(k_range),
            'silhouette_scores': silhouette_scores,
            'calinski_scores': calinski_scores
        }

    def perform_clustering(self, n_clusters):
        """Performs K-Means clustering and returns performance metrics."""
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.df['cluster'] = self.kmeans.fit_predict(self.X_scaled)

        sil_score = silhouette_score(self.X_scaled, self.df['cluster'])
        ch_score = calinski_harabasz_score(self.X_scaled, self.df['cluster'])

        return {
            'silhouette_score': sil_score,
            'calinski_harabasz_score': ch_score,
            'cluster_counts': self.df['cluster'].value_counts().sort_index()
        }

    def predict_new_data(self, new_data):
        """Predicts the cluster for a new data point."""
        if self.kmeans is None: return None
        new_data_scaled = self.scaler.transform(new_data)
        return self.kmeans.predict(new_data_scaled)[0]

# --- Main Application UI ---
def main():
    # --- MODIFIED SECTION START ---
    # Automatically load data at the start.
    df = load_data()

    if df is None:
        st.stop() # Stop the app if data loading fails

    # Initialize the dashboard class with the loaded data.
    # Use session state to persist the dashboard object across reruns.
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = HouseClusteringDashboard(df)

    dashboard = st.session_state.dashboard
    # --- MODIFIED SECTION END ---

    # Sidebar with additional info
    st.sidebar.title("🌐 Navigasi")
    st.sidebar.markdown("---")
    st.sidebar.info("Dashboard ini dapat diakses melalui URL publik Ngrok.")

    st.sidebar.markdown("[🔗 Supervised Learning](https://supervisedlearning-2klury893s9kkjhyusenni.streamlit.app/)")
    st.sidebar.markdown("[🔗 Unsupervised Learning](https://unsupervisedlearning-bap5f5pbvjdwhqraeekmjp.streamlit.app/)")

    # Initialize session state variables if they don't exist
    if 'clustering_completed' not in st.session_state:
        st.session_state.clustering_completed = False
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = []

    # --- Tabbed Navigation ---
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "**1. Dashboard Overview**",
        "**2. Business Understanding**",
        "**3. Data Exploration**",
        "**4. Clustering Analysis**",
        "**5. Cluster Visualization**",
        "**6. New Data Prediction**"
    ])
    with tab1:
        st.header("🏠 Dashboard Overview")
        st.write("""
              Selamat datang di Dashboard Analisis dan Prediksi Harga Rumah. Dashboard ini dirancang untuk memberikan wawasan mendalam dari dataset harga rumah dan menyediakan alat prediksi yang akurat.

              **Tujuan Dashboard:**
              - **Eksplorasi Data:** Memvisualisasikan distribusi, korelasi, dan hubungan antar fitur dalam data.
              - **Evaluasi Model:** Menilai dan membandingkan performa model machine learning (XGBoost dan Random Forest).
              - **Prediksi Interaktif:** Memungkinkan pengguna untuk memasukkan spesifikasi rumah dan mendapatkan estimasi harga secara real-time.
              - **Wawasan Bisnis:** Memberikan rangkuman dan rekomendasi strategis berdasarkan analisis data.
              """)

        st.subheader("Kelompok 3")
        st.markdown("""
              - **220102005** - Aditya Zhafari Nur Itmam
              - **220102044** - Marshal Yanda Saputra
              - **220102051** - Muhamad Nur Ramdoni
              - **220102069** - Radhea Izzul Muttaqin
              """)
    with tab2:
        st.header("🌐 Business Understanding")
        st.subheader("Latar Belakang")
        st.write("""
            Pasar properti adalah sektor yang sangat dinamis dan kompetitif. Harga sebuah rumah dipengaruhi oleh berbagai faktor kompleks seperti lokasi, ukuran, fasilitas, dan kondisi pasar saat ini. Ketidakpastian dalam penentuan harga sering kali menjadi tantangan besar bagi penjual, pembeli, maupun agen real estate. Penjual berisiko menjual terlalu murah dan kehilangan potensi keuntungan, sementara pembeli berisiko membayar terlalu mahal.
            """)

        st.subheader("Tujuan Proyek")
        st.write("""
            Proyek ini bertujuan untuk membangun sebuah sistem cerdas yang dapat memprediksi harga rumah secara akurat berdasarkan fitur-fitur utamanya. Dengan memanfaatkan model machine learning, kami berupaya untuk:
            1.  **Memberikan Estimasi Harga yang Objektif:** Mengurangi subjektivitas dalam penilaian harga properti.
            2.  **Membantu Pengambilan Keputusan:** Memberikan alat bantu bagi penjual untuk menetapkan harga jual yang kompetitif dan bagi pembeli untuk melakukan penawaran yang wajar.
            3.  **Mengidentifikasi Faktor Kunci Harga:** Menganalisis dan menyajikan faktor-faktor apa saja yang paling signifikan mempengaruhi harga rumah di suatu area.
            """)

        st.subheader("Pemangku Kepentingan (Stakeholders)")
        st.write("""
            - **Penjual Properti:** Mendapatkan acuan harga yang realistis untuk properti mereka.
            - **Calon Pembeli:** Memverifikasi apakah harga yang ditawarkan untuk sebuah properti sudah wajar.
            - **Agen Real Estate:** Memberikan nasihat yang didukung data kepada klien dan mempercepat proses transaksi.
            - **Investor Properti:** Mengidentifikasi potensi investasi dan tren pasar properti.
            - **Bank dan Lembaga Keuangan:** Sebagai alat bantu dalam proses valuasi agunan untuk pengajuan kredit pemilikan rumah (KPR).
            """)
            
        st.subheader("Manfaat yang Diharapkan")
        st.write("""
            Dengan adanya dashboard ini, diharapkan proses jual-beli properti menjadi lebih transparan, efisien, dan berbasis data, sehingga memberikan keuntungan bagi semua pihak yang terlibat.
            """)
    with tab3:
        st.header("Data Exploration")
        dashboard.data_exploration()

    with tab4:
        st.header("Configure and Run Clustering")
        numeric_cols = dashboard.df.select_dtypes(include=np.number).columns.tolist()
        feature_cols = [col for col in numeric_cols if 'price' not in col.lower()]

        st.subheader("🎯 1. Select Features")
        selected_features = st.multiselect(
            "Select features for clustering:", feature_cols,
            default=st.session_state.get('selected_features', feature_cols[:min(5, len(feature_cols))])
        )
        st.session_state.selected_features = selected_features

        if len(selected_features) < 2:
            st.warning("Please select at least 2 features for clustering.")
        else:
            if dashboard.prepare_clustering_features(selected_features):
                st.success(f"Features prepared: {', '.join(selected_features)}")

                st.subheader("⚙️ 2. Find Optimal Number of Clusters")
                max_k = st.slider("Max clusters to evaluate (k):", 2, 15, 10)
                if st.button("Analyze Optimal k"):
                    with st.spinner("Running analysis... this may take a moment."):
                        results = dashboard.find_optimal_clusters(max_k)

                        # Create plots
                        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

                        # Elbow Method
                        axes[0].plot(results['k_range'], results['inertia'], 'bo-')
                        axes[0].set_title('Elbow Method (Inertia)', fontsize=14)
                        axes[0].set_xlabel('Number of Clusters (k)'); axes[0].set_ylabel('Inertia')

                        # Silhouette Score
                        best_k_sil = results['k_range_scores'][np.argmax(results['silhouette_scores'])]
                        axes[1].plot(results['k_range_scores'], results['silhouette_scores'], 'gs-')
                        axes[1].set_title('Silhouette Score', fontsize=14)
                        axes[1].set_xlabel('Number of Clusters (k)'); axes[1].set_ylabel('Silhouette Score')
                        axes[1].axvline(x=best_k_sil, color='r', linestyle='--', label=f'Best k = {best_k_sil}')
                        axes[1].legend()

                        # Calinski-Harabasz Score
                        best_k_ch = results['k_range_scores'][np.argmax(results['calinski_scores'])]
                        axes[2].plot(results['k_range_scores'], results['calinski_scores'], 'r^-')
                        axes[2].set_title('Calinski-Harabasz Score', fontsize=14)
                        axes[2].set_xlabel('Number of Clusters (k)'); axes[2].set_ylabel('CH Score')
                        axes[2].axvline(x=best_k_ch, color='b', linestyle='--', label=f'Best k = {best_k_ch}')
                        axes[2].legend()

                        plt.tight_layout()
                        st.pyplot(fig)
                        st.info(f"💡 Recommendation: Based on the scores, a good choice for k might be **{best_k_sil}** or **{best_k_ch}**.")

                st.subheader("🚀 3. Run Final Clustering")
                n_clusters = st.number_input("Select number of clusters:", 2, 15, 3)
                if st.button("Run Clustering"):
                    with st.spinner("Performing clustering..."):
                        results = dashboard.perform_clustering(n_clusters)
                        st.session_state.clustering_completed = True
                        st.success("Clustering complete! View results in the next tab.")

                        col1, col2 = st.columns(2)
                        col1.metric("Silhouette Score", f"{results['silhouette_score']:.4f}")
                        col2.metric("Calinski-Harabasz Score", f"{results['calinski_harabasz_score']:.2f}")


    with tab5:
        st.header("Visualize and Interpret Clusters")
        if not st.session_state.clustering_completed:
            st.warning("Please run the clustering analysis in the previous tab first.")
        else:
            # PCA Visualization
            st.subheader("🎨 Cluster Visualization using PCA")
            X_pca = dashboard.pca.fit_transform(dashboard.X_scaled)

            fig = px.scatter(
                x=X_pca[:, 0], y=X_pca[:, 1],
                color=dashboard.df['cluster'].astype(str),
                title="2D PCA of Clusters",
                labels={'x': f'PC1 ({dashboard.pca.explained_variance_ratio_[0]:.1%})',
                        'y': f'PC2 ({dashboard.pca.explained_variance_ratio_[1]:.1%})', 'color': 'Cluster'},
                color_discrete_sequence=px.colors.qualitative.Vivid
            )
            st.plotly_chart(fig, use_container_width=True)

            # Cluster Characteristics
            st.subheader("🔥 Cluster Characteristics")
            cluster_means = dashboard.df.groupby('cluster')[selected_features].mean()
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(cluster_means.T, annot=True, cmap='viridis', fmt='.2f', ax=ax)
            ax.set_title('Mean Feature Values per Cluster')
            st.pyplot(fig)

            # Detailed analysis per cluster
            st.subheader("📋 Detailed Cluster Analysis")
            for cluster_id in sorted(dashboard.df['cluster'].unique()):
                with st.expander(f"**Cluster {cluster_id} Details**"):
                    cluster_data = dashboard.df[dashboard.df['cluster'] == cluster_id]
                    col1, col2 = st.columns([1,2])
                    with col1:
                        st.metric("Number of Houses", f"{len(cluster_data)}")
                        if 'price' in dashboard.df.columns:
                            st.metric("Average Price", f"Rp {cluster_data['price'].mean():,.0f}")
                    with col2:
                        st.write("**Average Feature Values:**")
                        st.dataframe(cluster_data[selected_features].mean().to_frame('Average Value'), use_container_width=True)


    with tab6:
        st.header("Predict the Cluster for a New House")
        if not st.session_state.clustering_completed:
            st.warning("Please run the clustering analysis first.")
        else:
            st.subheader("Enter House Details")
            new_data = {}
            cols = st.columns(2)
            for i, feature in enumerate(st.session_state.selected_features):
                mean_val = float(dashboard.df[feature].mean())
                min_val = float(dashboard.df[feature].min())
                max_val = float(dashboard.df[feature].max())

                with cols[i % 2]:
                    if 'count' in feature:
                        new_data[feature] = st.number_input(f"{feature}", int(min_val), int(max_val), int(mean_val), 1)
                    else:
                        new_data[feature] = st.number_input(f"{feature}", min_val, max_val, mean_val)

            if st.button("Predict Cluster", type="primary"):
                try:
                    new_data_df = pd.DataFrame([new_data])
                    predicted_cluster = dashboard.predict_new_data(new_data_df)
                    st.success(f"## 🎯 This house belongs to **Cluster {predicted_cluster}**")

                    st.subheader("Comparison with Cluster Average")
                    cluster_means = dashboard.df[dashboard.df.cluster == predicted_cluster][st.session_state.selected_features].mean()

                    comparison_df = pd.DataFrame({
                        'Feature': st.session_state.selected_features,
                        'Your Input': list(new_data.values()),
                        'Cluster Average': cluster_means.values
                    }).set_index('Feature')
                    comparison_df['Difference'] = comparison_df['Your Input'] - comparison_df['Cluster Average']
                    st.dataframe(comparison_df.style.format('{:,.2f}').bar(subset=['Difference'], align='mid', color=['#d65f5f', '#5fba7d']), use_container_width=True)

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")

if __name__ == "__main__":
    main()
