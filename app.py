import pandas as pd
import numpy as np
import plotly.express as px
import os

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

DIR_PATH = os.path.abspath(os.path.dirname(__file__))

df = pd.read_parquet(os.path.join(DIR_PATH, 'data', 'cluster_data.parquet'))

X = np.array(df['embs'].to_list())
X_tsne = TSNE(perplexity=50, n_components=3, metric='cosine', init='pca', random_state=42, n_jobs=-2).fit_transform(X)
kmeans = KMeans(n_clusters=23, n_init='auto', random_state=42).fit_predict(X_tsne)

fig = px.scatter_3d(x=X_tsne[:,0], y=X_tsne[:,1], z=X_tsne[:,2], color=kmeans, size_max=0.1, color_continuous_scale='turbo', labels=kmeans)
fig.update_layout(width=700, height=700)
fig.show()

# st.title('Text clustering')
# st.write('Text clustering is to automatically group textual documents (for example, documents in plain text, web pages, emails and etc) into clusters based on their content similarity.')
# st.plotly_chart(fig, theme=None, use_container_width=False)