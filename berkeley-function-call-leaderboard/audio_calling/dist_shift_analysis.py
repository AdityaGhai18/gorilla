import json
import numpy as np
from typing import List, Dict, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import umap
import openai
import os
from dotenv import load_dotenv

class DistShiftAnalyzer:
    def __init__(self, json_path: str, openai_api_key: str = None):
        load_dotenv()
        self.json_path = json_path
        self.data = self._load_json(json_path)
        self.originals = [ex['original'] for ex in self.data]
        self.speechlike = [ex['transformed'] for ex in self.data]
        self.speechlike_asr = [ex['final_asr'] for ex in self.data if 'final_asr' in ex]
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set in environment or passed as argument.")
        self.client = openai.OpenAI(api_key=self.openai_api_key)
        self.embedding_model = "text-embedding-ada-002"

    def _load_json(self, path: str) -> List[Dict]:
        print(f"Loading JSON data from {path}...")
        with open(path, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} examples.")
        return data

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed a list of texts using OpenAI's text-embedding-ada-002.
        Returns a numpy array of shape (len(texts), embedding_dim).
        """
        batch_size = 100
        embeddings = []
        print(f"Embedding {len(texts)} texts...")
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            print(f"  Embedding batch {i//batch_size + 1} ({i} to {i+len(batch)-1})...")
            response = self.client.embeddings.create(
                input=batch,
                model=self.embedding_model
            )
            batch_embeds = [e.embedding for e in response.data]
            embeddings.extend(batch_embeds)
        print("  Embedding complete.")
        return np.array(embeddings)

    def _cluster_embeddings(self, embeddings: np.ndarray, k: int = 3) -> np.ndarray:
        print(f"Clustering {len(embeddings)} embeddings into {k} clusters...")
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        print("  Clustering complete.")
        return labels

    def _compute_cluster_distribution(self, labels: np.ndarray, k: int) -> np.ndarray:
        counts = np.bincount(labels, minlength=k)
        dist = counts / counts.sum()
        print(f"  Cluster distribution: {dist}")
        return dist

    def _compute_jsd(self, p: np.ndarray, q: np.ndarray) -> float:
        jsd = jensenshannon(p, q, base=2)
        print(f"  Jensen-Shannon Divergence: {jsd:.4f}")
        return jsd

    def _compute_wasserstein(self, p: np.ndarray, q: np.ndarray) -> float:
        wass = wasserstein_distance(p, q)
        print(f"  Wasserstein Distance: {wass:.4f}")
        return wass

    def _visualize_umap(self, embeddings: np.ndarray, labels: List[str], title: str = "UMAP Visualization"):
        print("Reducing embeddings to 2D with UMAP for visualization...")
        reducer = umap.UMAP(random_state=42)
        emb_2d = reducer.fit_transform(embeddings)
        
        # Calculate fixed limits based on the data
        x_min, x_max = emb_2d[:, 0].min(), emb_2d[:, 0].max()
        y_min, y_max = emb_2d[:, 1].min(), emb_2d[:, 1].max()
        
        # Add some padding to the limits
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        
        x_limits = (x_min - x_padding, x_max + x_padding)
        y_limits = (y_min - y_padding, y_max + y_padding)
        
        plt.figure(figsize=(12, 8))
        for label in set(labels):
            idx = [i for i, l in enumerate(labels) if l == label]
            plt.scatter(emb_2d[idx,0], emb_2d[idx,1], label=label, alpha=0.7, s=100)
        plt.legend(fontsize=12)
        plt.title(title, fontsize=16)
        plt.xlabel("UMAP Dimension 1", fontsize=12)
        plt.ylabel("UMAP Dimension 2", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Set dynamic limits with padding
        plt.xlim(x_limits)
        plt.ylim(y_limits)
        
        # Save plot
        output_path = f"{title.replace(' ', '_').replace('(', '').replace(')', '').replace('.', '')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  UMAP visualization saved to: {output_path}")
        print(f"  Plot limits - X: {x_limits}, Y: {y_limits}")
        print("  UMAP visualization complete.")

    def _pairwise_cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> Tuple[float, float]:
        from sklearn.metrics.pairwise import cosine_similarity
        print("Computing pairwise cosine similarity for paired samples...")
        sims = [cosine_similarity(emb1[i].reshape(1,-1), emb2[i].reshape(1,-1))[0,0] for i in range(len(emb1))]
        mean_sim, std_sim = np.mean(sims), np.std(sims)
        print(f"  Mean cosine similarity: {mean_sim:.4f}, Std: {std_sim:.4f}")
        return mean_sim, std_sim

    def analyze_clean_to_speechlike(self, k: int = 3):
        print("\n==== Analyzing Clean to Speech-like (pre-ASR) ====")
        print("Step 1: Embedding clean texts...")
        emb_clean = self._embed_texts(self.originals)
        print("Step 2: Embedding speech-like texts...")
        emb_speech = self._embed_texts(self.speechlike)
        all_emb = np.vstack([emb_clean, emb_speech])
        print("Step 3: Clustering...")
        labels = self._cluster_embeddings(all_emb, k)
        labels_clean = labels[:len(emb_clean)]
        labels_speech = labels[len(emb_clean):]
        print("Step 4: Computing cluster distributions and distances...")
        p = self._compute_cluster_distribution(labels_clean, k)
        q = self._compute_cluster_distribution(labels_speech, k)
        self._compute_jsd(p, q)
        self._compute_wasserstein(p, q)
        print("Step 5: Visualizing with UMAP...")
        self._visualize_umap(all_emb, ["clean"]*len(emb_clean) + ["speechlike"]*len(emb_speech), title="Clean vs. Speech-like (pre-ASR)")
        self._pairwise_cosine_similarity(emb_clean, emb_speech)
        print("==== Done with Clean to Speech-like analysis ====")

    def analyze_clean_to_speechlike_asr(self, k: int = 3):
        print("\n==== Analyzing Clean to Speech-like+ASR (post-ASR) ====")
        print("Step 1: Embedding clean texts...")
        emb_clean = self._embed_texts(self.originals)
        print("Step 2: Embedding speech-like+ASR texts...")
        emb_asr = self._embed_texts(self.speechlike_asr)
        all_emb = np.vstack([emb_clean, emb_asr])
        print("Step 3: Clustering...")
        labels = self._cluster_embeddings(all_emb, k)
        labels_clean = labels[:len(emb_clean)]
        labels_asr = labels[len(emb_clean):]
        print("Step 4: Computing cluster distributions and distances...")
        p = self._compute_cluster_distribution(labels_clean, k)
        q = self._compute_cluster_distribution(labels_asr, k)
        self._compute_jsd(p, q)
        self._compute_wasserstein(p, q)
        print("Step 5: Visualizing with UMAP...")
        self._visualize_umap(all_emb, ["clean"]*len(emb_clean) + ["speechlike+ASR"]*len(emb_asr), title="Clean vs. Speech-like+ASR (post-ASR)")
        self._pairwise_cosine_similarity(emb_clean, emb_asr)
        print("==== Done with Clean to Speech-like+ASR analysis ====")

if __name__ == "__main__":
    # Example usage
    # Set your OpenAI API key in the environment or pass as argument
    analyzer = DistShiftAnalyzer(
        json_path="BFCL_v3_live_simple_granular_spoken.json"
    )
    analyzer.analyze_clean_to_speechlike(k=5)
    analyzer.analyze_clean_to_speechlike_asr(k=5)




#k = 5, 30 test cases.

#clean vs pre-asr:
#   Cluster distribution: [0.2        0.1        0.13333333 0.1        0.46666667]
#   Cluster distribution: [0.16666667 0.1        0.13333333 0.13333333 0.46666667]
#   Jensen-Shannon Divergence: 0.0531
#   Wasserstein Distance: 0.0133
#   Mean cosine similarity: 0.9310, Std: 0.0328


#clean vs post-asr:
#   Cluster distribution: [0.16666667 0.03333333 0.1        0.5        0.2       ]
#   Cluster distribution: [0.03333333 0.5        0.1        0.03333333 0.33333333]
#   Jensen-Shannon Divergence: 0.6329
#   Wasserstein Distance: 0.0533
#   Mean cosine similarity: 0.8990, Std: 0.0366


