import torch
import numpy as np
from IsoScore import IsoScore
from sklearn.decomposition import PCA

def compute_isoscore_cosine_fused(embeddings: torch.Tensor, 
                                  pca_dims=[32, 64, 128], 
                                  top_k=5):
    """
    Compute IsoScore and cosine similarity after PCA reduction,
    then select top-k embeddings and fuse them by averaging.

    Args:
        embeddings: torch.Tensor of shape [num_models, L, D]
        pca_dims: list of target dimensions for PCA
        top_k: number of best embeddings to select

    Returns:
        fused_embedding: torch.Tensor of shape [L, D] (averaged result)
        selected_embeddings: torch.Tensor of shape [top_k, L, D]
        avg_scores: numpy array of average scores for all models
    """
    num_models, L, D = embeddings.shape
    scores = {dim: [] for dim in pca_dims}
    
    # Convert to numpy for PCA and IsoScore
    emb_np = embeddings.detach().cpu().numpy()

    for dim in pca_dims:
        pca = PCA(n_components=dim)
        for i in range(num_models):
            X = emb_np[i]  # shape [L, D]
            X_red = pca.fit_transform(X)  # [L, dim]

            # IsoScore
            iso_score = IsoScore(X_red).compute()

            # Cosine similarity (mean pairwise cosine)
            X_norm = X_red / (np.linalg.norm(X_red, axis=1, keepdims=True) + 1e-9)
            cos_sim = (X_norm @ X_norm.T).mean()

            # Combine score (you can change the weight)
            combined = 0.5 * iso_score + 0.5 * cos_sim
            scores[dim].append(combined)

    # Average score across different PCA dims
    avg_scores = np.mean([scores[dim] for dim in pca_dims], axis=0)

    # Select top-k models
    top_idx = np.argsort(avg_scores)[-top_k:]
    selected_embeddings = embeddings[top_idx]  # [top_k, L, D]

    # Fuse by averaging over selected embeddings
    fused_embedding = selected_embeddings.mean(dim=0)  # [L, D]

    return fused_embedding, selected_embeddings, avg_scores
