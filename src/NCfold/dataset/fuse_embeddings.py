import numpy as np
from IsoScore import IsoScore
from sklearn.decomposition import PCA


def fuse_topk_embeddings(embeddings, top_k=3):
    """
    Compute IsoScore and cosine similarity of original embeddings
    then select top-k embeddings

    Args:
        embeddings: list of embeddings
            [LxD1, LxD2]
        top_k: number of best embeddings to select

    Returns:
        fused_embedding: top_K x Lx L
        selected_idxs: [int]
            idx of selected embeddings
        avg_scores: numpy array of average scores for all models
    """
    N = len(embeddings)
    combined_scores = []
    isoscores = []
    cossim_scores = []
    for i in range(N):
        X = embeddings[i].detach().cpu().numpy()  # shape [L, D_i]
        embeddings[i] = X
        # IsoScore
        iso_score = IsoScore.IsoScore(X)

        # Cosine similarity (mean pairwise cosine)
        X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        cos_sim = (X_norm @ X_norm.T).mean()

        # Combine score (you can change the weight)
        combined = 0.5 * iso_score - 0.5 * cos_sim
        combined_scores.append(combined)
        isoscores.append(iso_score)
        cossim_scores.append(cos_sim)

    avg_scores = combined_scores
    top_idx = np.argsort(avg_scores)[-top_k:]
    fused_embedding = np.stack([embeddings[i] @ embeddings[i].T  for i in top_idx], axis=0) # top_k x L x L
    return fused_embedding, top_idx, avg_scores



def fuse_PCA_topk_embeddings(embeddings, pca_dims=[32, 64, 128], top_k=3):
    """
    Compute IsoScore and cosine similarity after PCA reduction,
    then select top-k embeddings

    Args:
        embeddings: list of embeddings
            [LxD1, LxD2]
        pca_dims: list of target dimensions for PCA
        top_k: number of best embeddings to select

    Returns:
        fused_embedding: Lx sum(pca_dims)*top_k
        selected_idxs: [int]
            idx of selected embeddings
        avg_scores: numpy array of average scores for all models
    """
    N = len(embeddings)
    scores = {dim: [] for dim in pca_dims}
    pca_embeddings = {dim: [] for dim in pca_dims}
    
    # Convert to numpy for PCA and IsoScore
    for dim in pca_dims:
        pca = PCA(n_components=dim)
        for i in range(N):
            X = embeddings[i].detach().cpu().numpy()  # shape [L, D_i]
            X_red = pca.fit_transform(X)  # [L, dim]
            pca_embeddings[dim].append(X_red)

            # IsoScore
            iso_score = IsoScore.IsoScore(X_red)

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

    fused_embedding = np.cat([pca_embeddings[i][dim] for dim in pca_dims for i in top_idx], axis=1) # L x sum(pca_dims)*top_k
    return fused_embedding, top_idx, avg_scores
