
import torch
# Contrastive loss function for training
def contrastive_loss_fn(pos_similarity, neg_similarities, margin=0.5):
    """
    Compute contrastive loss with margin
    
    Args:
        pos_similarity: Similarity scores for positive pairs
        neg_similarities: List of similarity scores for negative pairs
        margin: Margin for the contrastive loss
    
    Returns:
        Contrastive loss value
    """
    # Initialize loss
    loss = torch.zeros_like(pos_similarity)
    
    # Add loss for each negative pair
    for neg_similarity in neg_similarities:
        # Compute hinge loss: max(0, margin - (pos_sim - neg_sim))
        pair_loss = torch.clamp(margin - (pos_similarity - neg_similarity), min=0)
        loss += pair_loss
    
    # Average over negative samples and batch
    return loss.mean()
