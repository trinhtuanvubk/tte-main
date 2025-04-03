import torch
import torch.nn as nn
import torch.optim as optim


class UserTowerModel(nn.Module):
    def __init__(
        self,
        gender_vocab_size: int,
        province_text_embedding_dim: int,
        job_text_embedding_dim: int = 768,
        vinid_tier_vocab_size: int = 5,
        tcbr_tier_vocab_size: int = 5,
        search_embedding_dim: int = 768,
        claim_embedding_dim: int = 768,
        embedding_dim: int = 10,
        out_project_embedding_dim: int = 256,
        final_embedding_dim: int = 128
    ):
        super(UserTowerModel, self).__init__()
        
        # Embedding layers for categorical features
        self.gender_embedding = nn.Embedding(gender_vocab_size, embedding_dim)
        self.vinid_tier_embedding = nn.Embedding(vinid_tier_vocab_size, embedding_dim)
        self.tcbr_tier_embedding = nn.Embedding(tcbr_tier_vocab_size, embedding_dim)
        
        # Age binning and embedding (continuous feature -> categorical bins)
        self.age_embedding = nn.Embedding(10, embedding_dim)  # 10 age groups (0-9, 10-19, 20-29, etc.)
        

        self.province_projection = nn.Sequential(
                        nn.Linear(province_text_embedding_dim, 512),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, out_project_embedding_dim))
        # Linear layer to project job text embedding to the same dimension
        self.job_projection = nn.Sequential(
                        nn.Linear(job_text_embedding_dim, 512),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, out_project_embedding_dim))
        
        # Linear layer to project search text embeddings to the same dimension
        self.search_projection = nn.Sequential(
                        nn.Linear(search_embedding_dim, 512),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, out_project_embedding_dim))
        
        # Linear layer to project claim history embeddings to the same dimension
        self.claim_projection = nn.Sequential(
                        nn.Linear(claim_embedding_dim, 512),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, out_project_embedding_dim))
        
        # Combine all embeddings and project to final embedding
        # 5 categorical features + job + search + claim history
        total_concat_dim = embedding_dim * 4 + out_project_embedding_dim * 4
        
        # Neural network for final embedding
        self.embedding_network = nn.Sequential(
            nn.Linear(total_concat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, final_embedding_dim),
            nn.LayerNorm(final_embedding_dim)
        )
        
    def forward(
        self, 
        gender_idx: torch.Tensor,
        age: torch.Tensor,
        vinid_tier_idx: torch.Tensor,
        tcbr_tier_idx: torch.Tensor,
        province_embedding: torch.Tensor,
        job_embedding: torch.Tensor,
        search_embedding: torch.Tensor,
        claim_embedding: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass of the Enhanced User Tower model
        
        Args:
            gender_idx: Tensor with indices for gender
            province_idx: Tensor with indices for province
            age: Tensor with continuous age values
            job_embedding: Tensor with pre-computed job text embeddings
            vinid_tier_idx: Tensor with indices for VinID tier
            tcbr_tier_idx: Tensor with indices for TCBR tier
            search_embedding: Tensor with search embeddings (combined from all searches)
            claim_embedding: Tensor with claim history embeddings
            
        Returns:
            Tensor with user embeddings
        """
        # Process categorical features
        gender_emb = self.gender_embedding(gender_idx)
        vinid_tier_emb = self.vinid_tier_embedding(vinid_tier_idx)
        tcbr_tier_emb = self.tcbr_tier_embedding(tcbr_tier_idx)
        
        # Process age: bin into decades
        age_bin = torch.clamp(age // 10, 0, 9).long()  # Bin into age groups: 0-9, 10-19, ..., 90+
        age_emb = self.age_embedding(age_bin)
        

        province_emb = self.province_projection(province_embedding)
        # Process job embeddings
        job_emb = self.job_projection(job_embedding)
        
        # Process search embeddings
        search_emb = self.search_projection(search_embedding)
        
        # Process claim history embeddings if provided
        if claim_embedding is None:
            claim_emb = torch.zeros_like(job_emb)  # Default to zeros if no claim history
        else:
            claim_emb = self.claim_projection(claim_embedding)
        
        # Concatenate all embeddings
        all_embeddings = [
            gender_emb, 
            age_emb, 
            vinid_tier_emb, 
            tcbr_tier_emb,
            province_emb, 
            job_emb, 
            search_emb,
            claim_emb
        ]
        
        concat_emb = torch.cat(all_embeddings, dim=1)
        
        # Get final embedding
        user_embedding = self.embedding_network(concat_emb)
        
        return user_embedding