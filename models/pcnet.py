import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.slot_atten import SlotAttention

from models.transformer import DualTransformer
import math

# ======= Configure the operating environment to ensure reproducibility
torch.use_deterministic_algorithms(True) 
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# ===========


class PCNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = config['dropout']
        self.vocab_size = config['vocab_size']
        self.sigma = config["sigma"]
        self.sigma_neg = config["sigma_neg"]
        self.use_negative = config['use_negative']
        self.num_props = config['num_props']
        self.max_epoch = config['max_epoch']
        self.gamma = config['gamma']

        self.frame_fc = nn.Linear(config['frames_input_size'], config['hidden_size'])
        self.word_fc = nn.Linear(config['words_input_size'], config['hidden_size'])
        self.mask_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        self.start_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        self.pred_vec = nn.Parameter(torch.zeros(config['frames_input_size']).float(), requires_grad=True)

        self.trans = DualTransformer(**config['DualTransformer'])
        self.fc_comp = nn.Linear(config['hidden_size'], self.vocab_size)

        self.fc_gauss = nn.Linear(config['hidden_size'], self.num_props*2)
        self.word_pos_encoder = SinusoidalPositionalEmbedding(config['hidden_size'], 0, 20)
        # ========= local proposal generation
        num_iteration=config['num_iteration']
        self.slot_atten = SlotAttention(num_iteration, self.num_props , config['hidden_size'])
        self.fc_sl = nn.Linear(config['hidden_size'], 2)
        
        self.merge_alpha = nn.Parameter(torch.tensor(0.)) # the learnable coefficient of local & global proposals fusion
        self.range_width = nn.Parameter(torch.tensor(0.2)) # the leanable coefficient for adjusting peak area in aggregator


    def forward(self, frames_feat, frames_len, words_id, words_feat, words_len, weights, **kwargs):
        bsz, n_frames, _ = frames_feat.shape
        pred_vec = self.pred_vec.view(1, 1, -1).expand(bsz, 1, -1)
        frames_feat = torch.cat([frames_feat, pred_vec], dim=1)
        frames_feat = F.dropout(frames_feat, self.dropout, self.training)
        frames_feat = self.frame_fc(frames_feat)
        frames_mask = _generate_mask(frames_feat, frames_len)

        words_feat[:, 0] = self.start_vec.cuda()
        words_pos = self.word_pos_encoder(words_feat)
        words_feat = F.dropout(words_feat, self.dropout, self.training)
        words_feat = self.word_fc(words_feat)
        words_mask = _generate_mask(words_feat, words_len + 1)

        # Modality Interaction 
        enc_out, h = self.trans(frames_feat, frames_mask, words_feat + words_pos, words_mask, decoding=1)
        word_cls = enc_out[:,0]

        # ===================== proposal boundary generation
        gauss_center, gauss_width = self.proposal_generator(h, bsz)

        # ====================== proposal feature aggregation
        cl_loss = self.semantic_alignment(frames_feat, word_cls, gauss_center, gauss_width, self.num_props) # sematic alignment
        props_len = n_frames//4 # downsample for effeciency, following CPL
        keep_idx = torch.linspace(0, n_frames-1, steps=props_len).long()
        frames_feat = frames_feat[:, keep_idx]
        frames_mask = frames_mask[:, keep_idx]
        props_feat = frames_feat.unsqueeze(1) \
            .expand(bsz, self.num_props, -1, -1).contiguous().view(bsz*self.num_props, props_len, -1)
        props_mask = frames_mask.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)
        gauss_weight = self.peak_aware_gauss_weight(props_len, gauss_center, gauss_width) # peak_aware Guassian weighing 
        
        # =================== Masked Query Reconstruction
        words_feat, masked_words = self._mask_words(words_feat, words_len, weights=weights)
        words_feat = words_feat + words_pos
        words_feat = words_feat[:, :-1]
        words_mask = words_mask[:, :-1]

        words_mask1 = words_mask.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)
        words_id1 = words_id.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)
        words_feat1 = words_feat.unsqueeze(1) \
            .expand(bsz, self.num_props, -1, -1).contiguous().view(bsz*self.num_props, words_mask1.size(1), -1)

        pos_weight = gauss_weight/gauss_weight.max(dim=-1, keepdim=True)[0]
        _, h, attn_weight = self.trans(props_feat, props_mask, words_feat1, words_mask1, decoding=2, gauss_weight=pos_weight, need_weight=True)
        words_logit = self.fc_comp(h)

        if self.use_negative:
            # Contructing simple negative samples for constrastive learning
            neg_1_weight, neg_2_weight = self.negative_proposal_mining(props_len, gauss_center, gauss_width, kwargs['epoch'])
            
            _, neg_h_1 = self.trans(props_feat, props_mask, words_feat1, words_mask1, decoding=2, gauss_weight=neg_1_weight)
            neg_words_logit_1 = self.fc_comp(neg_h_1)
  
            _, neg_h_2 = self.trans(props_feat, props_mask, words_feat1, words_mask1, decoding=2, gauss_weight=neg_2_weight)
            neg_words_logit_2 = self.fc_comp(neg_h_2)
            # The entire video is used as a reference proposal for contrastive learning
            _, ref_h = self.trans(frames_feat, frames_mask, words_feat, words_mask, decoding=2)
            ref_words_logit = self.fc_comp(ref_h)
        else:
            neg_words_logit_1 = None
            neg_words_logit_2 = None
            ref_words_logit = None

        return {
            'neg_words_logit_1': neg_words_logit_1,
            'neg_words_logit_2': neg_words_logit_2,
            'ref_words_logit': ref_words_logit,
            'words_logit': words_logit,
            'words_id': words_id,
            'words_mask': words_mask,
            'width': gauss_width,
            'center': gauss_center,
            'gauss_weight': gauss_weight,
            'cl_loss':cl_loss,
        }
    
    def proposal_generator(self, h, bsz):
        h_slot = self.fc_sl(self.slot_atten(h)) # torch.Size([32, 8, 256])
        reshaped_tensor = h_slot.squeeze(-1).view(bsz*self.num_props, 2)
        gauss_param = (torch.tanh(reshaped_tensor) + 1) / 2  # [-1,1] → [0,1] # fine-grained proposals
        
        gauss_center = gauss_param[:, 0]
        gauss_width = gauss_param[:, 1]

        # ============================== 
        gauss_param_raw = torch.sigmoid(self.fc_gauss(h[:,-1]).view(bsz*self.num_props, 2)) # global proposals
        # Calculate the Euclidean distance matrix
        cost_matrix = torch.cdist(gauss_param_raw, torch.stack([gauss_center, gauss_width], dim=1)).detach().cpu().numpy()
        # Compute Hungarian algorithm matches
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # Get the center point and width of the match
        matched_center = gauss_center[col_ind]
        matched_width = gauss_width[col_ind]
        # Adaptive Fusion
        gauss_center = gauss_param_raw[row_ind, 0] * torch.sigmoid(self.merge_alpha) + matched_center * (1 - torch.sigmoid(self.merge_alpha))
        gauss_width = gauss_param_raw[row_ind, 1] * torch.sigmoid(self.merge_alpha) + matched_width  * (1 - torch.sigmoid(self.merge_alpha))
        return gauss_center, gauss_width

    def semantic_alignment(self, frames_feat, word_cls, gauss_center, gauss_width, num_proposal, margin=0.5):
        bs, seq_len, dim = frames_feat.shape
        device = frames_feat.device
        num_total = bs * num_proposal
        
        # Coordinate transformation
        center_idx = (gauss_center * (seq_len-1)).long()  # (num_total,)
        width = (gauss_width * seq_len).long()            # (num_total,)
        
        # Generate Index Grid
        frames_expanded = frames_feat.repeat_interleave(num_proposal, dim=0)  # (num_total, seq_len, dim)
        time_idx = torch.arange(seq_len, device=device).expand(num_total, seq_len)  # (num_total, seq_len)
        
        # Create proposal masks
        start = torch.clamp(center_idx - width//2, 0, seq_len-1).unsqueeze(1)
        end = torch.clamp(center_idx + width//2 + 1, 0, seq_len).unsqueeze(1)
        proposal_mask = (time_idx >= start) & (time_idx < end)  # (num_total, seq_len)
        
        # Feature extraction of internal frames of proposal (query-related frames)
        proposal_feats = torch.zeros(num_total, dim, device=device)
        valid_proposal = proposal_mask.sum(dim=1) > 0
        proposal_feats[valid_proposal] = (
            frames_expanded[valid_proposal] * 
            proposal_mask[valid_proposal].unsqueeze(-1)
        ).sum(dim=1) / proposal_mask[valid_proposal].sum(dim=1, keepdim=True)
        
        # Feature extraction of external frames of proposal (query-irrelated frames)
        non_proposal_mask = ~proposal_mask
        non_proposal_feats = torch.zeros(num_total, dim, device=device)
        valid_non_proposal = non_proposal_mask.sum(dim=1) > 0
        
        # Handling the situation where there are non-proposal areas
        non_proposal_feats[valid_non_proposal] = (
            frames_expanded[valid_non_proposal] * 
            non_proposal_mask[valid_non_proposal].unsqueeze(-1)
        ).sum(dim=1) / non_proposal_mask[valid_non_proposal].sum(dim=1, keepdim=True)
        
        # Handling the special case where the entire sequence is selected
        full_mask = (proposal_mask.sum(dim=1) == seq_len)
        non_proposal_feats[full_mask] = frames_expanded[full_mask].mean(dim=1)
        
        # Calculating Similarity
        word_cls_expanded = word_cls.repeat_interleave(num_proposal, dim=0)
        sim_pos = F.cosine_similarity(proposal_feats, word_cls_expanded)
        sim_neg = F.cosine_similarity(non_proposal_feats, word_cls_expanded)
        
        # Contrastive loss
        return F.margin_ranking_loss(sim_pos, sim_neg, torch.ones_like(sim_pos), margin=margin)

    def peak_aware_gauss_weight(self, props_len, center, width):
        # Generate location grid [batch_size, props_len]
        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
        center = center.unsqueeze(-1)  # [batch_size, 1]
        width = width.unsqueeze(-1).clamp(1e-2) / self.sigma  # [batch_size, 1]

        # Calculate Gaussian weights
        w = 0.3989422804014327  # 1/sqrt(2π)
        gauss_weight = w / width * torch.exp(-(weight - center)**2 / (2 * width**2))
        gauss_weight = gauss_weight / gauss_weight.max(dim=-1, keepdim=True)[0]

        # Compute the mask of the peak region (differentiable)
        # Using sigmoid smoothing instead of Boolean masking
        delta = weight - center
        mask = torch.sigmoid(
            (self.range_width * width - torch.abs(delta)) * 1e3
        )  # [batch_size, props_len]

        # Increase the weight of the peak area to 1 (differentiable mixing)
        final_weight = gauss_weight * (1 - mask) + 1.0 * mask
        return final_weight


    def negative_proposal_mining(self, props_len, center, width, epoch):
        def Gauss(pos, w1, c):
            w1 = w1.unsqueeze(-1).clamp(1e-2) / (self.sigma_neg)
            c = c.unsqueeze(-1)
            w = 0.3989422804014327
            y1 = w/w1*torch.exp(-(pos-c)**2/(2*w1**2))
            gauss_weight = y1/y1.max(dim=-1, keepdim=True)[0]

            delta = w1 - c
            mask = torch.sigmoid(
                (self.range_width * w1 - torch.abs(delta)) * 1e3 
            )  # [batch_size, props_len]）
            final_weight = gauss_weight * (1 - mask) + 1.0 * mask
            return final_weight

        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)

        left_width = torch.clamp(center-width/2, min=0)
        left_center = left_width * min(epoch/self.max_epoch, 1)**self.gamma * 0.5
        right_width = torch.clamp(1-center-width/2, min=0)
        right_center = 1 - right_width * min(epoch/self.max_epoch, 1)**self.gamma * 0.5

        left_neg_weight = Gauss(weight, left_center, left_center)
        right_neg_weight = Gauss(weight, 1-right_center, right_center)

        return left_neg_weight, right_neg_weight

    def _mask_words(self, words_feat, words_len, weights=None):
        token = self.mask_vec.cuda().unsqueeze(0).unsqueeze(0)
        token = self.word_fc(token)

        masked_words = []
        for i, l in enumerate(words_len):
            l = int(l)
            num_masked_words = max(l // 3, 1) 
            masked_words.append(torch.zeros([words_feat.size(1)]).byte().cuda())
            if l < 1:
                continue
            p = weights[i, :l].cpu().numpy() if weights is not None else None
            choices = np.random.choice(np.arange(1, l + 1), num_masked_words, replace=False, p=p)
            masked_words[-1][choices] = 1
        
        masked_words = torch.stack(masked_words, 0).unsqueeze(-1)
        masked_words_vec = words_feat.new_zeros(*words_feat.size()) + token
        masked_words_vec = masked_words_vec.masked_fill_(masked_words == 0, 0)
        words_feat1 = words_feat.masked_fill(masked_words == 1, 0) + masked_words_vec
        return words_feat1, masked_words


def _generate_mask(x, x_len):
    if False and int(x_len.min()) == x.size(1):
        mask = None
    else:
        mask = []
        for l in x_len:
            mask.append(torch.zeros([x.size(1)]).byte().cuda())
            mask[-1][:l] = 1
        mask = torch.stack(mask, 0)
    return mask


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        import math
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, **kwargs):
        bsz, seq_len, _ = input.size()
        max_pos = seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.cuda(input.device)[:max_pos]
        return self.weights.unsqueeze(0)

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number
