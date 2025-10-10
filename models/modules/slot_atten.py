import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class SlotAttention(nn.Module):
    """Slot Attention module."""

    def __init__(self, num_iterations, num_slots, d_model,
                epsilon=1e-8):
        """Builds the Slot Attention module.
        Args:
            num_iterations: Number of iterations.
            num_slots: Number of slots.
            d_model: Hidden layer size of MLP.
            epsilon: Offset for attention coefficients before normalization.
        """
        super().__init__()
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.d_model = d_model
        self.epsilon = epsilon

        self.norm_inputs = nn.LayerNorm(d_model)
        self.norm_slots = nn.LayerNorm(d_model)
        self.norm_mlp = nn.LayerNorm(d_model)
        self.norm_out = nn.LayerNorm(d_model)

        self.slots = nn.Parameter(torch.randn(num_slots, d_model))
        nn.init.xavier_normal_(self.slots)

        # Linear maps for the attention module.
        self.project_q = nn.Linear(d_model, d_model)
        self.project_k = nn.Linear(d_model, d_model)
        self.project_v = nn.Linear(d_model, d_model)

        self.attn_holder = nn.Identity()

        # Slot update functions.
        self.mlp = MLP(d_model, d_model, d_model, 2)

    def forward(self, inputs, mask= None):
        b = inputs.shape[0]  # [bsz, n_inputs, d_model]

        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input
        k = self.project_k(inputs)  # [bsz, n_inputs, d_model]
        v = self.project_v(inputs)   # [bsz, n_inputs, d_model]

        slots = self.slots.repeat(b,1,1)

        # Multiple rounds of attention.
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)

            # Attention.
            q = self.project_q(slots)   # [bsz, num_slots, d_model]
            scale = self.d_model ** -0.5    # Normalization

            dots = torch.einsum('bid,bjd->bij', q, k) * scale  # [bsz, num_slots, n_inputs]

            max_neg_value = -torch.finfo(dots.dtype).max
            if mask is not None:
                dots.masked_fill_(mask.unsqueeze(1), max_neg_value)

            attn = dots.softmax(dim=1)
            attn = self.attn_holder(attn)
            attn = attn / (attn.sum(dim=-1, keepdim=True) + self.epsilon)  # softmax over slots
            updates = torch.einsum('bjd,bij->bid', v, attn)  # [bsz, num_slots, d_model].

            # Slot update.
            slots = slots_prev + updates
            slots = slots + self.mlp(self.norm_mlp(slots))

        return self.norm_out(slots)