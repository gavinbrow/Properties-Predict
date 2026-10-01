"""PyTorch Geometric models for Gen 5 multitask MP/BP prediction."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import AttentionalAggregation, BatchNorm, GINEConv, global_add_pool, global_mean_pool
from torch_geometric.nn.models import AttentiveFP

TARGETS = ("mp", "bp")


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


@dataclass(frozen=True)
class ModelConfig:
    architecture: str = "gine"
    hidden_dim: int = 320
    num_layers: int = 6
    dropout: float = 0.10
    attentivefp_timesteps: int = 3
    pooling: str = "attention"
    use_global_features: bool = True
    global_feature_dim: int = 0
    descriptor_hidden_dim: int = 192
    use_fingerprint_features: bool = True
    fingerprint_feature_dim: int = 0
    fingerprint_hidden_dim: int = 256
    fusion_hidden_dim: int = 320
    shared_trunk_hidden_dim: int = 384
    shared_trunk_layers: int = 2
    target_expert_hidden_dim: int = 352
    target_expert_layers: int = 3
    descriptor_residual_mode: str = "none"
    descriptor_residual_hidden_dim: int = 256
    max_logvar: float = 6.0
    min_logvar: float = -8.0
    # v10 Track A: a frozen chemical-language-model embedding as a fourth fusion
    # branch. The reference architecture behind the 23.06 C scaffold-split
    # result is SMILES-LM + D-MPNN + descriptors; we already had the graph and
    # descriptor branches, so this fills the one component we were missing.
    # Embeddings are precomputed offline and attached at load time, so nothing
    # here depends on `transformers` and the graph cache is never rebuilt.
    use_lm_features: bool = False
    lm_feature_dim: int = 0
    lm_hidden_dim: int = 512

    def to_dict(self) -> dict[str, int | float | str | bool]:
        return asdict(self)


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(x + self.block(x))


class TargetExpert(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, layers: int, dropout: float) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("target_expert_layers must be >= 1")
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            ResidualMLPBlock(hidden_dim, hidden_dim, dropout) for _ in range(max(layers - 1, 0))
        )

    def forward(self, x: Tensor) -> Tensor:
        h = self.proj(x)
        for block in self.blocks:
            h = block(h)
        return h


class GINEBackbone(nn.Module):
    def __init__(self, atom_dim: int, bond_dim: int, config: ModelConfig) -> None:
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.dropout = config.dropout
        self.pooling = config.pooling

        self.node_encoder = nn.Linear(atom_dim, config.hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(config.num_layers):
            conv = GINEConv(
                _mlp(config.hidden_dim, config.hidden_dim, config.hidden_dim, config.dropout),
                train_eps=True,
                edge_dim=bond_dim,
            )
            self.convs.append(conv)
            self.norms.append(BatchNorm(config.hidden_dim))
        if self.pooling == "attention":
            self.attention_pool = AttentionalAggregation(
                gate_nn=nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.GELU(),
                    nn.Linear(config.hidden_dim, 1),
                )
            )
        elif self.pooling in {"sum", "mean"}:
            self.attention_pool = None
        else:
            raise ValueError(f"Unsupported pooling mode: {self.pooling}")

    def encode_nodes(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        h = self.node_encoder(x)
        for conv, norm in zip(self.convs, self.norms, strict=True):
            residual = h
            h = conv(h, edge_index, edge_attr)
            h = norm(h)
            h = F.gelu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + residual
        return h

    def forward(self, data: Data) -> Tensor:
        node_embeddings = self.encode_nodes(data.x, data.edge_index, data.edge_attr)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=node_embeddings.device)
        if self.pooling == "sum":
            return global_add_pool(node_embeddings, batch)
        if self.pooling == "mean":
            return global_mean_pool(node_embeddings, batch)
        assert self.attention_pool is not None
        return self.attention_pool(node_embeddings, batch)


class AttentiveFPBackbone(nn.Module):
    def __init__(self, atom_dim: int, bond_dim: int, config: ModelConfig) -> None:
        super().__init__()
        self.model = AttentiveFP(
            in_channels=atom_dim,
            hidden_channels=config.hidden_dim,
            out_channels=config.hidden_dim,
            edge_dim=bond_dim,
            num_layers=config.num_layers,
            num_timesteps=config.attentivefp_timesteps,
            dropout=config.dropout,
        )

    def forward(self, data: Data) -> Tensor:
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)
        return self.model(data.x, data.edge_index, data.edge_attr, batch)


class MultiTaskMVEModel(nn.Module):
    def __init__(self, atom_dim: int, bond_dim: int, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.atom_dim = atom_dim
        self.bond_dim = bond_dim
        self.use_global_features = bool(self.config.use_global_features and self.config.global_feature_dim > 0)
        self.use_fingerprint_features = bool(
            self.config.use_fingerprint_features and self.config.fingerprint_feature_dim > 0
        )
        self.use_lm_features = bool(self.config.use_lm_features and self.config.lm_feature_dim > 0)
        self.descriptor_residual_mode = str(self.config.descriptor_residual_mode).lower()
        if self.descriptor_residual_mode not in {"none", "bp", "all"}:
            raise ValueError(f"Unsupported descriptor_residual_mode: {self.descriptor_residual_mode}")

        if self.config.architecture == "gine":
            self.backbone: nn.Module = GINEBackbone(atom_dim, bond_dim, self.config)
        elif self.config.architecture == "attentivefp":
            self.backbone = AttentiveFPBackbone(atom_dim, bond_dim, self.config)
        else:
            raise ValueError(f"Unsupported architecture: {self.config.architecture}")

        branch_dim = self.config.fusion_hidden_dim
        self.graph_projector = nn.Sequential(
            nn.Linear(self.config.hidden_dim, branch_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )

        if self.use_global_features:
            self.global_encoder = nn.Sequential(
                nn.Linear(self.config.global_feature_dim, self.config.descriptor_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.descriptor_hidden_dim, self.config.descriptor_hidden_dim),
                nn.GELU(),
            )
            self.global_projector = nn.Sequential(
                nn.Linear(self.config.descriptor_hidden_dim, branch_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
            )
            self.register_buffer("global_mean", torch.zeros(self.config.global_feature_dim), persistent=True)
            self.register_buffer("global_std", torch.ones(self.config.global_feature_dim), persistent=True)
        else:
            self.global_encoder = None
            self.global_projector = None
            self.register_buffer("global_mean", torch.zeros(0), persistent=True)
            self.register_buffer("global_std", torch.ones(0), persistent=True)

        if self.use_fingerprint_features:
            self.fingerprint_encoder = nn.Sequential(
                nn.Linear(self.config.fingerprint_feature_dim, self.config.fingerprint_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.fingerprint_hidden_dim, self.config.fingerprint_hidden_dim),
                nn.GELU(),
            )
            self.fingerprint_projector = nn.Sequential(
                nn.Linear(self.config.fingerprint_hidden_dim, branch_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
            )
        else:
            self.fingerprint_encoder = None
            self.fingerprint_projector = None

        if self.use_lm_features:
            self.lm_encoder = nn.Sequential(
                nn.Linear(self.config.lm_feature_dim, self.config.lm_hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.lm_hidden_dim, self.config.lm_hidden_dim),
                nn.GELU(),
            )
            self.lm_projector = nn.Sequential(
                nn.Linear(self.config.lm_hidden_dim, branch_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
            )
            # Transformer hidden states are not zero-centred and their scale is
            # arbitrary, so the branch gets the same standardisation treatment
            # the descriptor branch already has. Without it the first Linear
            # sees inputs an order of magnitude larger than the other branches
            # and the target gate learns to ignore whichever branch is quieter.
            self.register_buffer("lm_mean", torch.zeros(self.config.lm_feature_dim), persistent=True)
            self.register_buffer("lm_std", torch.ones(self.config.lm_feature_dim), persistent=True)
        else:
            self.lm_encoder = None
            self.lm_projector = None
            self.register_buffer("lm_mean", torch.zeros(0), persistent=True)
            self.register_buffer("lm_std", torch.ones(0), persistent=True)

        self.branch_names = ["graph"]
        if self.use_global_features:
            self.branch_names.append("global")
        if self.use_fingerprint_features:
            self.branch_names.append("fingerprint")
        if self.use_lm_features:
            self.branch_names.append("lm")
        self.branch_count = len(self.branch_names)
        shared_input_dim = branch_dim * self.branch_count

        self.shared_input = nn.Sequential(
            nn.Linear(shared_input_dim, self.config.shared_trunk_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
        )
        self.shared_blocks = nn.ModuleList(
            ResidualMLPBlock(
                self.config.shared_trunk_hidden_dim,
                self.config.shared_trunk_hidden_dim,
                self.config.dropout,
            )
            for _ in range(max(self.config.shared_trunk_layers - 1, 0))
        )
        self.target_gates = nn.ModuleDict(
            {
                target: nn.Sequential(
                    nn.Linear(shared_input_dim, branch_dim),
                    nn.GELU(),
                    nn.Linear(branch_dim, self.branch_count),
                )
                for target in TARGETS
            }
        )
        self.target_experts = nn.ModuleDict(
            {
                target: TargetExpert(
                    self.config.shared_trunk_hidden_dim + branch_dim,
                    self.config.target_expert_hidden_dim,
                    self.config.target_expert_layers,
                    self.config.dropout,
                )
                for target in TARGETS
            }
        )
        self.output_heads = nn.ModuleDict(
            {target: nn.Linear(self.config.target_expert_hidden_dim, 2) for target in TARGETS}
        )
        residual_targets = (
            TARGETS
            if self.descriptor_residual_mode == "all"
            else (("bp",) if self.descriptor_residual_mode == "bp" else ())
        )
        if self.use_global_features and residual_targets:
            self.descriptor_residual_heads = nn.ModuleDict(
                {
                    target: nn.Sequential(
                        nn.Linear(self.config.descriptor_hidden_dim, self.config.descriptor_residual_hidden_dim),
                        nn.GELU(),
                        nn.Dropout(self.config.dropout),
                        nn.Linear(self.config.descriptor_residual_hidden_dim, 1),
                    )
                    for target in residual_targets
                }
            )
        else:
            self.descriptor_residual_heads = nn.ModuleDict()

    def set_global_feature_stats(self, mean: Tensor, std: Tensor) -> None:
        if not self.use_global_features:
            return
        if mean.numel() != self.config.global_feature_dim or std.numel() != self.config.global_feature_dim:
            raise ValueError("Global feature statistics do not match model_config.global_feature_dim")
        self.global_mean.copy_(mean.detach().to(self.global_mean.device))
        self.global_std.copy_(std.detach().to(self.global_std.device).clamp_min(1e-6))

    def set_lm_feature_stats(self, mean: Tensor, std: Tensor) -> None:
        if not self.use_lm_features:
            return
        if mean.numel() != self.config.lm_feature_dim or std.numel() != self.config.lm_feature_dim:
            raise ValueError("LM feature statistics do not match model_config.lm_feature_dim")
        self.lm_mean.copy_(mean.detach().to(self.lm_mean.device))
        self.lm_std.copy_(std.detach().to(self.lm_std.device).clamp_min(1e-6))

    def _encode_branches(self, data: Data) -> tuple[list[Tensor], Tensor, Tensor | None]:
        branches: list[Tensor] = [self.graph_projector(self.backbone(data))]
        global_embedding: Tensor | None = None
        if self.use_global_features:
            raw_global = getattr(data, "global_features", None)
            if raw_global is None:
                raise ValueError("global_features missing from batched graph data")
            if raw_global.dim() == 1:
                raw_global = raw_global.view(1, -1)
            normalized_global = (raw_global - self.global_mean) / self.global_std
            assert self.global_encoder is not None and self.global_projector is not None
            global_embedding = self.global_encoder(normalized_global)
            branches.append(self.global_projector(global_embedding))
        if self.use_fingerprint_features:
            raw_fp = getattr(data, "fingerprint_features", None)
            if raw_fp is None:
                raise ValueError("fingerprint_features missing from batched graph data")
            if raw_fp.dim() == 1:
                raw_fp = raw_fp.view(1, -1)
            assert self.fingerprint_encoder is not None and self.fingerprint_projector is not None
            branches.append(self.fingerprint_projector(self.fingerprint_encoder(raw_fp)))
        if self.use_lm_features:
            raw_lm = getattr(data, "lm_features", None)
            if raw_lm is None:
                raise ValueError(
                    "lm_features missing from batched graph data; pass --lm-embeddings "
                    "so the loader attaches them"
                )
            if raw_lm.dim() == 1:
                raw_lm = raw_lm.view(1, -1)
            normalized_lm = (raw_lm - self.lm_mean) / self.lm_std
            assert self.lm_encoder is not None and self.lm_projector is not None
            branches.append(self.lm_projector(self.lm_encoder(normalized_lm)))
        shared_input = torch.cat(branches, dim=1)
        shared = self.shared_input(shared_input)
        for block in self.shared_blocks:
            shared = block(shared)
        return branches, shared, global_embedding

    def forward(self, data: Data) -> tuple[Tensor, Tensor]:
        branches, shared, global_embedding = self._encode_branches(data)
        branch_stack = torch.stack(branches, dim=1)
        flat = torch.cat(branches, dim=1)

        means: list[Tensor] = []
        logvars: list[Tensor] = []
        for target in TARGETS:
            gate_logits = self.target_gates[target](flat)
            gate_weights = torch.softmax(gate_logits, dim=1).unsqueeze(-1)
            target_mix = torch.sum(branch_stack * gate_weights, dim=1)
            expert_input = torch.cat([shared, target_mix], dim=1)
            expert_embeddings = self.target_experts[target](expert_input)
            output = self.output_heads[target](expert_embeddings)
            if target in self.descriptor_residual_heads and global_embedding is not None:
                output = output.clone()
                output[:, 0] = output[:, 0] + self.descriptor_residual_heads[target](global_embedding).squeeze(-1)
            means.append(output[:, 0])
            logvars.append(output[:, 1].clamp(self.config.min_logvar, self.config.max_logvar))
        return torch.stack(means, dim=1), torch.stack(logvars, dim=1)

    def load_backbone_state(self, state_dict: dict[str, Tensor]) -> None:
        self.backbone.load_state_dict(state_dict)

    def backbone_state(self) -> dict[str, Tensor]:
        return self.backbone.state_dict()


class MaskedAtomPretrainModel(nn.Module):
    """GINE encoder with a masked-atom reconstruction head for QM9 warm starts."""

    def __init__(self, atom_dim: int, bond_dim: int, num_atom_classes: int, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        if self.config.architecture != "gine":
            raise ValueError("QM9 masked-atom pretraining is only supported for the GINE backbone")
        self.backbone = GINEBackbone(atom_dim, bond_dim, self.config)
        self.classifier = _mlp(self.config.hidden_dim, self.config.hidden_dim, num_atom_classes, self.config.dropout)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        node_embeddings = self.backbone.encode_nodes(x, edge_index, edge_attr)
        return self.classifier(node_embeddings)


def masked_mve_loss(
    mean: Tensor,
    logvar: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    head_weights: Tensor | None = None,
    beta: float = 0.0,
) -> Tensor:
    """Gaussian negative log-likelihood, optionally in its beta-NLL form.

    Plain NLL (beta=0) scales each residual by 1/sigma^2, so a sample the model
    finds hard can be discounted simply by predicting a large variance for it.
    That under-fits the mean -- which matters here because Gen 9 shipped a
    confidence field that was ANTI-correlated with error, and because this term
    had never received a gradient at all (every Gen 7/8/9 arm passed
    --uncertainty-loss-weight 0.0).

    beta-NLL (Seitzer et al. 2022) multiplies the per-sample loss by a DETACHED
    sigma^(2*beta), cancelling that discount: beta=0 is plain NLL, beta=1 is
    plain MSE with a free-riding variance head, and beta=0.5 keeps most of the
    noise-adaptive weighting while restoring the mean fit. The weight is
    detached so it reweights samples without giving the variance head a second,
    contradictory gradient path.
    """
    observed = mask > 0
    if not observed.any():
        return mean.sum() * 0.0
    squared_error = (mean - target).pow(2)
    loss = 0.5 * (torch.exp(-logvar) * squared_error + logvar)
    if beta > 0.0:
        loss = loss * torch.exp(logvar * beta).detach()
    if head_weights is None:
        head_weights = torch.ones(loss.size(1), dtype=loss.dtype, device=loss.device)
    per_head_losses = []
    active_weights: list[Tensor] = []
    for head_index in range(loss.size(1)):
        if not observed[:, head_index].any():
            continue
        weight = head_weights[head_index]
        per_head_losses.append(loss[:, head_index][observed[:, head_index]].mean() * weight)
        active_weights.append(weight)
    return torch.stack(per_head_losses).sum() / torch.stack(active_weights).sum().clamp_min(1e-8)


def masked_mae(
    mean: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    head_weights: Tensor | None = None,
) -> Tensor:
    observed = mask > 0
    if not observed.any():
        return mean.sum() * 0.0
    absolute_error = (mean - target).abs()
    if head_weights is None:
        head_weights = torch.ones(absolute_error.size(1), dtype=absolute_error.dtype, device=absolute_error.device)
    per_head_maes = []
    active_weights: list[Tensor] = []
    for head_index in range(absolute_error.size(1)):
        if not observed[:, head_index].any():
            continue
        weight = head_weights[head_index]
        per_head_maes.append(absolute_error[:, head_index][observed[:, head_index]].mean() * weight)
        active_weights.append(weight)
    return torch.stack(per_head_maes).sum() / torch.stack(active_weights).sum().clamp_min(1e-8)


def masked_huber_loss(
    mean: Tensor,
    target: Tensor,
    mask: Tensor,
    *,
    delta: float = 30.0,
    head_weights: Tensor | None = None,
) -> Tensor:
    observed = mask > 0
    if not observed.any():
        return mean.sum() * 0.0
    absolute_error = (mean - target).abs()
    quadratic = torch.clamp(absolute_error, max=delta)
    linear = absolute_error - quadratic
    loss = 0.5 * quadratic.pow(2) + delta * linear
    if head_weights is None:
        head_weights = torch.ones(loss.size(1), dtype=loss.dtype, device=loss.device)
    per_head_losses = []
    active_weights: list[Tensor] = []
    for head_index in range(loss.size(1)):
        if not observed[:, head_index].any():
            continue
        weight = head_weights[head_index]
        per_head_losses.append(loss[:, head_index][observed[:, head_index]].mean() * weight)
        active_weights.append(weight)
    return torch.stack(per_head_losses).sum() / torch.stack(active_weights).sum().clamp_min(1e-8)


def combine_ensemble_predictions(
    member_means: list[Tensor],
    member_logvars: list[Tensor],
    *,
    combine: str = "mean",
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Aggregate ensemble members into a single prediction.

    `combine="median"` returns the per-row median across members instead of the
    arithmetic mean. The median is the MAE-optimal summary of a set of point
    estimates, so it is the right choice when the reported metric is MAE and the
    member disagreement is skewed; the mean stays the default because it is what
    Gen 7 reported and it minimises RMSE. The dispersion terms are unchanged in
    both modes so that calibration artefacts remain comparable.
    """
    mean_stack = torch.stack(member_means, dim=0)
    logvar_stack = torch.stack(member_logvars, dim=0)
    # Aleatoric variance is the mean of the members' predicted variances. Note
    # that through Gen 9 this term received no gradient at all -- every arm ran
    # --uncertainty-loss-weight 0.0 -- so it converged to roughly the constant
    # training-label variance and, being far larger than the epistemic term,
    # dominated `var_total`. That is why the shipped confidence field ranked
    # ANTI-correlated with actual error. See `uncertainty_components` for the
    # per-component split the reporting layer now uses.
    aleatoric = torch.exp(logvar_stack).mean(dim=0)
    if combine == "median":
        ensemble_mean = mean_stack.median(dim=0).values
    elif combine == "mean":
        ensemble_mean = mean_stack.mean(dim=0)
    else:
        raise ValueError(f"Unsupported ensemble combine mode: {combine}")
    epistemic = mean_stack.var(dim=0, unbiased=False)
    total_var = (aleatoric + epistemic).clamp_min(1e-8)
    return ensemble_mean, total_var, aleatoric, epistemic
