import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import Attention, Mlp


class AdaptivePatchRefiner(nn.Module):
    def __init__(
        self,
        hidden_size,
        patch_num,
        patch_size,
        num_heads,
        merge_tau,
        mlp_ratio=1.0,
        merge_temperature=1.0,
        merge_target=-1.0,
    ):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.patch_num = patch_num
        self.patch_size = patch_size
        self.merge_tau = merge_tau
        self.merge_temperature = max(float(merge_temperature), 1e-3)
        self.merge_target = float(merge_target)

        self.patch_norm1 = nn.LayerNorm(hidden_size)
        self.patch_attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.patch_norm2 = nn.LayerNorm(hidden_size)
        self.patch_mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)

        self.merge_gate = nn.Linear(hidden_size * 3, 1)
        self.token_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.res_scale = nn.Parameter(torch.tensor([0.05]))
        self.latest_merge_rate = 0.0
        self.merge_reg = None

    def forward(self, x):
        batch_size, steps, _, hidden = x.shape
        patches = x.reshape(batch_size, steps, self.patch_num, self.patch_size, hidden)
        patch_tokens = patches.mean(dim=3)

        refined = patch_tokens.reshape(batch_size * steps, self.patch_num, hidden)
        refined = refined + self.patch_attn(self.patch_norm1(refined))
        refined = refined + self.patch_mlp(self.patch_norm2(refined))
        refined = refined.reshape(batch_size, steps, self.patch_num, hidden)

        if self.patch_num > 1:
            left = refined[:, :, :-1, :]
            right = refined[:, :, 1:, :]
            pair = torch.cat([left, right, left - right], dim=-1)
            gate = torch.sigmoid(self.merge_gate(pair) / self.merge_temperature)
            self.latest_merge_rate = gate.mean().item()
            if self.merge_target >= 0:
                self.merge_reg = (gate.mean() - self.merge_target) ** 2
            else:
                self.merge_reg = None
        else:
            gate = None
            self.latest_merge_rate = 0.0
            self.merge_reg = None

        patch_context = refined.unsqueeze(3).expand(-1, -1, -1, self.patch_size, -1)
        token_gate = torch.sigmoid(self.token_gate(torch.cat([patches, patch_context], dim=-1)))
        patches = patches + self.res_scale * token_gate * patch_context

        if gate is not None:
            neighbor_context = torch.zeros_like(refined)
            neighbor_norm = torch.ones_like(refined[..., :1])
            neighbor_context[:, :, :-1, :] += gate * right
            neighbor_context[:, :, 1:, :] += gate * left
            neighbor_norm[:, :, :-1, :] += gate
            neighbor_norm[:, :, 1:, :] += gate
            neighbor_context = (neighbor_context / neighbor_norm).unsqueeze(3).expand(-1, -1, -1, self.patch_size, -1)
            patches = patches + self.res_scale * 0.5 * neighbor_context

        return patches.reshape(batch_size, steps, -1, hidden)


class BiasAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0, bias_scale=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.bias_scale = nn.Parameter(torch.tensor(float(bias_scale)))

    def forward(self, x, attn_bias=None):
        batch_size, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            attn = attn + self.bias_scale * attn_bias.to(dtype=attn.dtype, device=attn.device).unsqueeze(1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowAttBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        num,
        size,
        merge_tau,
        mlp_ratio=4.0,
        spatial_attn_bias=None,
        graph_bias_scale=0.0,
        merge_temperature=1.0,
        merge_target=-1.0,
    ):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num, self.size = num, size

        self.nnorm1 = nn.LayerNorm(hidden_size)
        self.nattn = BiasAttention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1)
        self.nnorm2 = nn.LayerNorm(hidden_size)
        self.nmlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)

        self.snorm1 = nn.LayerNorm(hidden_size)
        self.sattn = BiasAttention(hidden_size, num_heads=num_heads, qkv_bias=True, attn_drop=0.1, proj_drop=0.1, bias_scale=graph_bias_scale)
        self.snorm2 = nn.LayerNorm(hidden_size)
        self.smlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.1)

        if spatial_attn_bias is None:
            self.register_buffer("spatial_attn_bias", None, persistent=False)
        else:
            self.register_buffer("spatial_attn_bias", spatial_attn_bias.float(), persistent=False)
        self.adaptive_refiner = AdaptivePatchRefiner(
            hidden_size,
            num,
            size,
            num_heads,
            merge_tau,
            mlp_ratio=1.0,
            merge_temperature=merge_temperature,
            merge_target=merge_target,
        )
        self.latest_merge_rate = 0.0

    def forward(self, x):
        batch_size, steps, tokens, hidden = x.shape
        patch_num, patch_size = self.num, self.size
        assert patch_num * patch_size == tokens
        x = x.reshape(batch_size, steps, patch_num, patch_size, hidden)

        qkv = self.snorm1(x.reshape(batch_size * steps * patch_num, patch_size, hidden))
        spatial_bias = None
        if self.spatial_attn_bias is not None:
            spatial_bias = self.spatial_attn_bias.repeat(batch_size * steps, 1, 1)
        x = x + self.sattn(qkv, spatial_bias).reshape(batch_size, steps, patch_num, patch_size, hidden)
        x = x + self.smlp(self.snorm2(x))

        qkv = self.nnorm1(x.transpose(2, 3).reshape(batch_size * steps * patch_size, patch_num, hidden))
        x = x + self.nattn(qkv).reshape(batch_size, steps, patch_size, patch_num, hidden).transpose(2, 3)
        x = x + self.nmlp(self.nnorm2(x))

        x = self.adaptive_refiner(x.reshape(batch_size, steps, tokens, hidden))
        self.latest_merge_rate = self.adaptive_refiner.latest_merge_rate
        return x


class TemporalMixingBlock(nn.Module):
    def __init__(self, hidden_size, kernel_size=3, mlp_ratio=1.0, dropout=0.1):
        super().__init__()
        hidden_channels = int(hidden_size * mlp_ratio)
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(hidden_size)
        self.temporal_conv = nn.Conv2d(
            hidden_size,
            hidden_size,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            groups=hidden_size,
            bias=True,
        )
        self.gate_conv = nn.Conv2d(
            hidden_size,
            hidden_size,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            groups=hidden_size,
            bias=True,
        )
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_channels, hidden_size, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.res_scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        if x.shape[1] <= 1:
            return x

        residual = x
        y = self.norm(x).permute(0, 3, 1, 2)
        mixed = self.temporal_conv(y)
        gate = torch.sigmoid(self.gate_conv(y))
        mixed = self.channel_mlp(mixed * gate)
        return residual + self.res_scale * mixed.permute(0, 2, 3, 1)


class PatchSTG(nn.Module):
    def __init__(
        self,
        tem_patchsize,
        tem_patchnum,
        node_num,
        spa_patchsize,
        spa_patchnum,
        tod,
        dow,
        layers,
        factors,
        input_dims,
        node_dims,
        tod_dims,
        dow_dims,
        ori_parts_idx,
        reo_parts_idx,
        reo_all_idx,
        merge_tau=0.5,
        output_len=None,
        temporal_mixer=False,
        temporal_kernel=3,
        spatial_attn_bias=None,
        graph_bias_scale=0.0,
        merge_temperature=1.0,
        merge_target=-1.0,
    ):
        super(PatchSTG, self).__init__()
        self.node_num = node_num
        self.ori_parts_idx, self.reo_parts_idx = ori_parts_idx, reo_parts_idx
        self.reo_all_idx = reo_all_idx
        self.tod, self.dow = tod, dow
        self.output_len = output_len or tem_patchsize * tem_patchnum
        self.tem_patchnum = tem_patchnum
        self.temporal_mixer_enabled = temporal_mixer

        dims = input_dims + tod_dims + dow_dims + node_dims
        self.dims = dims

        self.input_st_fc = nn.Conv2d(in_channels=3, out_channels=input_dims, kernel_size=(1, tem_patchsize), stride=(1, tem_patchsize), bias=True)
        self.node_emb = nn.Parameter(torch.empty(node_num, node_dims))
        nn.init.xavier_uniform_(self.node_emb)
        self.time_in_day_emb = nn.Parameter(torch.empty(tod, tod_dims))
        nn.init.xavier_uniform_(self.time_in_day_emb)
        self.day_in_week_emb = nn.Parameter(torch.empty(dow, dow_dims))
        nn.init.xavier_uniform_(self.day_in_week_emb)

        self.temporal_mixer = TemporalMixingBlock(dims, kernel_size=temporal_kernel, mlp_ratio=1.0) if temporal_mixer else nn.Identity()

        self.spa_encoder = nn.ModuleList([
            WindowAttBlock(
                dims,
                1,
                spa_patchnum // factors,
                spa_patchsize * factors,
                merge_tau,
                mlp_ratio=1,
                spatial_attn_bias=spatial_attn_bias,
                graph_bias_scale=graph_bias_scale,
                merge_temperature=merge_temperature,
                merge_target=merge_target,
            ) for _ in range(layers)
        ])

        self.regression_conv = nn.Conv2d(in_channels=tem_patchnum * dims, out_channels=self.output_len, kernel_size=(1, 1), bias=True)
        self.future_time_proj = nn.Sequential(
            nn.Linear(tod_dims + dow_dims, dims),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dims, dims),
        )
        self.future_query_norm = nn.LayerNorm(dims)
        self.future_context_norm = nn.LayerNorm(dims)
        self.future_context_proj = nn.Linear(dims, dims)
        self.future_node_bias = nn.Sequential(
            nn.LayerNorm(dims),
            nn.Linear(dims, 1),
        )
        self.future_res_scale = nn.Parameter(torch.tensor(0.05))
        self.latest_merge_rates = []

    def merge_regularization(self):
        regs = [
            block.adaptive_refiner.merge_reg
            for block in self.spa_encoder
            if block.adaptive_refiner.merge_reg is not None
        ]
        if not regs:
            return torch.zeros((), device=self.node_emb.device)
        return torch.stack(regs).mean()

    def forward(self, x, te, y_te=None):
        embeded_x = self.temporal_mixer(self.embedding(x, te))
        rex = embeded_x[:, :, self.reo_all_idx, :]

        merge_rates = []
        for block in self.spa_encoder:
            rex = block(rex)
            merge_rates.append(block.latest_merge_rate)
        self.latest_merge_rates = merge_rates

        orginal = torch.zeros(rex.shape[0], rex.shape[1], self.node_num, rex.shape[-1], device=x.device)
        orginal[:, :, self.ori_parts_idx, :] = rex[:, :, self.reo_parts_idx, :]

        pred_y = self.regression_conv(orginal.transpose(2, 3).reshape(orginal.shape[0], -1, orginal.shape[-2], 1))
        if y_te is not None:
            pred_y = pred_y + self.future_res_scale * self.future_decoder(orginal, y_te)
        return pred_y

    def future_decoder(self, encoded, y_te):
        batch_size, pred_steps, _, _ = y_te.shape
        context = encoded.mean(dim=1)

        time_index = y_te[:, :, 0, 0].long().to(encoded.device).clamp(0, self.tod - 1)
        day_index = y_te[:, :, 0, 1].long().to(encoded.device).clamp(0, self.dow - 1)
        future_time = torch.cat([self.time_in_day_emb[time_index], self.day_in_week_emb[day_index]], dim=-1)
        future_query = self.future_query_norm(self.future_time_proj(future_time))
        future_key = self.future_context_proj(self.future_context_norm(context))

        temporal_bias = torch.einsum("bqd,bnd->bqn", future_query, future_key) / math.sqrt(self.dims)
        node_bias = self.future_node_bias(context).transpose(1, 2)
        return (temporal_bias + node_bias).unsqueeze(-1)

    def embedding(self, x, te):
        batch_size, _, _, _ = x.shape

        x1 = torch.cat([x, (te[..., 0:1] / self.tod), (te[..., 1:2] / self.dow)], -1).float()
        input_data = self.input_st_fc(x1.transpose(1, 3)).transpose(1, 3)
        steps = input_data.shape[1]

        tod_idx = te[:, -steps:, :, 0].long().to(x.device)
        input_data = torch.cat([input_data, self.time_in_day_emb[tod_idx]], -1)

        dow_idx = te[:, -steps:, :, 1].long().to(x.device)
        input_data = torch.cat([input_data, self.day_in_week_emb[dow_idx]], -1)

        node_emb = self.node_emb.unsqueeze(0).unsqueeze(1).expand(batch_size, steps, -1, -1)
        input_data = torch.cat([input_data, node_emb], -1)
        return input_data
