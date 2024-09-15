# based heavily on the implementation here: https://github.com/antofuller/CROMA/blob/main/pretrain_croma.py
# this is mostly in the 

import torch
import numpy as np
import math
import itertools
import torch.nn.functional as F
from torch import distributed as dist
from torch import nn, einsum
from einops import rearrange, pack
from PIL import Image

class ScaleAlibi(nn.Module):
    def __init__(
            self,
            patch_size=8,
            encoder_dim=768,
            encoder_layers=12,
            attention_heads=16,
            decoder_dim=512,
            decoder_layers=1,
            total_channels=14,
            num_patches=225,
        ):
        super().__init__()

        self.encoder_dim     = encoder_dim
        self.encoder_layers  = encoder_layers
        self.decoder_dim     = decoder_dim
        self.decoder_layers  = decoder_layers
        self.attention_heads = attention_heads
        self.num_patches     = num_patches
        self.patch_size      = patch_size
        self.total_channels  = total_channels

        self.radar_encoder = ViT(
            num_patches=self.num_patches,
            dim=self.encoder_dim,
            layers=int(self.encoder_layers/2),
            attention_heads=self.attention_heads,
            in_channels=2, # 2 radar channels
            patch_size=self.patch_size,
        )
        self.optical_encoder = ViT(
            num_patches=self.num_patches,
            dim=self.encoder_dim,
            layers=self.encoder_layers,
            attention_heads=self.attention_heads,
            # in_channels=12,
            in_channels=3, # RGB lores optical
            patch_size=self.patch_size,
        )
        self.hires_encoder = ViT(
            num_patches=self.num_patches,
            dim=self.encoder_dim,
            layers=self.encoder_layers,
            attention_heads=self.attention_heads,
            # in_channels=12,
            in_channels=3, # RGB hires optical
            patch_size=self.patch_size,
        )

        self.cross_encoder = BaseTransformerCrossAttn(
            dim=self.encoder_dim,
            layers=int(self.encoder_layers/2),
            attention_heads=self.attention_heads,
        )

        self.GAP_FFN_radar = nn.Sequential(
            nn.LayerNorm(self.encoder_dim),
            nn.Linear(self.encoder_dim, int(4*self.encoder_dim)),
            nn.GELU(),
            nn.Linear(int(4*self.encoder_dim), self.encoder_dim)
        )
        self.GAP_FFN_optical = nn.Sequential(
            nn.LayerNorm(self.encoder_dim),
            nn.Linear(self.encoder_dim, int(4*self.encoder_dim)),
            nn.GELU(),
            nn.Linear(int(4*self.encoder_dim), self.encoder_dim)
        )
        self.GAP_FFN_hires = nn.Sequential(
            nn.LayerNorm(self.encoder_dim),
            nn.Linear(self.encoder_dim, int(4*self.encoder_dim)),
            nn.GELU(),
            nn.Linear(int(4*self.encoder_dim), self.encoder_dim)
        )

        self.decoder = DecoderMAE(
            num_patches=self.num_patches,
            encoder_dim=self.encoder_dim,
            decoder_dim=self.decoder_dim,
            decoder_layers=self.decoder_layers,
            attention_heads=8,
            total_channels=self.total_channels,
            patch_size=self.patch_size,
        )

        # extension point: implement scale-alibi attention
        print('generating patches...')
        self.attn_bias = get_scale_alibi(
            attention_heads=self.attention_heads,
            num_patches=self.num_patches
        )
        print('attn bias', self.attn_bias.shape)
        print('generating hires patches...')
        self.attn_bias_hires = get_scale_alibi(
            attention_heads=self.attention_heads,
            num_patches=self.num_patches * 4,
            scale_multiplier=0.5
        )
        print('attn bias hires', self.attn_bias_hires.shape)
        print('attn bias hires (strided)', self.attn_bias_hires[:,:,:,::4].shape)
        print('it is done')

        self.global_contrast_loss = ContrastLossInput(
            projection_input=self.encoder_dim,
            projection_output=self.encoder_dim,
        )


    def forward(
            self,
            radar_imgs,
            optical_imgs,
            hires_imgs,
            radar_mask_info,
            optical_mask_info,
            hires_mask_info,
            rank,
            world_size
        ):
        # split stacked image into optical and radar
        #radar_imgs = imgs[:, 12:, ...]
        #optical_imgs = imgs[:, :12, ...]

        # create independent random masks
        radar_masked_attn_bias = apply_mask_to_alibi(
            alibi=self.attn_bias.to(radar_imgs.device),
            ids_keep_queries=radar_mask_info['ids_keep'],
            ids_keep_keys=radar_mask_info['ids_keep'],
            batch_size=radar_imgs.shape[0],
            orig_seq_len=self.num_patches,
            masked_seq_len=radar_mask_info['len_keep'],
            attention_heads=self.attention_heads
        )

        optical_masked_attn_bias = apply_mask_to_alibi(
            alibi=self.attn_bias.to(optical_imgs.device),
            ids_keep_queries=optical_mask_info['ids_keep'],
            ids_keep_keys=optical_mask_info['ids_keep'],
            batch_size=radar_imgs.shape[0],
            orig_seq_len=self.num_patches,
            masked_seq_len=optical_mask_info['len_keep'],
            attention_heads=self.attention_heads
        )
        
        hires_masked_attn_bias = apply_mask_to_alibi(
            alibi=self.attn_bias_hires.to(hires_imgs.device),
            ids_keep_queries=hires_mask_info['ids_keep'],
            ids_keep_keys=hires_mask_info['ids_keep'],
            batch_size=radar_imgs.shape[0],
            orig_seq_len=self.num_patches * 4,
            masked_seq_len=hires_mask_info['len_keep'],
            attention_heads=self.attention_heads
        )


        # encode each sensor independently
        radar_encodings = self.radar_encoder(
            imgs=radar_imgs,
            attn_bias=radar_masked_attn_bias,
            mask_info=radar_mask_info
        )
        optical_encodings = self.optical_encoder(
            imgs=optical_imgs,
            attn_bias=optical_masked_attn_bias,
            mask_info=optical_mask_info
        )
        hires_encodings = self.hires_encoder(
            imgs=hires_imgs,
            attn_bias=hires_masked_attn_bias,
            mask_info=hires_mask_info
        )
        
        print('optical encodings', optical_encodings.shape)
        print('radar encodings', radar_encodings.shape)
        print('hires encodings', hires_encodings.shape)

        # create unimodal representations with an FFN
        radar_GAP = self.GAP_FFN_radar(
            radar_encodings.mean(dim=1)
        )
        optical_GAP = self.GAP_FFN_optical(
            optical_encodings.mean(dim=1)
        )
        hires_GAP = self.GAP_FFN_optical( # projects down again
            hires_encodings.mean(dim=1)
        )
        
        print('optical gap encodings', optical_GAP.shape)
        print('radar gap encodings', radar_GAP.shape)
        print('hires gap encodings', hires_GAP.shape)
        


        # perform contrastive loss on unimodal representations
        contrastive_loss = self.global_contrast_loss(
            radar_features=radar_GAP,
            optical_features=optical_GAP,
            hires_features=hires_GAP,
            world_size=world_size,
            rank=rank
        )
        print(contrastive_loss)

        # two step cross attention: first do radar and optical
        # create cross attention bias and create joint multimodal encodings
        cross_attn_1_bias = apply_mask_to_alibi(
            alibi=self.attn_bias.to(radar_imgs.device),
            ids_keep_queries=radar_mask_info['ids_keep'],
            ids_keep_keys=optical_mask_info['ids_keep'],
            batch_size=radar_imgs.shape[0],
            orig_seq_len=self.num_patches,
            masked_seq_len=optical_mask_info['len_keep'],
            attention_heads=self.attention_heads
        )

        print('cross_attn_1_bias', cross_attn_1_bias.shape)
        joint_encodings_radar_lores = self.cross_encoder(
            x=radar_encodings,
            context=optical_encodings,
            self_alibi=cross_attn_1_bias,
            cross_alibi=cross_attn_1_bias
        )
        print('joint_encodings_radar_lores', joint_encodings_radar_lores.shape)
        print('hires_encodings', hires_encodings.shape)
        
        # step 2 : do radar/lores and optical
        
#         cross_attn_2_bias = apply_mask_to_alibi(
#             alibi=self.attn_bias_hires[:,:,:,::4].to(radar_imgs.device),
#             ids_keep_queries=optical_mask_info['ids_keep'],
#             ids_keep_keys=hires_mask_info['ids_keep'],
#             batch_size=radar_imgs.shape[0],
#             orig_seq_len=self.num_patches,
#             masked_seq_len=optical_mask_info['len_keep'],
#             attention_heads=self.attention_heads
#         )


#         joint_encodings = self.cross_encoder(
#             x=joint_encodings_radar_lores,
#             context=hires_encodings,
#             alibi=cross_attn_2_bias
#         )
#         print('joint_encodings', joint_encodings)

        num_patches_hires = self.attn_bias_hires.shape[-1]  # Should be 1024

        # Adjust the ALiBi tensor to match the original sequence lengths
        alibi_adjusted = self.attn_bias_hires[:, :, :, ::4].to(radar_imgs.device)

        # Now apply the modified function
        # cross_attn_2_bias = apply_mask_to_alibi_variable_lengths(
        #     alibi=alibi_adjusted,
        #     ids_keep_queries=optical_mask_info['ids_keep'],  # Shape: [batch_size, masked_seq_len_queries]
        #     ids_keep_keys=hires_mask_info['ids_keep'],       # Shape: [batch_size, masked_seq_len_keys]
        #     batch_size=radar_imgs.shape[0],
        #     orig_seq_len_queries=self.num_patches,           # e.g., 256
        #     orig_seq_len_keys=self.num_patches,   # Since we sliced with ::4
        #     masked_seq_len_queries=optical_mask_info['len_keep'],  # e.g., 128
        #     masked_seq_len_keys=hires_mask_info['len_keep'],       # e.g., 512
        #     attention_heads=self.attention_heads
        # )
        
        # # Generate ALiBi for self-attention
        # self_alibi = apply_mask_to_alibi_variable_lengths(
        #     alibi=self.attn_bias.to(device),  # Original ALiBi tensor for self-attention
        #     ids_keep_queries=optical_mask_info['ids_keep'],
        #     ids_keep_keys=optical_mask_info['ids_keep'],
        #     batch_size=radar_imgs.shape[0],
        #     orig_seq_len_queries=self.num_patches,  # e.g., 256
        #     orig_seq_len_keys=self.num_patches,     # Same as queries
        #     masked_seq_len_queries=optical_mask_info['len_keep'],  # e.g., 128
        #     masked_seq_len_keys=optical_mask_info['len_keep'],     # Same as queries
        #     attention_heads=self.attention_heads
        # )

        # Generate cross-attention ALiBi tensor
        cross_alibi = apply_mask_to_alibi_variable_lengths(
            alibi=alibi_adjusted,  # Adjusted ALiBi tensor for cross-attention
            ids_keep_queries=optical_mask_info['ids_keep'],
            ids_keep_keys=hires_mask_info['ids_keep'],
            batch_size=radar_imgs.shape[0],
            orig_seq_len_queries=self.num_patches,             # e.g., 256
            orig_seq_len_keys=self.num_patches,     # e.g., 256
            masked_seq_len_queries=optical_mask_info['len_keep'],  # e.g., 128
            masked_seq_len_keys=hires_mask_info['len_keep'],       # e.g., 512
            attention_heads=self.attention_heads
        )
        # print('cross_attn_2_bias', cross_attn_2_bias.shape)

        
        
        # Proceed with the cross-attention
        joint_encodings = self.cross_encoder(
            x=joint_encodings_radar_lores,
            context=hires_encodings,
            self_alibi=cross_attn_1_bias,
            cross_alibi=cross_alibi
        )
        print('joint_encodings', joint_encodings.shape)


        # reconstruct all sensors
        # print('radar shape', radar_imgs.shape)
        # print('optical shape', optical_imgs.shape)
        
        # our hires target needs to be the same shape as the other images, so we'll need a downsampled target (blah)
        hires_downsampled = hires_imgs[:,:,::2,::2]
        
        imgs, _ = pack([optical_imgs, radar_imgs, hires_downsampled], 'b * h w')
        # print('imgs shape', imgs.shape)
        # print('ps', ps)
        # imgs = None
        patchified_imgs = rearrange(imgs, 'b c (h i) (w j) -> b (h w) (c i j)', i=self.patch_size, j=self.patch_size)
        mae_loss = self.decoder(
            x=joint_encodings,
            mask_info_radar=radar_mask_info,
            mask_info_optical=optical_mask_info,
            mask_info_hires=hires_mask_info,
            target=patchified_imgs,
        )

        return contrastive_loss, mae_loss


class FFN(nn.Module):
    def __init__(self,
                 dim,
                 mult=4,
                 ):
        super().__init__()
        inner_dim = int(dim * mult)
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, dim)
        )
        self.input_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.input_norm(x)
        return self.net(x)


class Attention(nn.Module):
    def __init__(self,
            dim,
            attention_heads=8,
        ):
        super().__init__()
        self.attention_heads = attention_heads
        dim_head = int(dim / attention_heads)
        self.scale = dim_head ** -0.5
        self.create_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)
        self.input_norm = nn.LayerNorm(dim)

    def forward(self, x, alibi):
        x = self.input_norm(x)
        q, k, v = self.create_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.attention_heads), (q, k, v))
        attention_scores = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        if alibi is not None:
            print('attention_scores', attention_scores.shape)
            print('alibi', alibi.shape)
            attention_scores = attention_scores + alibi
        attn = attention_scores.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        return self.out(rearrange(out, 'b h n d -> b n (h d)'))


# extension point: may need to modify this for the cross scale cross attention
class CrossAttention(nn.Module):
    def __init__(
            self,
            dim,
            attention_heads=8
        ):
        super().__init__()
        self.attention_heads = attention_heads
        dim_head = int(dim / attention_heads)
        self.scale = dim_head ** -0.5
        self.create_q = nn.Linear(dim, dim, bias=False)
        self.create_k = nn.Linear(dim, dim, bias=False)
        self.create_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.input_norm = nn.LayerNorm(dim)

    def forward(self, x, context, alibi):
        x = self.input_norm(x)
        context = self.input_norm(context)
        q = self.create_q(x)
        k = self.create_k(context)
        v = self.create_v(context)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.attention_heads), (q, k, v))
        attention_scores = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attention_scores = attention_scores + alibi
        attn = attention_scores.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class BaseTransformer(nn.Module):
    def __init__(
            self,
            dim,
            layers,
            attention_heads=8,
            ff_mult=4,
            final_norm=True,
        ):
        super().__init__()
        self.final_norm = final_norm
        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, attention_heads=attention_heads),
                FFN(dim=dim, mult=ff_mult),
            ]))
        if self.final_norm:
            self.norm_out = nn.LayerNorm(dim)

    def forward(self, x, alibi=None):
        for self_attn, ffn in self.layers:
            x = self_attn(x, alibi) + x
            x = ffn(x) + x
        if self.final_norm:
            return self.norm_out(x)
        else:
            return x


class BaseTransformerCrossAttn(nn.Module):
    def __init__(
            self,
            dim,
            layers,
            attention_heads=8,
            ff_mult=4,
        ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(layers):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, attention_heads=attention_heads),
                CrossAttention(dim=dim, attention_heads=attention_heads),
                FFN(dim=dim, mult=ff_mult),
            ]))
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x, context, self_alibi=None, cross_alibi=None):
        for self_attn, cross_attn, ffn in self.layers:
            x = self_attn(x, self_alibi) + x
            x = cross_attn(x, context, cross_alibi) + x
            x = ffn(x) + x
        x = self.norm_out(x)
        return x


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# def get_alibi(attention_heads, num_patches):
#     points = list(itertools.product(range(int(math.sqrt(num_patches))), range(int(math.sqrt(num_patches)))))

#     print('points: ', len(points))
    
#     def get_slopes(n):
#         def get_slopes_power_of_2(n):
#             start = (2 ** (-2 ** -(math.log2(n) - 3)))
#             ratio = start
#             return [start * ratio ** i for i in range(n)]

#         if math.log2(n).is_integer():
#             return get_slopes_power_of_2(n)
#         else:
#             closest_power_of_2 = 2 ** math.floor(math.log2(n))
#             return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][
#                                                                :n - closest_power_of_2]

#     slopes = torch.Tensor(get_slopes(attention_heads)).unsqueeze(1)
#     idxs = []
#     for p1 in points:
#         for p2 in points:
#             dist = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
#             idxs.append(dist * slopes * -1)
#     all_bias = torch.cat(idxs, dim=1)
#     return all_bias.view(1, attention_heads, num_patches, num_patches)


def get_scale_alibi(attention_heads, num_patches, scale_multiplier=1.0):
    grid_size = int(math.sqrt(num_patches))
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(grid_size),
            torch.arange(grid_size),
            indexing='ij'
        ),
        dim=-1
    ).view(-1, 2).float()
    # coords shape: (num_patches, 2)

    # Compute pairwise distances
    diff = coords.unsqueeze(1) - coords.unsqueeze(0)  # (num_patches, num_patches, 2)
    dist = torch.norm(diff, dim=2)  # (num_patches, num_patches)
    dist *= scale_multiplier
        
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-2 ** -(math.log2(n) - 3))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + \
                   get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

    slopes = torch.Tensor(get_slopes(attention_heads)).unsqueeze(1).unsqueeze(1)  # (attention_heads, 1, 1)

    bias = dist.unsqueeze(0) * slopes * -1  # (attention_heads, num_patches, num_patches)

    return bias.unsqueeze(0)  # (1, attention_heads, num_patches, num_patches)


def get_mask(bsz, seq_len, device, mask_ratio):
    len_keep = int(seq_len * (1 - mask_ratio))
    noise = torch.rand(bsz, seq_len, device=device)  # noise in [0, 1]
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    mask = torch.ones([bsz, seq_len], device=device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    mask_info = {
        'ids_restore': ids_restore,
        'ids_keep': ids_keep,
        'len_keep': len_keep,
        'mask_for_mae': mask
    }
    return mask_info


def apply_mask_to_sequence(x, ids_keep):
    return torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))


def apply_mask_to_alibi(
        alibi,
        ids_keep_queries,
        ids_keep_keys,
        batch_size,
        orig_seq_len,
        masked_seq_len,
        attention_heads
    ):
    ids_keep_matrix = rearrange(ids_keep_queries, 'b i -> b i 1')\
                      + rearrange(ids_keep_keys, 'b i -> b 1 i') * orig_seq_len
    ids_keep_long_sequence = rearrange(ids_keep_matrix, 'b i j -> b (i j)')
    alibi_long_sequence = rearrange(alibi.repeat(batch_size, 1, 1, 1), 'b n i j -> b (i j) n')
    alibi_masked = torch.gather(alibi_long_sequence, dim=1,
                                index=ids_keep_long_sequence.unsqueeze(-1).repeat(1, 1, attention_heads))
    return rearrange(alibi_masked, 'b (i j) n -> b n i j', i=masked_seq_len, j=masked_seq_len)

def apply_mask_to_alibi_variable_lengths(
        alibi,
        ids_keep_queries,
        ids_keep_keys,
        batch_size,
        orig_seq_len_queries,
        orig_seq_len_keys,
        masked_seq_len_queries,
        masked_seq_len_keys,
        attention_heads
    ):
    # Compute the position indices for the kept query-key pairs
    ids_keep_matrix = rearrange(ids_keep_queries, 'b i -> b i 1') * orig_seq_len_keys \
                      + rearrange(ids_keep_keys, 'b j -> b 1 j')
    
    # Flatten the index matrix to align with the flattened ALiBi tensor
    ids_keep_long_sequence = rearrange(ids_keep_matrix, 'b i j -> b (i j)')
    
    # Prepare the ALiBi tensor for gathering
    alibi_long_sequence = rearrange(
        alibi.repeat(batch_size, 1, 1, 1),
        'b n i j -> b (i j) n'
    )
    
    # Gather the masked ALiBi values using the computed indices
    alibi_masked = torch.gather(
        alibi_long_sequence,
        dim=1,
        index=ids_keep_long_sequence.unsqueeze(-1).repeat(1, 1, attention_heads)
    )
    
    # Reshape back to the 4D tensor suitable for the attention mechanism
    return rearrange(
        alibi_masked,
        'b (i j) n -> b n i j',
        i=masked_seq_len_queries,
        j=masked_seq_len_keys
    )



def gather_features(features, world_size):
    gathered_image_features = [torch.zeros_like(features) for _ in range(world_size)]
    dist.all_gather(gathered_image_features, features)
    all_features = torch.cat(gathered_image_features, dim=0)
    return all_features


class ContrastLossInput(nn.Module):
    def __init__(
            self,
            projection_input=768,
            projection_output=768,
    ):
        super().__init__()
        self.radar_proj = nn.Linear(projection_input, projection_output)
        self.optical_proj = nn.Linear(projection_input, projection_output)
        self.hires_proj = nn.Linear(projection_input, projection_output) # downscales
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, radar_features, optical_features, hires_features, world_size, rank):
        # linear projection of unimodal representations
        print('contrast loss: radar before', radar_features.shape)
        radar_features = self.radar_proj(radar_features)
        print('contrast loss: radar after ', radar_features.shape)
        print('contrast loss: lores before', optical_features.shape)
        optical_features = self.optical_proj(optical_features)
        print('contrast loss: lores after ', optical_features.shape)
        print('contrast loss: hires before', hires_features.shape)
        hires_features = self.hires_proj(hires_features)
        print('contrast loss: hires after ', hires_features.shape)

        # L2 normalize
        radar_features = radar_features / radar_features.norm(dim=1, keepdim=True)
        optical_features = optical_features / optical_features.norm(dim=1, keepdim=True)
        hires_features = hires_features / hires_features.norm(dim=1, keepdim=True)

        # gather features from other GPUs
        all_radar_features   = gather_features(features=radar_features,   world_size=world_size)
        all_optical_features = gather_features(features=optical_features, world_size=world_size)
        all_hires_features   = gather_features(features=hires_features,   world_size=world_size)

        # dot product to get logits
        logit_scale = self.logit_scale.exp()
        logits_per_optical = logit_scale * optical_features @ all_radar_features.t()
        logits_per_radar   = logit_scale * radar_features @ all_optical_features.t()
        logits_per_hires   = logit_scale * hires_features @ all_hires_features.t()

        # organize labels
        num_logits = logits_per_optical.shape[0]
        labels = torch.arange(num_logits, device=radar_features.device, dtype=torch.long)
        labels = labels + num_logits * rank
        
        
        # split loss calculations up for debugging
        loss_optical = F.cross_entropy(logits_per_optical, labels)
        loss_radar   = F.cross_entropy(logits_per_radar, labels)
        loss_hires   = F.cross_entropy(logits_per_hires, labels)

        # calculate loss
        # loss = (F.cross_entropy(logits_per_optical, labels) + F.cross_entropy(logits_per_radar, labels)) / 2
        loss = (loss_optical + loss_radar + loss_hires) / 3.0
        return loss


class ViT(nn.Module):
    def __init__(
            self,
            num_patches,
            dim=768,
            layers=12,
            attention_heads=16,
            in_channels=12,
            patch_size=8,
        ):
        super().__init__()
        self.dim = dim
        self.layers = layers
        self.attention_heads = attention_heads
        self.num_patches = num_patches
        self.patch_size = patch_size
        pixels_per_patch = int(patch_size * patch_size * in_channels)
        self.linear_input = nn.Linear(pixels_per_patch, self.dim)
        self.transformer = BaseTransformer(
            dim=self.dim,
            layers=self.layers,
            attention_heads=self.attention_heads,
        )

    def forward(self, imgs, attn_bias, mask_info=None):
        x = rearrange(imgs, 'b c (h i) (w j) -> b (h w) (c i j)', i=self.patch_size, j=self.patch_size)
        x = self.linear_input(x)
        if mask_info is None:
            x = self.transformer(x, alibi=attn_bias)
            return x
        else:
            x_masked = apply_mask_to_sequence(x=x, ids_keep=mask_info['ids_keep'])
            x_masked = self.transformer(x_masked, alibi=attn_bias)
            return x_masked


class DecoderMAE(nn.Module):
    def __init__(
            self,
            num_patches,
            encoder_dim=768,
            decoder_dim=768,
            decoder_layers=12,
            attention_heads=16,
            total_channels=14,
            patch_size=8,
        ):
        super().__init__()
        self.decoder_dim = decoder_dim
        self.decoder_layers = decoder_layers
        self.attention_heads = attention_heads
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.encoder_to_decoder = nn.Linear(encoder_dim, self.decoder_dim)
        self.decoder = BaseTransformer(
            dim=self.decoder_dim,
            layers=self.decoder_layers,
            attention_heads=self.attention_heads,
        )

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.decoder_dim), requires_grad=False)
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(num_patches ** .5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        pixels_per_patch = int(patch_size * patch_size * total_channels)
        # print('pixels_per_patch', pixels_per_patch)
        # self.linear_output = nn.Linear(
        #     self.decoder_dim,
        #     pixels_per_patch
        # )
        self.linear_output = nn.Linear(
            self.decoder_dim,
            pixels_per_patch
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_dim))
        torch.nn.init.normal_(self.mask_token, std=.02)

    def forward(self, x, mask_info_radar, mask_info_optical, target):
        # prepare inputs for decoder
        x = self.encoder_to_decoder(x)
        # print('x (prepped for decoder)', x.shape)

        mask_tokens = self.mask_token.repeat(x.shape[0], mask_info_radar['ids_restore'].shape[1] + 1 - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, dim=1, index=mask_info_radar['ids_restore'].unsqueeze(-1).repeat(1, 1, x.shape[2]))
        
        

        # decode embeddings
        x = x + self.decoder_pos_embed
        x = self.linear_output(self.decoder(x))
    
        # split pixel predictions into optical and radar
        # pred = rearrange(x, 'b (h w) (c i j) -> b c (h i) (w j)', c=14, i=8, j=8, h=15, w=15)
        # pred_optical = rearrange(pred[:, :12, :, :], 'b c (h i) (w j) -> b (h w) (c i j)', c=12, i=8, j=8)
        # pred_radar = rearrange(pred[:, 12:, :, :], 'b c (h i) (w j) -> b (h w) (c i j)', c=2, i=8, j=8)
        pred = rearrange(x, 'b (h w) (c i j) -> b c (h i) (w j)', c=8, i=16, j=16, h=16, w=16)
        pred_optical = rearrange(pred[:, :3, :, :], 'b c (h i) (w j) -> b (h w) (c i j)', c=3, i=16, j=16)
        pred_radar = rearrange(pred[:, 3:, :, :], 'b c (h i) (w j) -> b (h w) (c i j)', c=2, i=16, j=16)
        
        # print('x', x.shape)
        # print('pred', pred.shape)
        # print('pred_optical', pred_optical.shape)
        # print('pred_radar', pred_radar.shape)
        # print('target', target.shape)
        
        # apply patch-wise normalization
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6) ** .5
        

        # split target into optical and radar
        # target = rearrange(target, 'b (h w) (c i j) -> b c (h i) (w j)', c=14, i=8, j=8, h=15, w=15)
        # target_optical = rearrange(target[:, :12, :, :], 'b c (h i) (w j) -> b (h w) (c i j)', c=12, i=8, j=8)
        # target_radar = rearrange(target[:, 12:, :, :], 'b c (h i) (w j) -> b (h w) (c i j)', c=2, i=8, j=8)
        
        target = rearrange(target, 'b (h w) (c i j) -> b c (h i) (w j)', c=5, i=16, j=16, h=16, w=16)
        target_optical = rearrange(target[:, :3, :, :], 'b c (h i) (w j) -> b (h w) (c i j)', c=3, i=16, j=16)
        target_radar = rearrange(target[:, 3:, :, :], 'b c (h i) (w j) -> b (h w) (c i j)', c=2, i=16, j=16)

        # calculate optical reconstruction loss
        loss_optical = (pred_optical - target_optical) ** 2
        loss_optical = loss_optical.mean(dim=-1)  # [N, L], mean loss per patch
        loss_optical = (loss_optical * mask_info_optical['mask_for_mae']).sum() / mask_info_optical['mask_for_mae'].sum()  # mean loss on removed patches

        # calculate radar reconstruction loss
        loss_radar = (pred_radar - target_radar) ** 2
        loss_radar = loss_radar.mean(dim=-1)  # [N, L], mean loss per patch
        loss_radar = (loss_radar * mask_info_radar['mask_for_mae']).sum() / mask_info_radar['mask_for_mae'].sum()  # mean loss on removed patches

        loss = loss_optical + loss_radar
        return loss

