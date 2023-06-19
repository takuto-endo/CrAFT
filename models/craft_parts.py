import torch
import torch.nn.functional as F
from torch import nn, einsum
import numpy as np
from einops import rearrange
from torch.autograd import Function

# original module
from utils import set_random_seed


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def ff_encodings(x,B):
    x_proj = (2. * np.pi * x.unsqueeze(-1)) @ B.t()
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class EntmaxBisectFunction(Function):
    @classmethod
    def _gp(cls, x, alpha):
        return x ** (alpha - 1)

    @classmethod
    def _gp_inv(cls, y, alpha):
        return y ** (1 / (alpha - 1))

    @classmethod
    def _p(cls, X, alpha):
        return cls._gp_inv(torch.clamp(X, min=0), alpha)

    @classmethod
    def forward(cls, ctx, X, alpha=1.5, dim=-1, n_iter=50, ensure_sum_one=True):

        if not isinstance(alpha, torch.Tensor):
            alpha = torch.tensor(alpha, dtype=X.dtype, device=X.device)
        alpha_shape = list(X.shape)
        alpha_shape[dim] = 1
        alpha = alpha.expand(*alpha_shape)

        ctx.alpha = alpha
        ctx.dim = dim
        d = X.shape[dim]
        X = X * (alpha - 1)

        max_val, _ = X.max(dim=dim, keepdim=True)
        tau_lo = max_val - cls._gp(1, alpha)
        tau_hi = max_val - cls._gp(1 / d, alpha)
        f_lo = cls._p(X - tau_lo, alpha).sum(dim) - 1
        dm = tau_hi - tau_lo

        for it in range(n_iter):
            dm /= 2
            tau_m = tau_lo + dm
            p_m = cls._p(X - tau_m, alpha)
            f_m = p_m.sum(dim) - 1
            mask = (f_m * f_lo >= 0).unsqueeze(dim)
            tau_lo = torch.where(mask, tau_m, tau_lo)

        if ensure_sum_one:
            p_m /= p_m.sum(dim=dim).unsqueeze(dim=dim)
        ctx.save_for_backward(p_m)

        return p_m
    
    @classmethod
    def backward(cls, ctx, dY):
        Y, = ctx.saved_tensors

        gppr = torch.where(Y > 0, Y ** (2 - ctx.alpha), Y.new_zeros(1))

        dX = dY * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr

        d_alpha = None
        if ctx.needs_input_grad[1]:

            # alpha gradient computation
            # d_alpha = (partial_y / partial_alpha) * dY
            # NOTE: ensure alpha is not close to 1
            # since there is an indetermination
            # batch_size, _ = dY.shape

            # shannon terms
            S = torch.where(Y > 0, Y * torch.log(Y), Y.new_zeros(1))
            # shannon entropy
            ent = S.sum(ctx.dim).unsqueeze(ctx.dim)
            Y_skewed = gppr / gppr.sum(ctx.dim).unsqueeze(ctx.dim)

            d_alpha = dY * (Y - Y_skewed) / ((ctx.alpha - 1) ** 2)
            d_alpha -= dY * (S - Y_skewed * ent) / (ctx.alpha - 1)
            d_alpha = d_alpha.sum(ctx.dim).unsqueeze(ctx.dim)

        return dX, d_alpha, None, None, None
    
def entmax_bisect(X, alpha=1.5, dim=-1, n_iter=50, ensure_sum_one=True):
    """alpha-entmax: normalizing sparse transform (a la softmax).

    Solves the optimization problem:

        max_p <x, p> - H_a(p)    s.t.    p >= 0, sum(p) == 1.

    where H_a(p) is the Tsallis alpha-entropy with custom alpha >= 1,
    using a bisection (root finding, binary search) algorithm.

    This function is differentiable with respect to both X and alpha.

    Parameters
    ----------
    X : torch.Tensor
        The input tensor.

    alpha : float or torch.Tensor
        Tensor of alpha parameters (> 1) to use. If scalar
        or python float, the same value is used for all rows, otherwise,
        it must have shape (or be expandable to)
        alpha.shape[j] == (X.shape[j] if j != dim else 1)
        A value of alpha=2 corresponds to sparsemax, and alpha=1 corresponds to
        softmax (but computing it this way is likely unstable).

    dim : int
        The dimension along which to apply alpha-entmax.

    n_iter : int
        Number of bisection iterations. For float32, 24 iterations should
        suffice for machine precision.

    ensure_sum_one : bool,
        Whether to divide the result by its sum. If false, the result might
        sum to close but not exactly 1, which might cause downstream problems.

    Returns
    -------
    P : torch tensor, same shape as X
        The projection result, such that P.sum(dim=dim) == 1 elementwise.
    """
    return EntmaxBisectFunction.apply(X, alpha, dim, n_iter, ensure_sum_one)

class EntmaxBisect(nn.Module):
    def __init__(self, alpha=1.5, dim=-1, n_iter=50):
        """alpha-entmax: normalizing sparse map (a la softmax) via bisection.

        Solves the optimization problem:

            max_p <x, p> - H_a(p)    s.t.    p >= 0, sum(p) == 1.

        where H_a(p) is the Tsallis alpha-entropy with custom alpha >= 1,
        using a bisection (root finding, binary search) algorithm.

        Parameters
        ----------
        alpha : float or torch.Tensor
            Tensor of alpha parameters (> 1) to use. If scalar
            or python float, the same value is used for all rows, otherwise,
            it must have shape (or be expandable to)
            alpha.shape[j] == (X.shape[j] if j != dim else 1)
            A value of alpha=2 corresponds to sparsemax; alpha=1 corresponds
            to softmax (but computing it this way is likely unstable).

        dim : int
            The dimension along which to apply alpha-entmax.

        n_iter : int
            Number of bisection iterations. For float32, 24 iterations should
            suffice for machine precision.

        """
        self.dim = dim
        self.n_iter = n_iter
        self.alpha = alpha
        super().__init__()

    def forward(self, X):
        return entmax_bisect(
            X, alpha=self.alpha, dim=self.dim, n_iter=self.n_iter
        )
    
# important: refuctering <<< 
class SparseAttLayer(nn.Module):#                                                                                        here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def __init__(self, nhead: int, nfield: int, nemb: int, d_k: int, nhid: int, alpha: float = 1.5, inner_dim: int = 64):
        """ Multi-Head Sparse Attention Layer """
        super(SparseAttLayer, self).__init__()
        self.sparsemax = nn.Softmax(dim=-1) if alpha == 1. else EntmaxBisect(alpha, dim=-1)
        self.sigmoid = nn.Sigmoid()

        self.scale = d_k ** -0.5
        self.nhead = nhead
        self.dim = nfield*nemb

        self.bilinear_w = nn.Parameter(torch.zeros(nhead, int(nemb/nhead), d_k), requires_grad=True) 
        self.query = nn.Parameter(torch.zeros(nhead, d_k), requires_grad=True)
        self.values = nn.Parameter(torch.zeros(nhead, nfield), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.bilinear_w, gain=1.414)
        nn.init.xavier_uniform_(self.query, gain=1.414)
        nn.init.xavier_uniform_(self.values, gain=1.414)

    def forward(self, x, return_att=False):
        h = self.nhead
        keys = rearrange(x, 'b r c (h hd) -> b h r c hd', h=h)
        keys = torch.squeeze(keys)
        att_gates = torch.einsum('h r c d, h d k, h k -> h r c',
                            keys, self.bilinear_w, self.query) * self.scale
        sparse_gates = self.sparsemax(att_gates)
        if return_att:
            return torch.einsum('hbf,hf->hbf', sparse_gates ,self.values), sparse_gates, self.values
        else:
            return torch.einsum('hbf,hf->hbf', sparse_gates ,self.values)

# important
class ExponentialNeuron(nn.Module):
    def __init__(self, nfield: int, nemb: int, nhead: int, alpha: float, nhid: int):

        print(f'cross-alpha: {alpha}')
        super().__init__()
        self.attn_layer = SparseAttLayer(nhead, nfield-1, nemb, nemb, nhid, alpha)
        self.arm_bn = nn.BatchNorm1d(nhead)
        # self.bn = nn.BatchNorm1d(nfield*nemb)
        self.nemb = nemb
        self.nhead = nhead
        self.syn_weight = nn.Parameter(torch.zeros(nhead, nemb), requires_grad=True)
        nn.init.xavier_uniform_(self.syn_weight, gain=1.414)

    def forward(self, x, return_att=False):
        original_shape = x.size()
        x = x.view(-1, original_shape[1], int(original_shape[2]/self.nemb), self.nemb)
        if return_att:
            cross_weight, local_weight, global_weight = self.attn_layer(x[:,:,1:,:], return_att=True)
        else:
            cross_weight = self.attn_layer(x[:,:,1:,:])
        x = torch.squeeze(x)
        x_cross = torch.exp(torch.einsum('bfe,hbf->bhe',x[:,1:,:], cross_weight))# ex. 512,10,32 * 4,512,10 > 512,10,32
        
        if torch.isnan(torch.max(x_cross)):
            print('error...')
            exit()
        
        # x_cross = x_cross.reshape((original_shape[1], self.nhead, -1))###
        # x_cross = self.arm_bn(x_cross)### 
        x_cross = x_cross.reshape((original_shape[0], original_shape[1], -1))
        
        x = x.view(original_shape[0], original_shape[1], -1)
        x = torch.cat((x, x_cross), dim=-1)

        if return_att:
            return x, cross_weight, local_weight, global_weight
        else:
            return x

class Residual(nn.Module):
    def __init__(self, fn, attention=False):
        super().__init__()
        self.fn = fn
        self.attention = attention

    def forward(self, x, **kwargs):
        if self.attention:
            fn_out = self.fn(x, **kwargs)
            return fn_out[0] + x, fn_out[1]
        else:
            return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# attention

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x, **kwargs):
        return self.net(x)
    
class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 16,
        dropout = 0.
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.heads
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out), attn
    
class RowColCrossTransformer(nn.Module):
    def __init__(self, num_tokens, dim, nfeats, depth, heads, dim_head, attn_dropout, ff_dropout, cross_nhead, cross_nhid, cross_alpha=1.5,style='col', random_seed=0):
        super().__init__()

        set_random_seed(random_seed)
        self.random_seed = random_seed

        self.embeds = nn.Embedding(num_tokens, dim)
        self.layers = nn.ModuleList([])
        self.mask_embed =  nn.Embedding(nfeats, dim)
        self.style = style
        self.cross_nhead = cross_nhead
        self.cross_nhid = cross_nhid
        for _ in range(depth):
            if self.style == 'colrow':
                self.layers.append(nn.ModuleList([
                    PreNorm(dim, Residual(Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout),attention=True)),
                    PreNorm(dim, Residual(FeedForward(dim, dropout = ff_dropout))),
                    PreNorm(dim*nfeats, Residual(Attention(dim*nfeats, heads = heads, dim_head = 64, dropout = attn_dropout),attention=True)),
                    PreNorm(dim*nfeats, Residual(FeedForward(dim*nfeats, dropout = ff_dropout))),
                    ExponentialNeuron(nfeats, dim, cross_nhead, cross_alpha, cross_nhid),
                    PreNorm(dim, Residual(Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout),attention=True)),# 1027/1503
                    PreNorm(dim, Residual(FeedForward(dim, dropout = ff_dropout)))# 1027/1503
                ]))
                # EXP+ implementation
            else:# 'row'
                self.layers.append(nn.ModuleList([
                    PreNorm(dim*nfeats, Residual(Attention(dim*nfeats, heads = heads, dim_head = 64, dropout = attn_dropout))),
                    PreNorm(dim*nfeats, Residual(FeedForward(dim*nfeats, dropout = ff_dropout))),
                    ExponentialNeuron(nfeats, dim, cross_nhead, cross_alpha, cross_nhid),
                ]))

    def forward(self, x, x_cont=None, mask = None, return_att=False):# model.transformer(x_categ_enc, x_cont_enc)
        if x_cont is not None:
            x = torch.cat((x,x_cont),dim=1)
        _, n, _ = x.shape
        if self.style == 'colrow':
            for attn1, ff1, attn2, ff2, mcf, attn3, ff3 in self.layers:
                x, col_att = attn1(x)
                x = ff1(x)
                x = rearrange(x, 'b n d -> 1 b (n d)')
                x, row_att = attn2(x)
                x = ff2(x)
                if return_att:
                    x, final_weight, local_weight, global_weight = mcf(x, return_att=True)
                else:
                    x = mcf(x)
                x = rearrange(x, '1 b (n d) -> b n d', n = n+(self.cross_nhead))#####  The number of variables increases with the number of cross_nheads.
                x, col_att = attn3(x)
                x = ff3(x)
        else:
            for attn1, ff1, mcf in self.layers:
                x = rearrange(x, 'b n d -> 1 b (n d)')
                x, row_att = attn1(x)
                x = ff1(x)
                x = mcf(x)
                x = rearrange(x, '1 b (n d) -> b n d', n = n+(self.cross_nhead))##### The number of variables increases with the number of cross_nheads.
        if return_att:
            return x, final_weight, local_weight, global_weight
        else:
            return x
        
# transformer
class Transformer(nn.Module):
    def __init__(self, num_tokens, dim, depth, heads, dim_head, attn_dropout, ff_dropout):
        super().__init__()
        self.layers = nn.ModuleList([])


        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Residual(Attention(dim, heads = heads, dim_head = dim_head, dropout = attn_dropout))),
                PreNorm(dim, Residual(FeedForward(dim, dropout = ff_dropout))),
            ]))

    def forward(self, x, x_cont=None):
        if x_cont is not None:
            x = torch.cat((x,x_cont),dim=1)
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x
    

#mlp
class MLP(nn.Module):
    def __init__(self, dims, act = None):
        super().__init__()
        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for ind, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = ind >= (len(dims) - 1)
            linear = nn.Linear(dim_in, dim_out)
            layers.append(linear)

            if is_last:
                continue
            if act is not None:
                layers.append(act)

        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class simple_MLP(nn.Module):
    def __init__(self,dims):
        super(simple_MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2])
        )
        
    def forward(self, x):
        if len(x.shape)==1:
            x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x

# main class

class TabAttention(nn.Module):
    def __init__(
        self,
        *,
        categories,
        num_continuous,
        dim,
        depth,
        heads,
        dim_head = 16,
        dim_out = 1,
        mlp_hidden_mults = (4, 2),
        mlp_act = None,
        num_special_tokens = 1,
        continuous_mean_std = None,
        attn_dropout = 0.,
        ff_dropout = 0.,
        lastmlp_dropout = 0.,
        cont_embeddings = 'MLP',
        scalingfactor = 10,
        attentiontype = 'col'
    ):
        super().__init__()
        assert all(map(lambda n: n > 0, categories)), 'number of each category must be positive'

        # categories related calculations
        self.num_categories = len(categories)
        self.num_unique_categories = sum(categories)

        # create category embeddings table

        self.num_special_tokens = num_special_tokens
        self.total_tokens = self.num_unique_categories + num_special_tokens

        # for automatically offsetting unique category ids to the correct position in the categories embedding table
        categories_offset = F.pad(torch.tensor(list(categories)), (1, 0), value = num_special_tokens)
        categories_offset = categories_offset.cumsum(dim = -1)[:-1]
        
        self.register_buffer('categories_offset', categories_offset)


        self.norm = nn.LayerNorm(num_continuous)
        self.num_continuous = num_continuous
        self.dim = dim
        self.cont_embeddings = cont_embeddings
        self.attentiontype = attentiontype

        if self.cont_embeddings == 'MLP':
            self.simple_MLP = nn.ModuleList([simple_MLP([1,100,self.dim]) for _ in range(self.num_continuous)])
            input_size = (dim * self.num_categories)  + (dim * num_continuous)
            nfeats = self.num_categories + num_continuous
        else:
            print('Continous features are not passed through attention')
            input_size = (dim * self.num_categories) + num_continuous
            nfeats = self.num_categories 

        # transformer
        if attentiontype == 'col':
            self.transformer = Transformer(
                num_tokens = self.total_tokens,
                dim = dim,
                depth = depth,
                heads = heads,
                dim_head = dim_head,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout
            )
        elif attentiontype in ['row','colrow'] :
            self.transformer = RowColCrossTransformer(
                num_tokens = self.total_tokens,
                dim = dim,
                nfeats= nfeats,
                depth = depth,
                heads = heads,
                dim_head = dim_head,
                attn_dropout = attn_dropout,
                ff_dropout = ff_dropout,
                style = attentiontype
            )

        l = input_size // 8
        hidden_dimensions = list(map(lambda t: l * t, mlp_hidden_mults))
        all_dimensions = [input_size, *hidden_dimensions, dim_out]
        
        self.mlp = MLP(all_dimensions, act = mlp_act)
        self.embeds = nn.Embedding(self.total_tokens, self.dim) #.to(device)

        cat_mask_offset = F.pad(torch.Tensor(self.num_categories).fill_(2).type(torch.int8), (1, 0), value = 0) 
        cat_mask_offset = cat_mask_offset.cumsum(dim = -1)[:-1]

        con_mask_offset = F.pad(torch.Tensor(self.num_continuous).fill_(2).type(torch.int8), (1, 0), value = 0) 
        con_mask_offset = con_mask_offset.cumsum(dim = -1)[:-1]

        self.register_buffer('cat_mask_offset', cat_mask_offset)
        self.register_buffer('con_mask_offset', con_mask_offset)

        self.mask_embeds_cat = nn.Embedding(self.num_categories*2, self.dim)
        self.mask_embeds_cont = nn.Embedding(self.num_continuous*2, self.dim)

    def forward(self, x_categ, x_cont,x_categ_enc,x_cont_enc):
        device = x_categ.device
        if self.attentiontype == 'justmlp':
            if x_categ.shape[-1] > 0:
                flat_categ = x_categ.flatten(1).to(device)
                x = torch.cat((flat_categ, x_cont.flatten(1).to(device)), dim = -1)
            else:
                x = x_cont.clone()
        else:
            if self.cont_embeddings == 'MLP':
                x = self.transformer(x_categ_enc,x_cont_enc.to(device))
            else:
                if x_categ.shape[-1] <= 0:
                    x = x_cont.clone()
                else: 
                    flat_categ = self.transformer(x_categ_enc).flatten(1)
                    x = torch.cat((flat_categ, x_cont), dim = -1)                    
        flat_x = x.flatten(1)
        return self.mlp(flat_x)

