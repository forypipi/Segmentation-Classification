import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        """
        :param dim: patch's dim after each attention block
        :param fn: layers need adding LayerNorm
        Wrap a LayerNorm before attention block or fc layer
        """
        super().__init__()
        print(dim)
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), 
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        """
        :param dim: patch's dim before(same as after) each attention block
        :param heads: head number, 1 for single head
        :param dim_head: each head dimision for multi-head
        Wrap a LayerNorm before attention block or fc layer
        """
        super().__init__()
        inner_dim = dim_head * heads
        project_out = heads != 1 or dim_head != dim      # if single head 

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)     # map each patch vector to [q, k, v] vector (3 for qkv is same size==inner_dim)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),      # for multi-head projection
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()     # False: single head and qk dim == dim

    def forward(self, x):   # 513 * 1024
        qkv = self.to_qkv(x)            # (b, n(513), dim) -> (b, n(513), dim*3)
        qkv = qkv.chunk(3, dim=-1)   # chunk: split patch vector (b, n(513), dim*3) ---> 3 * (b, n, dim); qkv: tuple, 3 element, each element is (b, n, dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)          # q, k, v   (b, h(16), n(513), dim_head(1024 / 16 = 64))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        """
        :param dim: patch's dim
        :param depth: number of transformer block
        :param heads: 
        :param dim_head: 
        :param mlp_dim: mlp layer dim inside transformer block
        :param dropout: transformer block drop rate
        """
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, hidden_dim=mlp_dim, dropout=dropout))
            ]))
    
    def forward(self, x):
        for attn, ff in self.layers:
            x += attn(x)
            x += ff(x)
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=1, dim_head=64, dropout=0., emb_dropout=0.):
        """
        :param image_size: resolution of input image
        :param patch_size: resolution of each patch
        :param num_classes: classification number
        :param dim: output dim of fc layer after input patch embedding, each patchs' dim after attention blocks 
        :param depth: number of transformer block
        :param heads: 
        :param pool: whether use cls head or avgpool for classification
        :param mlp_dim: mlp layer dim inside transformer block
        :param channels: image channel
        :param dim_head: 
        :param dropout: transformer drop rate
        :param emb_dropout: embedding drop rate

        """
        super().__init__()
        image_height, image_width, image_depth = pair(image_size)
        patch_height, patch_width, patch_depth = pair(patch_size)

        assert  image_height%patch_height==0 and image_width%patch_width==0

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (image_depth // patch_depth)
        patch_dim = channels * patch_height * patch_width * patch_depth
        assert pool in {'cls', 'mean'}

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c)', p1=patch_height, p2=patch_width, p3=patch_depth),
            nn.Linear(patch_dim, dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))   # nn.Parameter: make tensor trainable
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))					
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool        # pooling method
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)        # b c (h p1) (w p2) (d p3) -> b (h w d) (p1 p2 p3 c) -> b (h w d) dim
        b, n, _ = x.shape           # b: batchSize, n: number of patch, _: number of pixel within 1 patch (after fc layer, is super-param dim)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # repeat cls_token to all batch: (1, 1, dim) -> cls_tokens: (batchSize, 1, dim)  
        x = torch.cat((cls_tokens, x), dim=1)               # concat cls_token to patch token      (b, hwd+1, dim)
        x += self.pos_embedding                  # 加位置嵌入（直接加）      (b, hwd+1, dim)
        # x += self.pos_embedding[:, :(n+1)]                  # 加位置嵌入（直接加）      (b, hwd+1, dim)
        x = self.dropout(x)

        x = self.transformer(x)                                                 # (b, 65, dim)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]                   # (b, dim)

        x = self.to_latent(x)                                                   # Identity (b, dim)

        return self.mlp_head(x)                                                 # LayerNorm+fc (b, num_classes)


if __name__ == "__main__":
    model_vit = ViT(
        image_size = 128,
        patch_size = 16,
        channels=1,
        num_classes = 4,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    img = torch.randn(16, 1, 128, 128, 128)

    preds = model_vit(img) 

    print(preds.shape)  # (16, 1000)