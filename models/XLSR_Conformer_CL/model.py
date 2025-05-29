import torch
import torch.nn as nn

import fairseq
from conformer import ConformerBlock
from torch.nn.modules.transformer import _get_clones


class SSLModel(nn.Module):
    def __init__(self):
        super(SSLModel, self).__init__()
        cp_path = 'xlsr2_300m.pt'   # Change the pre-trained XLSR model path. 
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.out_dim = 1024
        return

    def extract_feat(self, input_data, input_len=None):
        # put the model to GPU if it not there
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()      

        # input should be in shape (batch, length)
        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data
        
        if input_len is not None:
            padding_mask = torch.zeros(input_tmp.shape).to(input_data.device)
            for i in range(len(input_len)):
                padding_mask[i, input_len[i]:] = 1
        else:
            padding_mask = None
                
        # [batch, length, dim]        
        emb = self.model(input_tmp, mask=False, padding_mask=padding_mask, features_only=True)['x']

        return emb
    
    def forward(self, input_data, input_len=None):
        return self.extract_feat(input_data, input_len)


class MyConformer(nn.Module):
    def __init__(self, emb_size=144, heads=4, ffmult=4, exp_fac=2, kernel_size=31, n_encoders=4):
        super(MyConformer, self).__init__()
        self.dim_head = emb_size // heads
        self.dim = emb_size
        self.heads = heads
        self.kernel_size = kernel_size
        self.n_encoders = n_encoders
        self.encoder_blocks = _get_clones(
            ConformerBlock(
                dim=emb_size,
                dim_head=self.dim_head,
                heads=heads,
                ff_mult=ffmult,
                conv_expansion_factor=exp_fac,
                conv_kernel_size=kernel_size
            ),
            n_encoders
        )
        self.class_token = nn.Parameter(torch.rand(1, 1, emb_size))
        self.fc5 = nn.Linear(emb_size, 2)

    def forward(self, x, x_len=None):  # x shape [bs, t, f]
        
        if x_len is not None:
            mask = torch.zeros(x.size(0), x.size(1) + 1).to(x.device).byte()
            for i in range(x.size(0)):
                mask[i, :x_len[i] + 1] = 1 # +1 for class token
            mask = mask.to(torch.bool)
        else:
            mask = None
        
        bs = x.size(0)
        class_tokens = self.class_token.expand(bs, -1, -1)  # [bs, 1, emb_size]
        x = torch.cat((class_tokens, x), dim=1)  # [bs, 1+tiempo, emb_size]
        
        for layer in self.encoder_blocks:
            x = layer(x, mask)  # [bs, 1+tiempo, emb_size]
        
        embedding = x[:, 0, :]  # [bs, emb_size]
        out = self.fc5(embedding)  # [bs, 2]
        
        return out, embedding


class XLSR_Conformer(nn.Module):
    def __init__(self, args):
        super().__init__()

        # Load pre-trained Wav2vec model and fine-tune
        self.variable = args.variable

        self.ssl_model = SSLModel()
        self.LL = nn.Linear(1024, args.dim)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

        self.conformer = MyConformer(
            emb_size=args.dim,
            n_encoders=args.num_layers,
            heads=args.heads,
            kernel_size=args.kernel_size
        )

    def forward(self, x, x_len=None):

        if self.variable:
            x_ssl_feat = self.ssl_model(x, x_len) # (bs, frame_number, 1024)
            x_len = torch.div(x_len, 320, rounding_mode="floor").int() - 1
        else:
            x_ssl_feat = self.ssl_model(x)

        x = self.LL(x_ssl_feat) # (bs, frame_number, feat_out_dim)
        x = x.unsqueeze(dim=1)  #(bs, 1, frame_number, feat_out_dim)
        x = self.first_bn(x)
        x = self.selu(x)
        x = x.squeeze(dim=1)

        if self.variable:
            out, emb = self.conformer(x, x_len)
        else:
            out, emb = self.conformer(x)
        
        return out, emb
