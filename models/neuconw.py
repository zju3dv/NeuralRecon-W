import numpy as np
import torch
from torch import nn
from collections import OrderedDict


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        "include_input": True,
        "input_dims": input_dims,
        "max_freq_log2": multires - 1,
        "num_freqs": multires,
        "log_sampling": True,
        "periodic_fns": [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)

    def embed(x, eo=embedder_obj):
        return eo.embed(x)

    return embed, embedder_obj.out_dim


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class RenderingNetwork(nn.Module):
    def __init__(
        self,
        d_feature,
        mode,
        d_in,
        d_out,
        d_hidden,
        n_layers,
        head_channels=128,
        in_channels_dir_a=48,
        static_head_layers=2,
        weight_norm=True,
        multires_view=4,
        squeeze_out=True,
        encode_apperence=True,
    ):
        super().__init__()

        self.mode = mode
        self.squeeze_out = squeeze_out
        self.encode_apperence = encode_apperence
        if self.encode_apperence:
            # -3 is to remove dir
            dims = (
                [d_in + head_channels - 3]
                + [d_hidden for _ in range(n_layers)]
                + [d_out]
            )
        else:
            dims = [d_in + d_feature] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += 0 if self.encode_apperence else (input_ch - 3)
            in_channels_dir_a += input_ch

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()

        if self.encode_apperence:
            # direction and appearance encoding layers
            static_encoding_od = OrderedDict(
                [
                    (
                        "static_linear_0",
                        nn.Linear(d_feature + in_channels_dir_a, head_channels),
                    ),
                    ("static_relu_0", nn.ReLU(True)),
                ]
            )
            for s_layer_i in range(1, static_head_layers):
                static_encoding_od[f"static_linear_{s_layer_i}"] = nn.Linear(
                    head_channels, head_channels
                )
                static_encoding_od[f"static_relu_{s_layer_i}"] = nn.ReLU(True)
            self.static_encoding = nn.Sequential(static_encoding_od)
            self.xyz_encoding_final = nn.Linear(d_feature, d_feature)

    def forward(self, points, normals, view_dirs, feature_vectors, input_dir_a=None):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.encode_apperence:
            # color prediction
            xyz_encoding_final = self.xyz_encoding_final(feature_vectors)  # (B, W)
            dir_encoding_input = torch.cat(
                [xyz_encoding_final, view_dirs, input_dir_a], 1
            )
            dir_encoding = self.static_encoding(dir_encoding_input)
        else:
            xyz_encoding_final = torch.zeros_like(feature_vectors)

        rendering_input = None

        if self.mode == "idr":
            if self.encode_apperence:
                rendering_input = torch.cat([points, normals, dir_encoding], dim=-1)
            else:
                rendering_input = torch.cat(
                    [points, view_dirs, normals, feature_vectors], dim=-1
                )
        elif self.mode == "no_view_dir":
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == "no_normal":
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        if self.squeeze_out:
            x = torch.sigmoid(x)
        return x, xyz_encoding_final, view_dirs


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter("variance", nn.Parameter(torch.tensor(init_val)))

    def forward(self, x):
        return torch.ones([len(x), 1]).to(x.device) * torch.exp(self.variance * 10.0)


# This implementation is borrowed from IDR: https://github.com/lioryariv/idr
class SDFNetwork(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        d_hidden,
        n_layers,
        skip_in=(4,),
        multires=6,
        bias=0.5,
        scale=1,
        geometric_init=True,
        weight_norm=True,
        inside_outside=False,
    ):
        super(SDFNetwork, self).__init__()

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        if multires > 0:
            embed_fn, input_ch = get_embedder(multires, input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale
        self.multires = multires

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    if not inside_outside:
                        torch.nn.init.normal_(
                            lin.weight,
                            mean=np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        torch.nn.init.constant_(lin.bias, -bias)
                    else:
                        torch.nn.init.normal_(
                            lin.weight,
                            mean=-np.sqrt(np.pi) / np.sqrt(dims[l]),
                            std=0.0001,
                        )
                        torch.nn.init.constant_(lin.bias, bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(
                        lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(
                        lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(
                        lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim)
                    )

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.activation = nn.Softplus(beta=100)

    def forward(self, inputs):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs.reshape(-1, 3))

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return gradients.reshape(-1, 3)


class NeuconW(nn.Module):
    def __init__(
        self,
        sdfNet_config,
        colorNet_config,
        SNet_config,
        in_channels_a,
        encode_a,
    ):

        super(NeuconW, self).__init__()
        self.sdfNet_config = sdfNet_config
        self.colorNet_config = colorNet_config
        self.SNet_config = SNet_config
        self.in_channels_a = in_channels_a
        self.encode_a = encode_a

        # xyz encoding layers + sdf layer
        self.sdf_net = SDFNetwork(**self.sdfNet_config)

        self.xyz_encoding_final = nn.Linear(512, 512)

        # Static deviation
        self.deviation_network = SingleVarianceNetwork(**self.SNet_config)

        # Static color
        self.color_net = RenderingNetwork(
            **self.colorNet_config,
            in_channels_dir_a=self.in_channels_a,
            encode_apperence=self.encode_a,
        )

    def sdf(self, input_xyz):
        # geometry prediction
        return self.sdf_net.sdf(input_xyz)  # (B, w+1)
        # return static_sdf[:, 1], static_sdf[:, 1:]

    def gradient(self, x):
        return self.sdf_net.gradient(x)

    def forward(self, x):
        device = x.device
        input_xyz, view_dirs, input_dir_a = torch.split(
            x, [3, 3, self.in_channels_a], dim=-1
        )

        n_rays, n_samples, _ = input_xyz.size()
        input_dir_a = input_dir_a.view(n_rays * n_samples, -1)

        # geometry prediction
        sdf_nn_output = self.sdf_net(input_xyz)  # (B, 1), (B, W)
        static_sdf = sdf_nn_output[:, :1]
        xyz_ = sdf_nn_output[:, 1:]

        # color prediction
        static_gradient = self.gradient(input_xyz)
        static_rgb, xyz_encoding_final, view_encoded = self.color_net(
            input_xyz.view(-1, 3),
            static_gradient.view(-1, 3),
            view_dirs.view(-1, 3),
            xyz_,
            input_dir_a,
        )  # (B, 3)
        # sdf gradient
        static_deviation = self.deviation_network(torch.zeros([1, 3], device=device))[
            :, :1
        ].clamp(
            1e-6, 1e6
        )  # (B, 1)

        static_out = (
            static_rgb.view(n_rays, n_samples, 3),
            static_deviation,
            static_sdf.view(n_rays, n_samples),
            static_gradient.view(n_rays, n_samples, 3),
        )

        return static_out
