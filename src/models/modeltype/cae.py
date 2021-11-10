import torch
import torch.nn as nn

from ..tools.losses import get_loss_function
from ..rotation2xyz import Rotation2xyz


class CAE(nn.Module):
    def __init__(self, encoder, decoder, device, lambdas, latent_dim, outputxyz,
                 pose_rep, glob, glob_rot, translation, jointstype, vertstrans, **kwargs):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

        self.outputxyz = outputxyz
        
        self.lambdas = lambdas
        
        self.latent_dim = latent_dim
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.device = device
        self.translation = translation
        self.jointstype = jointstype
        self.vertstrans = vertstrans
        
        self.losses = list(self.lambdas) + ["mixed"]

        self.rotation2xyz = Rotation2xyz(device=self.device)
        self.param2xyz = {"pose_rep": self.pose_rep,
                          "glob_rot": self.glob_rot,
                          "glob": self.glob,
                          "jointstype": self.jointstype,
                          "translation": self.translation,
                          "vertstrans": self.vertstrans}
    def interp_batch(self, batch, interp_type, interp_ratio):
        import numpy as np
        from scipy.interpolate import interp1d
        # print(batch.keys())
        # for k in batch.keys():
        #     print('{}: {}'.format(k, batch[k].shape))
        # print(interp_ratio)
        # print(interp_ratio is not None)
        assert type(interp_ratio) == int
        interp_keys = ['output']
        for k in interp_keys:
            # sample
            timeline = np.arange(batch[k].shape[-1])
            sample_timeline = timeline[0::interp_ratio]
            if sample_timeline[-1] != timeline[-1]:
                sample_timeline = np.append(sample_timeline, timeline[-1])
            sample = batch[k][..., sample_timeline].cpu().numpy()

            # print(timeline)
            # print(sample_timeline)
            # print(sample.shape)
            # print(len(sample.shape)-1)

            # interpoloate
            interp_fn = interp1d(sample_timeline, sample, axis=len(sample.shape)-1, kind=interp_type)
            interped = interp_fn(timeline)

            # scale_factor = tuple([1.] * (len(batch[k].shape) - 1 - 2) + [float(interp_ratio)])
            # interped = torch.nn.functional.interpolate(sample, scale_factor=scale_factor, mode=interp_type,
            #                                            align_corners=None, recompute_scale_factor=None)
            # print(interped.shape)

            assert interped.shape == batch[k].shape
            batch[k] = torch.tensor(interped, device=self.device, dtype=torch.float32)
        return batch

    def rot2xyz(self, x, mask, **kwargs):
        kargs = self.param2xyz.copy()
        kargs.update(kwargs)
        return self.rotation2xyz(x, mask, **kargs)
    
    def forward(self, batch, interp_ratio=None, interp_type='nearest'):
        if self.outputxyz:
            batch["x_xyz"] = self.rot2xyz(batch["x"], batch["mask"])
        elif self.pose_rep == "xyz":
            batch["x_xyz"] = batch["x"]
        # encode
        batch.update(self.encoder(batch))
        # decode
        batch.update(self.decoder(batch))
        if interp_ratio is not None:
            batch = self.interp_batch(batch, interp_type, interp_ratio)
        # if we want to output xyz
        if self.outputxyz:
            batch["output_xyz"] = self.rot2xyz(batch["output"], batch["mask"])
        elif self.pose_rep == "xyz":
            batch["output_xyz"] = batch["output"]
        return batch

    def compute_loss(self, batch):
        mixed_loss = 0
        losses = {}
        for ltype, lam in self.lambdas.items():
            loss_function = get_loss_function(ltype)
            loss = loss_function(self, batch)
            mixed_loss += loss*lam
            losses[ltype] = loss.item()
        losses["mixed"] = mixed_loss.item()
        return mixed_loss, losses

    @staticmethod
    def lengths_to_mask(lengths):
        max_len = max(lengths)
        if isinstance(max_len, torch.Tensor):
            max_len = max_len.item()
        index = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len)
        mask = index < lengths.unsqueeze(1)
        return mask

    def generate_one(self, cls, duration, fact=1, xyz=False):
        y = torch.tensor([cls], dtype=int, device=self.device)[None]
        lengths = torch.tensor([duration], dtype=int, device=self.device)
        mask = self.lengths_to_mask(lengths)
        z = torch.randn(self.latent_dim, device=self.device)[None]
        
        batch = {"z": fact*z, "y": y, "mask": mask, "lengths": lengths}
        batch = self.decoder(batch)

        if not xyz:
            return batch["output"][0]
        
        output_xyz = self.rot2xyz(batch["output"], batch["mask"])

        return output_xyz[0]
            
    def generate(self, classes, durations, nspa=1,
                 noise_same_action="random", noise_diff_action="random",
                 fact=1, interp_ratio=None, interp_type='nearest'):
        if nspa is None:
            nspa = 1
        nats = len(classes)
            
        y = classes.to(self.device).repeat(nspa)  # (view(nspa, nats))

        if len(durations.shape) == 1:
            lengths = durations.to(self.device).repeat(nspa)
        else:
            lengths = durations.to(self.device).reshape(y.shape)
        
        mask = self.lengths_to_mask(lengths)
        
        if noise_same_action == "random":
            if noise_diff_action == "random":
                z = torch.randn(nspa*nats, self.latent_dim, device=self.device)
            elif noise_diff_action == "same":
                z_same_action = torch.randn(nspa, self.latent_dim, device=self.device)
                z = z_same_action.repeat_interleave(nats, axis=0)
            else:
                raise NotImplementedError("Noise diff action must be random or same.")
        elif noise_same_action == "interpolate":
            if noise_diff_action == "random":
                z_diff_action = torch.randn(nats, self.latent_dim, device=self.device)
            elif noise_diff_action == "same":
                z_diff_action = torch.randn(1, self.latent_dim, device=self.device).repeat(nats, 1)
            else:
                raise NotImplementedError("Noise diff action must be random or same.")
            interpolation_factors = torch.linspace(-1, 1, nspa, device=self.device)
            z = torch.einsum("ij,k->kij", z_diff_action, interpolation_factors).view(nspa*nats, -1)
        elif noise_same_action == "same":
            if noise_diff_action == "random":
                z_diff_action = torch.randn(nats, self.latent_dim, device=self.device)
            elif noise_diff_action == "same":
                z_diff_action = torch.randn(1, self.latent_dim, device=self.device).repeat(nats, 1)
            else:
                raise NotImplementedError("Noise diff action must be random or same.")
            z = z_diff_action.repeat((nspa, 1))
        else:
            raise NotImplementedError("Noise same action must be random, same or interpolate.")

        batch = {"z": fact*z, "y": y, "mask": mask, "lengths": lengths}
        batch = self.decoder(batch)

        if interp_ratio is not None:
            batch = self.interp_batch(batch, interp_type, interp_ratio)

        if self.outputxyz:
            batch["output_xyz"] = self.rot2xyz(batch["output"], batch["mask"])
        elif self.pose_rep == "xyz":
            batch["output_xyz"] = batch["output"]
        return batch
    
    def return_latent(self, batch, seed=None):
        return self.encoder(batch)["z"]
