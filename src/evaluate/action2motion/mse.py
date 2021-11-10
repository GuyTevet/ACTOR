import torch


def calculate_mse(model, motion_loader, device):
    with torch.no_grad():
        loss = torch.nn.MSELoss(reduction='none')
        mse = []
        mse_xyz = []
        for batch in motion_loader:
            mse.append(loss(batch['x'], batch['output']))
            if 'x_xyz' in batch.keys():
                mse_xyz.append(loss(batch['x_xyz'], batch['output_xyz']))
        mse = torch.mean(torch.cat(mse))
        if 'x_xyz' in batch.keys():
            mse_xyz = torch.mean(torch.cat(mse_xyz))
        else:
            mse_xyz = None
    return mse, mse_xyz
