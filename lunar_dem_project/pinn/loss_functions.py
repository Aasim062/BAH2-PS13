import torch

def lunar_lambert_loss(pred_dem, brightness, albedo, i_angle_rad, l_weight=0.6):
    # Approximate surface normals
    dx = torch.gradient(pred_dem, dim=3)[0]
    dy = torch.gradient(pred_dem, dim=2)[0]
    dz = torch.ones_like(pred_dem)
    norm = torch.sqrt(dx**2 + dy**2 + dz**2)

    # Surface verticality approximation
    cos_i = torch.cos(i_angle_rad)
    cos_e = 1.0  # nadir assumed

    R_model = albedo * ((l_weight * cos_i) + (1 - l_weight) * cos_i / (cos_i + cos_e))
    brightness_pred = R_model

    return torch.mean((brightness - brightness_pred) ** 2)
