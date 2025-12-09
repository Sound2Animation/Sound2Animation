"""Rotation utilities - 6D rotation representation"""
import torch
import torch.nn.functional as F


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation to 3x3 rotation matrix.

    Reference: Zhou et al. "On the Continuity of Rotation Representations in Neural Networks"

    Args:
        d6: [*, 6] 6D rotation representation (first two rows flattened)
    Returns:
        [*, 3, 3] rotation matrix
    """
    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)  # stack as rows


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Convert 3x3 rotation matrix to 6D representation.

    Args:
        matrix: [*, 3, 3] rotation matrix
    Returns:
        [*, 6] 6D rotation
    """
    return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to quaternion (w, x, y, z).

    Args:
        matrix: [*, 3, 3] rotation matrix
    Returns:
        [*, 4] quaternion (w, x, y, z)
    """
    batch_dim = matrix.shape[:-2]
    m = matrix.reshape(-1, 3, 3)

    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]

    quat = torch.zeros(m.shape[0], 4, device=matrix.device, dtype=matrix.dtype)

    # Case 1: trace > 0
    mask = trace > 0
    s = torch.sqrt(trace[mask] + 1.0) * 2  # s = 4 * w
    quat[mask, 0] = 0.25 * s
    quat[mask, 1] = (m[mask, 2, 1] - m[mask, 1, 2]) / s
    quat[mask, 2] = (m[mask, 0, 2] - m[mask, 2, 0]) / s
    quat[mask, 3] = (m[mask, 1, 0] - m[mask, 0, 1]) / s

    # Case 2: m[0,0] > m[1,1] and m[0,0] > m[2,2]
    mask = ~mask & (m[:, 0, 0] > m[:, 1, 1]) & (m[:, 0, 0] > m[:, 2, 2])
    s = torch.sqrt(1.0 + m[mask, 0, 0] - m[mask, 1, 1] - m[mask, 2, 2]) * 2
    quat[mask, 0] = (m[mask, 2, 1] - m[mask, 1, 2]) / s
    quat[mask, 1] = 0.25 * s
    quat[mask, 2] = (m[mask, 0, 1] + m[mask, 1, 0]) / s
    quat[mask, 3] = (m[mask, 0, 2] + m[mask, 2, 0]) / s

    # Case 3: m[1,1] > m[2,2]
    mask = (trace <= 0) & ~(m[:, 0, 0] > m[:, 1, 1]) & (m[:, 1, 1] > m[:, 2, 2])
    s = torch.sqrt(1.0 + m[mask, 1, 1] - m[mask, 0, 0] - m[mask, 2, 2]) * 2
    quat[mask, 0] = (m[mask, 0, 2] - m[mask, 2, 0]) / s
    quat[mask, 1] = (m[mask, 0, 1] + m[mask, 1, 0]) / s
    quat[mask, 2] = 0.25 * s
    quat[mask, 3] = (m[mask, 1, 2] + m[mask, 2, 1]) / s

    # Case 4: else
    mask = (trace <= 0) & ~(m[:, 0, 0] > m[:, 1, 1]) & ~(m[:, 1, 1] > m[:, 2, 2])
    s = torch.sqrt(1.0 + m[mask, 2, 2] - m[mask, 0, 0] - m[mask, 1, 1]) * 2
    quat[mask, 0] = (m[mask, 1, 0] - m[mask, 0, 1]) / s
    quat[mask, 1] = (m[mask, 0, 2] + m[mask, 2, 0]) / s
    quat[mask, 2] = (m[mask, 1, 2] + m[mask, 2, 1]) / s
    quat[mask, 3] = 0.25 * s

    return quat.reshape(*batch_dim, 4)


def quaternion_to_rotation_6d(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion (w, x, y, z) to 6D rotation.

    Args:
        quat: [*, 4] quaternion (w, x, y, z)
    Returns:
        [*, 6] 6D rotation (first two rows of rotation matrix)
    """
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # Rotation matrix from quaternion (row-major)
    r00 = 1 - 2 * (y*y + z*z)
    r01 = 2 * (x*y - z*w)
    r02 = 2 * (x*z + y*w)
    r10 = 2 * (x*y + z*w)
    r11 = 1 - 2 * (x*x + z*z)
    r12 = 2 * (y*z - x*w)

    # 6D: first two rows of rotation matrix
    return torch.stack([r00, r01, r02, r10, r11, r12], dim=-1)
