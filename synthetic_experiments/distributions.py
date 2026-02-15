import math
import torch

def generate_x_and_gauss_distribution(num_samples: int, x_weight: float = 0.5, variance: float = 0.0) -> torch.Tensor:    
    component_ids = torch.rand(num_samples) < x_weight
    n_x = component_ids.sum().item()
    n_gaussian = num_samples - n_x

    samples = torch.zeros(num_samples, 2)
    samples[component_ids] = generate_x_distribution(n_x, variance=variance)
    samples[~component_ids] = torch.randn((n_gaussian, 2))  # N(0, I)

    return samples

def generate_x_distribution(num_samples: int, variance: float = 0.0) -> torch.Tensor:
    a = math.sqrt(3)  # this is the lower and upper bound we should sample from. 

    num_half = num_samples // 2
    remainder = num_samples - num_half * 2

    t1 = torch.empty(num_half).uniform_(-a, a)
    t2 = torch.empty(num_half + remainder).uniform_(-a, a)

    diag1 = torch.stack([t1, t1], dim=1)   # y = x
    diag2 = torch.stack([t2, -t2], dim=1)  # y = -x

    points = torch.cat([diag1, diag2], dim=0)

    if variance > 0.0:
        noise = torch.randn_like(points) * variance**0.5
        points += noise

    return points


def generate_o_distribution(num_samples: int, center=(0.5, 0.5), radius=0.4) -> torch.Tensor:
    """
    Generates a 2D dataset forming a circle ('O') shape.

    Args:
        num_samples (int): The number of 2D points to generate.
        center (tuple): The center of the circle.
        radius (float): The radius of the circle.

    Returns:
        torch.Tensor: A tensor of shape (num_samples, 2) containing the generated points.
    """
    angles = 2 * torch.pi * torch.rand(num_samples, 1)
    x1 = center[0] + radius * torch.cos(angles)
    x2 = center[1] + radius * torch.sin(angles)
    points = torch.cat((x1, x2), dim=1)
    return points


def generate_l_distribution(num_samples: int) -> torch.Tensor:
    """
    Generates a 2D dataset forming an "L" shape on the axes in the [0,1]x[0,1] plane.

    The distribution is a mixture of two uniform distributions on two perpendicular lines
    forming an L shape (the positive x and y axes).

    Args:
        num_samples (int): The number of 2D points to generate.

    Returns:
        torch.Tensor: A tensor of shape (num_samples, 2) containing the generated points.
    """
    points = torch.zeros(num_samples, 2)
    choice_mask = torch.rand(num_samples) < 0.5

    num_vertical = choice_mask.sum()
    num_horizontal = num_samples - num_vertical

    # Vertical line (y-axis)
    points[choice_mask, 0] = 0
    points[choice_mask, 1] = torch.rand(num_vertical)

    # Horizontal line (x-axis)
    points[~choice_mask, 0] = torch.rand(num_horizontal)
    points[~choice_mask, 1] = 0

    return points


def generate_dash_distribution(num_samples: int, y_coords=(0.25, 0.75)) -> torch.Tensor:
    """
    Generates a 2D dataset forming two horizontal lines ('-').

    Args:
        num_samples (int): The number of 2D points to generate.
        y_coords (tuple): The y-coordinates of the two horizontal lines.

    Returns:
        torch.Tensor: A tensor of shape (num_samples, 2) containing the generated points.
    """
    x1 = torch.rand(num_samples, 1)
    choice_mask = torch.rand(num_samples, 1) < 0.5
    x2 = torch.where(choice_mask, torch.tensor(y_coords[0]), torch.tensor(y_coords[1]))
    points = torch.cat((x1, x2), dim=1)
    return points

def generate_stretched_sunshine_distribution(num_samples: int) -> torch.Tensor:
    """
    Generates a specified number of 2D points based on a bimodal distribution
    created by stretching and angular filtering.

    Args:
        num_samples (int): The exact number of points to generate.

    Returns:
        torch.Tensor: A tensor of shape (num_samples, 2) containing the generated points.
    """
    pts = []
    
    while len(pts) < num_samples:
        # With 50% probability, stretch the y-dimension
        if torch.rand(1) < 0.5:
            p = torch.randn(2, 1)
            p[1] = 2 * p[1]
            
            theta = torch.atan2(p[0], p[1])
            degs = theta / math.pi * 180
            sector = math.floor(degs / 10)
            
            # Keep the point if the sector is even
            if sector % 2 == 0:
                pts.append(p.T)  # Transpose to get a 1x2 tensor
        # With 50% probability, stretch the x-dimension
        else:
            p = torch.randn(2, 1)
            p[0] = 2 * p[0]
            
            theta = torch.atan2(p[0], p[1])
            degs = theta / math.pi * 180
            sector = math.floor(degs / 10)
            
            # Keep the point if the sector is odd
            if sector % 2 != 0:
                pts.append(p.T)  # Transpose to get a 1x2 tensor

    return torch.cat(pts, dim=0)


def generate_gauss_distribution(num_samples: int, mean=None, cov=None) -> torch.Tensor:
    """
    Generates a 2D dataset from a Gaussian distribution.

    Args:
        num_samples (int): The number of 2D points to generate.
        mean (list, optional): The mean of the Gaussian. Defaults to [0.5, 0.5].
        cov (list, optional): The covariance matrix. Defaults to [[0.05, 0], [0, 0.05]].

    Returns:
        torch.Tensor: A tensor of shape (num_samples, 2) containing the generated points.
    """
    if mean is None:
        mean = torch.tensor([0.0, 0.0])
    if cov is None:
        cov = torch.tensor([[1.0, 0], [0, 1.0]])

    dist = torch.distributions.MultivariateNormal(mean, cov)
    points = dist.sample((num_samples,))
    return points

def cut_and_rotate_pie_slices(samples: torch.Tensor, num_slices: int) -> torch.Tensor:
    """
    Cuts the 2D Gaussian samples into pie slices and rotates even-indexed slices.

    This function first converts the Cartesian coordinates (x, y) to polar
    coordinates (r, theta) to easily define the pie slices. It then rotates
    the samples belonging to even-numbered slices clockwise by the angle of
    a single slice.

    Args:
        samples (torch.Tensor): The original 2D points, shape (num_samples, 2).
        num_slices (int): The number of slices to cut the data into.

    Returns:
        torch.Tensor: The new tensor of points after rotation.
    """
    # Get x and y coordinates from the samples tensor
    x = samples[:, 0]
    y = samples[:, 1]

    # Convert Cartesian (x, y) to polar (r, theta) coordinates
    # atan2 is used for robust angle calculation, returning values in [-pi, pi]
    r = torch.linalg.norm(samples, dim=1)
    theta = torch.atan2(y, x)

    # Normalize theta to be in the range [0, 2*pi) for easier slicing
    theta_normalized = (theta + 2 * math.pi) % (2 * math.pi)

    # Determine the angle for each slice
    slice_angle = 2 * math.pi / num_slices

    # Assign each point to a slice index based on its angle
    slice_indices = torch.floor(theta_normalized / slice_angle).long()

    # Create a mask for even-indexed slices (0, 2, 4, etc.)
    is_even_slice = (slice_indices % 2 == 0)

    # The rotation angle is the same as the slice angle, rotated clockwise
    # This means a negative angle in the standard rotation matrix formula.
    # The new angle for even slices will be (theta_normalized - slice_angle)
    theta_rotated = torch.where(is_even_slice, theta_normalized - slice_angle, theta_normalized)

    # Convert the points back to Cartesian coordinates from their new polar form
    x_rotated = r * torch.cos(theta_rotated)
    y_rotated = r * torch.sin(theta_rotated)

    # Combine the new x and y coordinates into a single tensor
    rotated_samples = torch.stack((x_rotated, y_rotated), dim=1)

    return rotated_samples



def generate_sunshine_distribution(num_samples: int, mean=None, cov=None, num_slices=16) -> torch.Tensor:
    gaussian_points = generate_gauss_distribution(num_samples, mean=mean, cov=cov)
    rotated_points = cut_and_rotate_pie_slices(samples=gaussian_points, num_slices=num_slices)
    return rotated_points


def generate_dot_distribution(num_samples: int, location=(0.5, 0.5)) -> torch.Tensor:
    """
    Generates a 2D dataset where all points are at a single location ('.').

    Args:
        num_samples (int): The number of 2D points to generate.
        location (tuple): The location of the points.

    Returns:
        torch.Tensor: A tensor of shape (num_samples, 2) containing the generated points.
    """
    points = torch.full((num_samples, 2), fill_value=location[0])
    points[:, 1] = location[1]
    return points 