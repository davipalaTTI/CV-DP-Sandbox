from typing import List, Tuple, Optional, Union
import numpy as np


# ======================== COORDINATE UTILITIES ========================

def normalize_coordinates(points: Union[Tuple[int, int], List[Tuple[int, int]]],
                          frame_size: Tuple[int, int]) -> Union[Tuple[float, float], List[Tuple[float, float]]]:
    """
    Normalize coordinates to 0-1 range

    Args:
        points: Single point or list of points (x, y)
        frame_size: Frame dimensions (width, height)

    Returns:
        Normalized coordinates
    """
    w, h = frame_size

    def normalize_point(point: Tuple[int, int]) -> Tuple[float, float]:
        x, y = point
        return (x / w, y / h)

    if isinstance(points, tuple) and len(points) == 2:
        return normalize_point(points)
    else:
        return [normalize_point(p) for p in points]


def denormalize_coordinates(points: Union[Tuple[float, float], List[Tuple[float, float]]],
                            frame_size: Tuple[int, int]) -> Union[Tuple[int, int], List[Tuple[int, int]]]:
    """
    Convert normalized coordinates back to pixels

    Args:
        points: Normalized coordinates (0-1 range)
        frame_size: Frame dimensions (width, height)

    Returns:
        Pixel coordinates
    """
    w, h = frame_size

    def denormalize_point(point: Tuple[float, float]) -> Tuple[int, int]:
        norm_x, norm_y = point
        return (int(norm_x * w), int(norm_y * h))

    if isinstance(points, tuple) and len(points) == 2:
        return denormalize_point(points)
    else:
        return [denormalize_point(p) for p in points]


def scale_coordinates(points: Union[Tuple[int, int], List[Tuple[int, int]]],
                      scale_factor: float) -> Union[Tuple[int, int], List[Tuple[int, int]]]:
    """
    Scale coordinates by a factor

    Args:
        points: Point(s) to scale
        scale_factor: Scaling factor

    Returns:
        Scaled coordinates
    """

    def scale_point(point: Tuple[int, int]) -> Tuple[int, int]:
        x, y = point
        return (int(x * scale_factor), int(y * scale_factor))

    if isinstance(points, tuple) and len(points) == 2:
        return scale_point(points)
    else:
        return [scale_point(p) for p in points]


# ======================== GEOMETRY UTILITIES ========================

def point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """
    Check if point is inside polygon using ray casting algorithm

    Args:
        point: Point to test (x, y)
        polygon: List of polygon vertices

    Returns:
        True if point is inside polygon
    """
    x, y = point
    n = len(polygon)
    if n < 3:
        return False

    inside = False
    p1x, p1y = polygon[0]

    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def line_intersection(line1: Tuple[Tuple[float, float], Tuple[float, float]],
                      line2: Tuple[Tuple[float, float], Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    """
    Find intersection point of two lines

    Args:
        line1: First line ((x1, y1), (x2, y2))
        line2: Second line ((x1, y1), (x2, y2))

    Returns:
        Intersection point or None if lines don't intersect
    """
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None  # Lines are parallel

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        # Lines intersect within segments
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (ix, iy)

    return None


def distance_point_to_line(point: Tuple[float, float],
                           line_start: Tuple[float, float],
                           line_end: Tuple[float, float]) -> float:
    """
    Calculate perpendicular distance from point to line segment

    Args:
        point: Point coordinates
        line_start: Line start point
        line_end: Line end point

    Returns:
        Distance to line
    """
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end

    # Line vector
    A = x2 - x1
    B = y2 - y1

    # Point vector from line start
    C = x0 - x1
    D = y0 - y1

    dot = C * A + D * B
    len_sq = A * A + B * B

    if len_sq == 0:
        # Line is actually a point
        return np.sqrt(C * C + D * D)

    param = dot / len_sq

    # Find closest point on line segment
    if param < 0:
        xx, yy = x1, y1
    elif param > 1:
        xx, yy = x2, y2
    else:
        xx = x1 + param * A
        yy = y1 + param * B

    # Calculate distance
    dx = x0 - xx
    dy = y0 - yy
    return np.sqrt(dx * dx + dy * dy)


def calculate_polygon_area(polygon: List[Tuple[float, float]]) -> float:
    """
    Calculate area of polygon using shoelace formula

    Args:
        polygon: List of polygon vertices

    Returns:
        Polygon area
    """
    if len(polygon) < 3:
        return 0.0

    area = 0.0
    n = len(polygon)

    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]

    return abs(area) / 2.0


def calculate_polygon_centroid(polygon: List[Tuple[float, float]]) -> Tuple[float, float]:
    """
    Calculate centroid of polygon

    Args:
        polygon: List of polygon vertices

    Returns:
        Centroid coordinates
    """
    if not polygon:
        return (0.0, 0.0)

    area = calculate_polygon_area(polygon)
    if area == 0:
        # Degenerate polygon, return average of points
        x_avg = sum(p[0] for p in polygon) / len(polygon)
        y_avg = sum(p[1] for p in polygon) / len(polygon)
        return (x_avg, y_avg)

    cx = 0.0
    cy = 0.0
    n = len(polygon)

    for i in range(n):
        j = (i + 1) % n
        cross = polygon[i][0] * polygon[j][1] - polygon[j][0] * polygon[i][1]
        cx += (polygon[i][0] + polygon[j][0]) * cross
        cy += (polygon[i][1] + polygon[j][1]) * cross

    factor = 1.0 / (6.0 * area)
    return (cx * factor, cy * factor)