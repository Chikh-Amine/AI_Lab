import numpy as np
from typing import List, Tuple

def calculate_distance(city1: Tuple[float, float], city2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two cities using x_km, y_km coordinates."""
    return np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

def calculate_total_distance(route: List[int], cities: np.ndarray) -> float:
    """
    Calculate total distance of a route (including return to start).
    
    Args:
        route: List of city indices representing the order to visit
        cities: numpy array of city coordinates (x_km, y_km)
        
    Returns:
        Total distance of the route in km
    """
    total = 0
    for i in range(len(route)):
        city1 = cities[route[i]]
        city2 = cities[route[(i + 1) % len(route)]]  # % to wrap back to start
        total += calculate_distance(city1, city2)
    return total
