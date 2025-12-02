import numpy as np
import random
from typing import List, Tuple

def calculate_distance(city1: Tuple[float, float], city2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two cities using x_km, y_km coordinates."""
    return np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

def calculate_total_distance(route: List[int], cities: np.ndarray) -> float:
    """Calculate total distance of a route (including return to start)."""
    total = 0
    for i in range(len(route)):
        city1 = cities[route[i]]
        city2 = cities[route[(i + 1) % len(route)]]  # % to wrap back to start
        total += calculate_distance(city1, city2)
    return total

def random_search(cities: np.ndarray, iterations: int = 1000):
    """
    Random Search algorithm for TSP.
    
    Args:
        cities: numpy array of city coordinates (x_km, y_km)
        iterations: number of random routes to try
        
    Yields:
        tuple: (current_route, current_distance, best_route, best_distance, iteration)
    """
    n_cities = len(cities)
    
    # Start with a random route
    best_route = list(range(n_cities))
    random.shuffle(best_route)
    best_distance = calculate_total_distance(best_route, cities)
    
    # Yield initial state
    yield (best_route.copy(), best_distance, best_route.copy(), best_distance, 0)
    
    # Try random permutations
    for iteration in range(1, iterations + 1):
        # Generate a random route
        current_route = list(range(n_cities))
        random.shuffle(current_route)
        current_distance = calculate_total_distance(current_route, cities)
        
        # Update best if current is better
        if current_distance < best_distance:
            best_route = current_route.copy()
            best_distance = current_distance
        
        # Yield current state (even if not better, for visualization)
        yield (current_route, current_distance, best_route, best_distance, iteration)
