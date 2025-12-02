import numpy as np
import random
from .utils import calculate_total_distance

def random_search(cities: np.ndarray, iterations: int = 1000):
    """
    Random Search algorithm for TSP.
    Generates random permutations and keeps track of the best one found.
    
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
