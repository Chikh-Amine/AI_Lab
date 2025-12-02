import numpy as np
import random
from .utils import calculate_total_distance

def two_opt_swap(route, i, k):
    """
    Perform a 2-opt swap on the route.
    Reverses the order of cities between positions i and k.
    """
    new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
    return new_route

def hill_climbing(cities: np.ndarray, iterations: int = 1000, variant: str = "steepest"):
    """
    Hill Climbing algorithm for TSP.
    
    Two variants:
    - Steepest Ascent: Evaluates all neighbors, picks the best
    - First Improvement: Accepts the first improvement found
    
    Args:
        cities: numpy array of city coordinates (x_km, y_km)
        iterations: maximum number of iterations
        variant: "steepest" or "first" improvement
        
    Yields:
        tuple: (current_route, current_distance, best_route, best_distance, iteration)
    """
    n_cities = len(cities)
    
    # Start with a random route
    current_route = list(range(n_cities))
    random.shuffle(current_route)
    current_distance = calculate_total_distance(current_route, cities)
    
    best_route = current_route.copy()
    best_distance = current_distance
    
    # Yield initial state
    yield (current_route.copy(), current_distance, best_route.copy(), best_distance, 0)
    
    iteration = 0
    
    while iteration < iterations:
        iteration += 1
        improved = False
        
        if variant == "steepest":
            # Steepest Ascent: Find the best neighbor
            best_neighbor = None
            best_neighbor_distance = current_distance
            
            for i in range(1, n_cities - 1):
                for k in range(i + 1, n_cities):
                    # Generate neighbor
                    neighbor = two_opt_swap(current_route, i, k)
                    neighbor_distance = calculate_total_distance(neighbor, cities)
                    
                    # Track best neighbor
                    if neighbor_distance < best_neighbor_distance:
                        best_neighbor = neighbor
                        best_neighbor_distance = neighbor_distance
                        improved = True
            
            # Move to best neighbor if improvement found
            if improved:
                current_route = best_neighbor
                current_distance = best_neighbor_distance
                
                # Update global best
                if current_distance < best_distance:
                    best_route = current_route.copy()
                    best_distance = current_distance
                
                # Yield state
                yield (current_route.copy(), current_distance, best_route.copy(), best_distance, iteration)
            else:
                # No improvement found - stuck at local optimum
                yield (current_route.copy(), current_distance, best_route.copy(), best_distance, iteration)
                break
        
        elif variant == "first":
            # First Improvement: Accept first better neighbor
            for i in range(1, n_cities - 1):
                if improved:
                    break
                for k in range(i + 1, n_cities):
                    # Generate neighbor
                    neighbor = two_opt_swap(current_route, i, k)
                    neighbor_distance = calculate_total_distance(neighbor, cities)
                    
                    # Accept first improvement
                    if neighbor_distance < current_distance:
                        current_route = neighbor
                        current_distance = neighbor_distance
                        improved = True
                        
                        # Update global best
                        if current_distance < best_distance:
                            best_route = current_route.copy()
                            best_distance = current_distance
                        
                        # Yield state
                        yield (current_route.copy(), current_distance, best_route.copy(), best_distance, iteration)
                        break
            
            # If no improvement found, we're stuck
            if not improved:
                yield (current_route.copy(), current_distance, best_route.copy(), best_distance, iteration)
                break
    
    # Final yield
    yield (best_route.copy(), best_distance, best_route.copy(), best_distance, iteration)
