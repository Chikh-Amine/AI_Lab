import numpy as np
import random
from .utils import calculate_total_distance

def two_opt_swap(route, i, k):
    """
    Perform a 2-opt swap on the route.
    Reverses the order of cities between positions i and k.
    
    Args:
        route: Current route
        i: Start position
        k: End position
        
    Returns:
        New route with the segment reversed
    """
    new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
    return new_route

def three_opt_swap(route, i, j, k):
    """
    Perform a 3-opt swap on the route.
    There are multiple ways to reconnect three segments - we use one efficient variant.
    
    Args:
        route: Current route
        i, j, k: Three cut positions (i < j < k)
        
    Returns:
        New route with segments reconnected
    """
    # Split route into segments
    # Segment A: route[0:i]
    # Segment B: route[i:j]
    # Segment C: route[j:k]
    # Segment D: route[k:]
    
    # Try one of the 3-opt reconnection variants (reverse middle segment)
    new_route = route[:i] + route[j:k] + route[i:j] + route[k:]
    return new_route

def local_search(cities: np.ndarray, iterations: int = 1000, opt_type: str = "2-opt"):
    """
    Local Search algorithm for TSP using 2-opt or 3-opt.
    Starts with a random solution and iteratively improves it.
    
    Args:
        cities: numpy array of city coordinates (x_km, y_km)
        iterations: maximum number of iterations
        opt_type: "2-opt" or "3-opt"
        
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
    
    improvements_found = True
    iteration = 0
    
    while iteration < iterations and improvements_found:
        iteration += 1
        improvements_found = False
        
        if opt_type == "2-opt":
            # Try all possible 2-opt swaps
            for i in range(1, n_cities - 1):
                for k in range(i + 1, n_cities):
                    # Create new route with 2-opt swap
                    new_route = two_opt_swap(current_route, i, k)
                    new_distance = calculate_total_distance(new_route, cities)
                    
                    # If improvement found, accept it
                    if new_distance < current_distance:
                        current_route = new_route
                        current_distance = new_distance
                        improvements_found = True
                        
                        # Update best if necessary
                        if current_distance < best_distance:
                            best_route = current_route.copy()
                            best_distance = current_distance
                        
                        # Yield current state
                        yield (current_route.copy(), current_distance, best_route.copy(), best_distance, iteration)
                        break  # Restart the search from the new solution
                
                if improvements_found:
                    break
        
        elif opt_type == "3-opt":
            # Try 3-opt swaps (sampling for efficiency)
            # Full 3-opt is very expensive, so we sample random triplets
            attempts = min(100, n_cities * (n_cities - 1) // 2)  # Limit attempts per iteration
            
            for _ in range(attempts):
                # Pick three random cut positions
                cuts = sorted(random.sample(range(1, n_cities), 3))
                i, j, k = cuts
                
                # Try the 3-opt swap
                new_route = three_opt_swap(current_route, i, j, k)
                new_distance = calculate_total_distance(new_route, cities)
                
                # If improvement found, accept it
                if new_distance < current_distance:
                    current_route = new_route
                    current_distance = new_distance
                    improvements_found = True
                    
                    # Update best if necessary
                    if current_distance < best_distance:
                        best_route = current_route.copy()
                        best_distance = current_distance
                    
                    # Yield current state
                    yield (current_route.copy(), current_distance, best_route.copy(), best_distance, iteration)
                    break  # Found improvement, restart
        
        # If no improvement in this iteration, yield the state anyway
        if not improvements_found:
            yield (current_route.copy(), current_distance, best_route.copy(), best_distance, iteration)
    
    # Final yield
    yield (best_route.copy(), best_distance, best_route.copy(), best_distance, iteration)
