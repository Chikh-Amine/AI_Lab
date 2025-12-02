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
    Returns the best reconnection among multiple 3-opt variants.
    
    Args:
        route: Current route
        i, j, k: Three cut positions (i < j < k)
        
    Returns:
        List of new route variants to try
    """
    # Split route into segments
    # A: route[0:i]
    # B: route[i:j]
    # C: route[j:k]
    # D: route[k:]
    
    A = route[:i]
    B = route[i:j]
    C = route[j:k]
    D = route[k:]
    
    # Generate different 3-opt reconnection variants
    variants = [
        A + B[::-1] + C + D,        # Reverse B
        A + B + C[::-1] + D,        # Reverse C
        A + C + B + D,              # Swap B and C
        A + C[::-1] + B + D,        # Reverse C, then swap
        A + B[::-1] + C[::-1] + D,  # Reverse both B and C
        A + C + B[::-1] + D,        # Swap, then reverse B
    ]
    
    return variants

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
            # Try 3-opt swaps
            # For efficiency, we try all combinations but limit to reasonable subset
            found_improvement_this_iter = False
            
            for i in range(1, n_cities - 2):
                if found_improvement_this_iter:
                    break
                for j in range(i + 1, n_cities - 1):
                    if found_improvement_this_iter:
                        break
                    for k in range(j + 1, n_cities):
                        # Get all 3-opt variants
                        variants = three_opt_swap(current_route, i, j, k)
                        
                        # Try each variant
                        for new_route in variants:
                            new_distance = calculate_total_distance(new_route, cities)
                            
                            # If improvement found, accept it
                            if new_distance < current_distance:
                                current_route = new_route
                                current_distance = new_distance
                                improvements_found = True
                                found_improvement_this_iter = True
                                
                                # Update best if necessary
                                if current_distance < best_distance:
                                    best_route = current_route.copy()
                                    best_distance = current_distance
                                
                                # Yield current state
                                yield (current_route.copy(), current_distance, best_route.copy(), best_distance, iteration)
                                break  # Found improvement, restart
                        
                        if found_improvement_this_iter:
                            break
        
        # If no improvement in this iteration, yield the state anyway
        if not improvements_found:
            yield (current_route.copy(), current_distance, best_route.copy(), best_distance, iteration)
    
    # Final yield
    yield (best_route.copy(), best_distance, best_route.copy(), best_distance, iteration)
