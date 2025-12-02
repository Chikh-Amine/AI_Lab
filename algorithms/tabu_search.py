import numpy as np
import random
from collections import deque
from .utils import calculate_total_distance

def two_opt_swap(route, i, k):
    """
    Perform a 2-opt swap on the route.
    Reverses the order of cities between positions i and k.
    """
    new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
    return new_route

def get_move_signature(i, k):
    """
    Create a signature for a move to store in tabu list.
    The signature is the sorted tuple of the two swap positions.
    """
    return tuple(sorted([i, k]))

def get_all_neighbors_with_moves(route):
    """
    Generate all possible neighbors using 2-opt swaps along with their move signatures.
    
    Args:
        route: Current route
        
    Returns:
        List of tuples: (neighbor_route, move_signature)
    """
    neighbors = []
    n = len(route)
    
    for i in range(1, n - 1):
        for k in range(i + 1, n):
            neighbor = two_opt_swap(route, i, k)
            move_sig = get_move_signature(i, k)
            neighbors.append((neighbor, move_sig))
    
    return neighbors

def tabu_search(cities: np.ndarray, iterations: int = 1000, 
                tabu_tenure: int = 10, aspiration_enabled: bool = True):
    """
    Tabu Search algorithm for TSP.
    
    Args:
        cities: numpy array of city coordinates (x_km, y_km)
        iterations: maximum number of iterations
        tabu_tenure: how many iterations a move stays in the tabu list
        aspiration_enabled: if True, allow tabu moves that improve the best solution
        
    Yields:
        tuple: (current_route, current_distance, best_route, best_distance, iteration, tabu_size)
    """
    n_cities = len(cities)
    
    # Start with a random route
    current_route = list(range(n_cities))
    random.shuffle(current_route)
    current_distance = calculate_total_distance(current_route, cities)
    
    # Track the best solution found
    best_route = current_route.copy()
    best_distance = current_distance
    
    # Tabu list: stores move signatures with their expiration iteration
    tabu_list = deque(maxlen=tabu_tenure)
    
    # Yield initial state
    yield (current_route.copy(), current_distance, best_route.copy(), best_distance, 0, 0)
    
    for iteration in range(1, iterations + 1):
        # Generate all neighbors with their move signatures
        neighbors_with_moves = get_all_neighbors_with_moves(current_route)
        
        # Find the best non-tabu neighbor (or best tabu neighbor that satisfies aspiration)
        best_neighbor = None
        best_neighbor_distance = float('inf')
        best_move_sig = None
        
        for neighbor, move_sig in neighbors_with_moves:
            neighbor_distance = calculate_total_distance(neighbor, cities)
            
            # Check if move is tabu
            is_tabu = move_sig in tabu_list
            
            # Aspiration criterion: accept tabu move if it improves best solution
            aspiration_satisfied = aspiration_enabled and neighbor_distance < best_distance
            
            # Accept move if:
            # 1. Not tabu, OR
            # 2. Tabu but aspiration criterion is satisfied
            if not is_tabu or aspiration_satisfied:
                if neighbor_distance < best_neighbor_distance:
                    best_neighbor = neighbor
                    best_neighbor_distance = neighbor_distance
                    best_move_sig = move_sig
        
        # If we found a valid neighbor, move to it
        if best_neighbor is not None:
            current_route = best_neighbor
            current_distance = best_neighbor_distance
            
            # Add the move to tabu list
            tabu_list.append(best_move_sig)
            
            # Update best solution if necessary
            if current_distance < best_distance:
                best_route = current_route.copy()
                best_distance = current_distance
        else:
            # This should rarely happen, but if all neighbors are tabu and
            # none satisfy aspiration, we're stuck
            # In this case, we can either:
            # 1. Clear some tabu list entries
            # 2. Accept the least bad tabu move
            # For simplicity, we'll just clear oldest entry and continue
            if tabu_list:
                tabu_list.popleft()
        
        # Yield current state
        yield (current_route.copy(), current_distance, best_route.copy(), 
               best_distance, iteration, len(tabu_list))
    
    # Final yield
    yield (best_route.copy(), best_distance, best_route.copy(), 
           best_distance, iterations, len(tabu_list))
