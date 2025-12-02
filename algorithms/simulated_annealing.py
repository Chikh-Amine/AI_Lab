import numpy as np
import random
import math
from .utils import calculate_total_distance

def two_opt_swap(route, i, k):
    """
    Perform a 2-opt swap on the route.
    Reverses the order of cities between positions i and k.
    """
    new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
    return new_route

def get_random_neighbor(route):
    """
    Generate a random neighbor using 2-opt swap.
    
    Args:
        route: Current route
        
    Returns:
        A neighbor route
    """
    n = len(route)
    i = random.randint(1, n - 2)
    k = random.randint(i + 1, n - 1)
    return two_opt_swap(route, i, k)

def simulated_annealing(cities: np.ndarray, iterations: int = 1000, 
                       initial_temp: float = 1000.0, cooling_rate: float = 0.995,
                       cooling_schedule: str = "exponential"):
    """
    Simulated Annealing algorithm for TSP.
    
    Args:
        cities: numpy array of city coordinates (x_km, y_km)
        iterations: maximum number of iterations
        initial_temp: starting temperature
        cooling_rate: rate at which temperature decreases (0 < rate < 1)
        cooling_schedule: "exponential", "linear", or "logarithmic"
        
    Yields:
        tuple: (current_route, current_distance, best_route, best_distance, iteration, temperature)
    """
    n_cities = len(cities)
    
    # Start with a random route
    current_route = list(range(n_cities))
    random.shuffle(current_route)
    current_distance = calculate_total_distance(current_route, cities)
    
    # Track the best solution found
    best_route = current_route.copy()
    best_distance = current_distance
    
    # Temperature
    temperature = initial_temp
    
    # Yield initial state
    yield (current_route.copy(), current_distance, best_route.copy(), best_distance, 0, temperature)
    
    for iteration in range(1, iterations + 1):
        # Generate a random neighbor
        neighbor_route = get_random_neighbor(current_route)
        neighbor_distance = calculate_total_distance(neighbor_route, cities)
        
        # Calculate the change in distance
        delta = neighbor_distance - current_distance
        
        # Decide whether to accept the neighbor
        if delta < 0:
            # Better solution - always accept
            current_route = neighbor_route
            current_distance = neighbor_distance
            
            # Update best if necessary
            if current_distance < best_distance:
                best_route = current_route.copy()
                best_distance = current_distance
        else:
            # Worse solution - accept with probability based on temperature
            acceptance_probability = math.exp(-delta / temperature)
            
            if random.random() < acceptance_probability:
                # Accept the worse solution
                current_route = neighbor_route
                current_distance = neighbor_distance
        
        # Cool down the temperature based on cooling schedule
        if cooling_schedule == "exponential":
            # Exponential cooling: T = T0 * (cooling_rate ^ iteration)
            temperature = initial_temp * (cooling_rate ** iteration)
        elif cooling_schedule == "linear":
            # Linear cooling: T = T0 - (T0 / iterations) * iteration
            temperature = initial_temp - (initial_temp / iterations) * iteration
            temperature = max(temperature, 0.01)  # Prevent negative temperature
        elif cooling_schedule == "logarithmic":
            # Logarithmic cooling: T = T0 / log(1 + iteration)
            temperature = initial_temp / math.log(1 + iteration)
        
        # Yield current state
        yield (current_route.copy(), current_distance, best_route.copy(), best_distance, iteration, temperature)
    
    # Final yield
    yield (best_route.copy(), best_distance, best_route.copy(), best_distance, iterations, temperature)
