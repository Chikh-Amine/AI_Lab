
import numpy as np

def calculate_total_distance(route, coords):
    """Compute total TSP route length in km."""
    total = 0.0
    for i in range(len(route)):
        city_a = coords[route[i]]
        city_b = coords[route[(i + 1) % len(route)]]  # wrap around
        total += np.linalg.norm(city_a - city_b)
    return total


def random_search(coords, iterations):
    """
    Generator that performs random search for TSP.
    Yields:
      current_route, current_distance, best_route, best_distance, iteration
    """
    num_cities = coords.shape[0]

    # create an initial route
    best_route = np.arange(num_cities)
    np.random.shuffle(best_route)
    best_route = best_route.tolist()

    best_distance = calculate_total_distance(best_route, coords)

    for iteration in range(1, iterations + 1):
        # generate a random route
        current_route = np.arange(num_cities)
        np.random.shuffle(current_route)
        current_route = current_route.tolist()

        current_distance = calculate_total_distance(current_route, coords)

        # update best if improved
        if current_distance < best_distance:
            best_route = current_route[:]
            best_distance = current_distance

        yield current_route, current_distance, best_route, best_distance, iteration
