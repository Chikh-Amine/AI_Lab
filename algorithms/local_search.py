import numpy as np
from .random_search import calculate_total_distance


def two_opt_swap(route, i, k):
    """Return a new route where the segment between i and k is reversed."""
    new_route = route[:i] + route[i:k+1][::-1] + route[k+1:]
    return new_route


def three_opt_swap(route, i, j, k):
    """
    Simplified 3-opt: tries a few common reconnections.
    More versions exist, but this is enough for your project.
    """
    a, b, c = i, j, k

    r0 = route[:]  # unchanged
    r1 = route[:a] + route[a:b][::-1] + route[b:c] + route[c:]      # reverse segment 1
    r2 = route[:a] + route[a:b] + route[b:c][::-1] + route[c:]      # reverse segment 2
    r3 = route[:a] + route[b:c] + route[a:b] + route[c:]            # swap two segments
    r4 = route[:a] + route[b:c][::-1] + route[a:b][::-1] + route[c:]  # reverse both

    return [r0, r1, r2, r3, r4]


def local_search(coords, iterations, method="2-opt"):
    """
    Local search TSP using 2-opt or 3-opt.
    Yields:
        current_route, current_distance, best_route, best_distance, iteration
    """
    num_cities = len(coords)
    best_route = np.arange(num_cities).tolist()
    np.random.shuffle(best_route)

    best_distance = calculate_total_distance(best_route, coords)

    for iteration in range(1, iterations + 1):

        improved = False
        current_route = best_route[:]
        current_distance = best_distance

        if method == "2-opt":
            # Try all 2-opt swaps
            for i in range(num_cities - 1):
                for k in range(i+1, num_cities):
                    new_route = two_opt_swap(best_route, i, k)
                    new_distance = calculate_total_distance(new_route, coords)

                    if new_distance < best_distance:
                        best_route = new_route
                        best_distance = new_distance
                        improved = True
                        break
                if improved:
                    break

        elif method == "3-opt":
            for i in range(num_cities - 2):
                for j in range(i+1, num_cities - 1):
                    for k in range(j+1, num_cities):
                        candidates = three_opt_swap(best_route, i, j, k)
                        for r in candidates:
                            new_distance = calculate_total_distance(r, coords)
                            if new_distance < best_distance:
                                best_route = r
                                best_distance = new_distance
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

        yield current_route, current_distance, best_route, best_distance, iteration

        if not improved:
            # No improvement → local optimum reached
            break
