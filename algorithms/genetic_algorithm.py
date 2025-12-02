import numpy as np
import random
from .utils import calculate_total_distance

"""
GENETIC ALGORITHM EXPLANATION:
==============================

Genetic Algorithms (GA) are inspired by natural evolution. Instead of improving a single 
solution, GA maintains a POPULATION of solutions that evolve over generations.

Key Concepts:
-------------
1. POPULATION: A group of solutions (called "individuals" or "chromosomes")
   - Each route is like a DNA sequence
   - Population size: typically 50-200 individuals

2. FITNESS: How good a solution is (shorter distance = higher fitness)
   - Better solutions have better chance to "reproduce"

3. SELECTION: Choosing parents to create offspring
   - Tournament: Pick random individuals, choose the best
   - Roulette: Probability based on fitness
   - Elitism: Always keep the best solutions

4. CROSSOVER (Recombination): Combining two parent routes to create children
   - Order Crossover (OX): Preserves the order of cities from parents
   - Takes a segment from one parent, fills rest from other parent

5. MUTATION: Random changes to maintain diversity
   - Swap: Exchange two cities
   - Inversion: Reverse a segment of the route
   - Helps escape local optima

6. GENERATIONS: Each iteration creates a new population
   - Select parents → Create offspring → Mutate → Replace old population

The algorithm mimics "survival of the fittest" - better routes are more likely
to pass their "genes" to the next generation!
"""

def calculate_fitness(route, cities):
    """
    Calculate fitness of a route.
    Fitness is inverse of distance (shorter = better).
    """
    distance = calculate_total_distance(route, cities)
    return 1.0 / distance if distance > 0 else 0

def initialize_population(n_cities, population_size):
    """
    Create initial random population.
    Each individual is a random permutation of cities.
    """
    population = []
    for _ in range(population_size):
        route = list(range(n_cities))
        random.shuffle(route)
        population.append(route)
    return population

def tournament_selection(population, cities, tournament_size=3):
    """
    Tournament Selection: Pick random individuals, return the best.
    This gives better solutions higher chance to reproduce.
    """
    tournament = random.sample(population, tournament_size)
    best = min(tournament, key=lambda route: calculate_total_distance(route, cities))
    return best.copy()

def order_crossover(parent1, parent2):
    """
    Order Crossover (OX): Creates a child by combining two parents.
    
    Process:
    1. Select a random segment from parent1
    2. Copy this segment to the child
    3. Fill remaining positions with cities from parent2 (in order)
    
    This preserves the relative order from both parents.
    """
    size = len(parent1)
    
    # Choose two random crossover points
    start, end = sorted(random.sample(range(size), 2))
    
    # Copy segment from parent1
    child = [None] * size
    child[start:end] = parent1[start:end]
    
    # Fill remaining positions from parent2
    parent2_filtered = [city for city in parent2 if city not in child]
    
    # Fill before the segment
    for i in range(start):
        child[i] = parent2_filtered.pop(0)
    
    # Fill after the segment
    for i in range(end, size):
        child[i] = parent2_filtered.pop(0)
    
    return child

def mutate_swap(route, mutation_rate=0.01):
    """
    Swap Mutation: Randomly swap two cities.
    Mutation rate controls how often this happens.
    """
    route = route.copy()
    
    if random.random() < mutation_rate:
        # Pick two random positions and swap
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    
    return route

def mutate_inversion(route, mutation_rate=0.01):
    """
    Inversion Mutation: Reverse a segment of the route.
    More disruptive than swap mutation.
    """
    route = route.copy()
    
    if random.random() < mutation_rate:
        # Pick two random positions and reverse the segment between them
        i, j = sorted(random.sample(range(len(route)), 2))
        route[i:j] = reversed(route[i:j])
    
    return route

def genetic_algorithm(cities: np.ndarray, iterations: int = 100,
                     population_size: int = 100, mutation_rate: float = 0.01,
                     elitism_count: int = 2, mutation_type: str = "swap"):
    """
    Genetic Algorithm for TSP.
    
    Args:
        cities: numpy array of city coordinates (x_km, y_km)
        iterations: number of generations
        population_size: number of routes in each generation
        mutation_rate: probability of mutation (0.0 to 1.0)
        elitism_count: number of best solutions to keep unchanged
        mutation_type: "swap" or "inversion"
        
    Yields:
        tuple: (current_best_route, current_best_distance, overall_best_route, 
                overall_best_distance, generation, avg_fitness)
    """
    n_cities = len(cities)
    
    # Initialize population
    population = initialize_population(n_cities, population_size)
    
    # Find initial best
    best_route = min(population, key=lambda route: calculate_total_distance(route, cities))
    best_distance = calculate_total_distance(best_route, cities)
    
    # Yield initial state
    avg_fitness = np.mean([calculate_fitness(route, cities) for route in population])
    yield (best_route.copy(), best_distance, best_route.copy(), best_distance, 0, avg_fitness)
    
    # Evolution loop
    for generation in range(1, iterations + 1):
        # Evaluate fitness of current population
        fitness_scores = [(route, calculate_total_distance(route, cities)) 
                         for route in population]
        fitness_scores.sort(key=lambda x: x[1])  # Sort by distance (lower is better)
        
        # Update overall best
        current_gen_best_route = fitness_scores[0][0]
        current_gen_best_distance = fitness_scores[0][1]
        
        if current_gen_best_distance < best_distance:
            best_route = current_gen_best_route.copy()
            best_distance = current_gen_best_distance
        
        # Create new population
        new_population = []
        
        # ELITISM: Keep the best individuals unchanged
        for i in range(elitism_count):
            new_population.append(fitness_scores[i][0].copy())
        
        # Create offspring to fill rest of population
        while len(new_population) < population_size:
            # SELECTION: Choose two parents using tournament selection
            parent1 = tournament_selection(population, cities)
            parent2 = tournament_selection(population, cities)
            
            # CROSSOVER: Create child from parents
            child = order_crossover(parent1, parent2)
            
            # MUTATION: Randomly modify the child
            if mutation_type == "swap":
                child = mutate_swap(child, mutation_rate)
            elif mutation_type == "inversion":
                child = mutate_inversion(child, mutation_rate)
            
            new_population.append(child)
        
        # Replace old population
        population = new_population
        
        # Calculate average fitness for statistics
        avg_fitness = np.mean([calculate_fitness(route, cities) for route in population])
        
        # Yield current state
        yield (current_gen_best_route.copy(), current_gen_best_distance,
               best_route.copy(), best_distance, generation, avg_fitness)
    
    # Final yield
    yield (best_route.copy(), best_distance, best_route.copy(), best_distance, 
           iterations, avg_fitness)
