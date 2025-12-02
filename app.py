import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from algorithms import (random_search, local_search, hill_climbing, 
                       simulated_annealing, tabu_search, genetic_algorithm,
                       calculate_total_distance)

# Page configuration
st.set_page_config(page_title="TSP Solver", layout="wide")

# Title
st.title("🗺️ Traveling Salesman Problem Solver")

# Sidebar controls
st.sidebar.header("Configuration")

# Algorithm selection
algorithm = st.sidebar.selectbox(
    "Select Algorithm",
    ["Random Search", "Local Search", "Hill Climbing", "Simulated Annealing", 
     "Tabu Search", "Genetic Algorithm"],
    key="algorithm_selector"
)

# Algorithm-specific parameters
opt_type = None
variant = None
initial_temp = None
cooling_rate = None
cooling_schedule = None
tabu_tenure = None
aspiration_enabled = None
population_size = None
mutation_rate = None
elitism_count = None
mutation_type = None

if algorithm == "Local Search":
    opt_type = st.sidebar.radio(
        "Local Search Type",
        ["2-opt", "3-opt"],
        help="2-opt: Reverses route segments | 3-opt: More complex reconnections",
        key="local_search_type"
    )
elif algorithm == "Hill Climbing":
    variant = st.sidebar.radio(
        "Hill Climbing Variant",
        ["steepest", "first"],
        format_func=lambda x: "Steepest Ascent" if x == "steepest" else "First Improvement",
        help="Steepest: Evaluates all neighbors, picks best | First: Accepts first improvement",
        key="hill_climbing_variant"
    )
elif algorithm == "Simulated Annealing":
    st.sidebar.subheader("Annealing Parameters")
    initial_temp = st.sidebar.slider(
        "Initial Temperature", 
        min_value=100.0, 
        max_value=10000.0, 
        value=1000.0, 
        step=100.0,
        help="Higher temperature = more exploration initially",
        key="initial_temp"
    )
    cooling_rate = st.sidebar.slider(
        "Cooling Rate", 
        min_value=0.900, 
        max_value=0.999, 
        value=0.995, 
        step=0.001,
        help="How quickly temperature decreases (closer to 1 = slower cooling)",
        key="cooling_rate"
    )
    cooling_schedule = st.sidebar.radio(
        "Cooling Schedule",
        ["exponential", "linear", "logarithmic"],
        format_func=lambda x: x.capitalize(),
        help="Exponential: Fast initial cooling | Linear: Steady cooling | Logarithmic: Slow cooling",
        key="cooling_schedule"
    )
elif algorithm == "Tabu Search":
    st.sidebar.subheader("Tabu Search Parameters")
    tabu_tenure = st.sidebar.slider(
        "Tabu Tenure",
        min_value=5,
        max_value=50,
        value=10,
        step=1,
        help="Number of iterations a move stays forbidden (higher = more restrictive)",
        key="tabu_tenure"
    )
    aspiration_enabled = st.sidebar.checkbox(
        "Enable Aspiration Criterion",
        value=True,
        help="Allow tabu moves that improve the best solution found so far",
        key="aspiration_enabled"
    )
elif algorithm == "Genetic Algorithm":
    st.sidebar.subheader("Genetic Algorithm Parameters")
    population_size = st.sidebar.slider(
        "Population Size",
        min_value=20,
        max_value=200,
        value=100,
        step=10,
        help="Number of routes in each generation (larger = more diversity)",
        key="population_size"
    )
    mutation_rate = st.sidebar.slider(
        "Mutation Rate",
        min_value=0.001,
        max_value=0.1,
        value=0.01,
        step=0.001,
        format="%.3f",
        help="Probability of random changes (higher = more exploration)",
        key="mutation_rate"
    )
    elitism_count = st.sidebar.slider(
        "Elitism Count",
        min_value=1,
        max_value=10,
        value=2,
        step=1,
        help="Number of best solutions kept unchanged each generation",
        key="elitism_count"
    )
    mutation_type = st.sidebar.radio(
        "Mutation Type",
        ["swap", "inversion"],
        format_func=lambda x: "Swap (exchange two cities)" if x == "swap" else "Inversion (reverse segment)",
        help="Swap: Less disruptive | Inversion: More disruptive",
        key="mutation_type"
    )

# Load dataset
@st.cache_data
def load_cities(file_path):
    """Load city data from CSV."""
    df = pd.read_csv(file_path)
    return df

try:
    cities_df = load_cities("cities.csv")
    st.sidebar.success(f"Loaded {len(cities_df)} cities")
except FileNotFoundError:
    st.error("❌ cities.csv not found! Please make sure it's in the same directory as app.py")
    st.stop()

# Display loaded cities
with st.sidebar.expander("View Cities"):
    st.dataframe(cities_df[['city', 'lat', 'lon']], height=200)

# Algorithm parameters
st.sidebar.header("Parameters")

if algorithm == "Random Search":
    iterations = st.sidebar.slider("Iterations", min_value=100, max_value=50000, value=5000, step=100, key="iterations_random")
elif algorithm == "Local Search":
    iterations = st.sidebar.slider("Max Iterations", min_value=10, max_value=1000, value=100, step=10, key="iterations_local")
    st.sidebar.caption("Local search may stop early if no improvements are found")
elif algorithm == "Hill Climbing":
    iterations = st.sidebar.slider("Max Iterations", min_value=10, max_value=500, value=100, step=10, key="iterations_hill")
    st.sidebar.caption("Hill climbing stops when it reaches a local optimum")
elif algorithm == "Simulated Annealing":
    iterations = st.sidebar.slider("Iterations", min_value=100, max_value=10000, value=2000, step=100, key="iterations_sa")
    st.sidebar.caption("More iterations allow for better convergence")
elif algorithm == "Tabu Search":
    iterations = st.sidebar.slider("Iterations", min_value=100, max_value=5000, value=1000, step=50, key="iterations_tabu")
    st.sidebar.caption("Tabu search explores systematically using memory")
elif algorithm == "Genetic Algorithm":
    iterations = st.sidebar.slider("Generations", min_value=10, max_value=500, value=100, step=10, key="iterations_genetic")
    st.sidebar.caption("Each generation evolves the population")

update_frequency = st.sidebar.slider("Update Frequency (updates during run)", min_value=10, max_value=500, value=50, step=10, key="update_freq")
st.sidebar.caption(f"Will update visualization every ~{iterations//update_frequency} iterations")

# Extract coordinates (using x_km, y_km for accurate distance calculation)
cities_coords = cities_df[['x_km', 'y_km']].values
city_names = cities_df['city'].values

# Run button
run_algorithm = st.sidebar.button("🚀 Run Algorithm", type="primary")

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col2:
    st.subheader("📊 Statistics")
    stat_iteration = st.empty()
    stat_current_distance = st.empty()
    stat_best_distance = st.empty()
    stat_improvement = st.empty()
    if algorithm == "Simulated Annealing":
        stat_temperature = st.empty()
    elif algorithm == "Tabu Search":
        stat_tabu_size = st.empty()
    elif algorithm == "Genetic Algorithm":
        stat_avg_fitness = st.empty()

with col1:
    st.subheader("🗺️ Route Visualization")
    plot_placeholder = st.empty()

# Function to plot the route
def plot_route(cities_coords, city_names, route, title="Current Route", distance=None, temperature=None):
    """Plot the TSP route on a map."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot cities
    ax.scatter(cities_coords[:, 0], cities_coords[:, 1], c='red', s=200, zorder=3, alpha=0.6)
    
    # Add city labels
    for idx, name in enumerate(city_names):
        ax.annotate(name, (cities_coords[idx, 0], cities_coords[idx, 1]), 
                   fontsize=9, ha='center', va='bottom', fontweight='bold')
    
    # Plot route
    if route is not None:
        route_coords = cities_coords[route]
        # Close the loop
        route_coords = np.vstack([route_coords, route_coords[0]])
        ax.plot(route_coords[:, 0], route_coords[:, 1], 'b-', linewidth=2, alpha=0.7, zorder=2)
        
        # Mark start city
        start_city = cities_coords[route[0]]
        ax.scatter(start_city[0], start_city[1], c='green', s=300, marker='*', 
                  zorder=4, label='Start', edgecolors='black', linewidth=2)
    
    ax.set_xlabel('X (km)', fontsize=12)
    ax.set_ylabel('Y (km)', fontsize=12)
    
    title_text = f"{title}"
    if distance:
        title_text += f" - Distance: {distance:.2f} km"
    if temperature is not None:
        title_text += f" | Temp: {temperature:.1f}"
    
    ax.set_title(title_text, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    return fig

# Initial plot (before running algorithm)
if not run_algorithm:
    initial_route = list(range(len(cities_coords)))
    initial_distance = calculate_total_distance(initial_route, cities_coords)
    fig = plot_route(cities_coords, city_names, initial_route, "Initial Route", initial_distance)
    plot_placeholder.pyplot(fig)
    plt.close()
    
    stat_iteration.metric("Iteration", "0")
    stat_current_distance.metric("Current Distance", f"{initial_distance:.2f} km")
    stat_best_distance.metric("Best Distance", f"{initial_distance:.2f} km")
    stat_improvement.metric("Improvement", "0.00%")
    if algorithm == "Simulated Annealing":
        stat_temperature.metric("Temperature", f"{initial_temp:.1f}" if initial_temp else "N/A")
    elif algorithm == "Tabu Search":
        stat_tabu_size.metric("Tabu List Size", "0")
    elif algorithm == "Genetic Algorithm":
        stat_avg_fitness.metric("Avg Population Fitness", "N/A")

# Run the algorithm
if run_algorithm:
    st.sidebar.info("🔄 Algorithm running...")
    
    # Progress bar
    progress_bar = st.sidebar.progress(0)
    
    initial_distance = None
    update_interval = max(1, iterations // update_frequency)  # Update every N iterations
    
    # Run the selected algorithm
    if algorithm == "Random Search":
        algorithm_generator = random_search(cities_coords, iterations)
    elif algorithm == "Local Search":
        algorithm_generator = local_search(cities_coords, iterations, opt_type)
    elif algorithm == "Hill Climbing":
        algorithm_generator = hill_climbing(cities_coords, iterations, variant)
    elif algorithm == "Simulated Annealing":
        algorithm_generator = simulated_annealing(cities_coords, iterations, initial_temp, cooling_rate, cooling_schedule)
    elif algorithm == "Tabu Search":
        algorithm_generator = tabu_search(cities_coords, iterations, tabu_tenure, aspiration_enabled)
    elif algorithm == "Genetic Algorithm":
        algorithm_generator = genetic_algorithm(cities_coords, iterations, population_size, mutation_rate, elitism_count, mutation_type)
    
    for result in algorithm_generator:
        # Handle different return formats
        if algorithm == "Simulated Annealing":
            current_route, current_distance, best_route, best_distance, iteration, temperature = result
            tabu_size = None
            avg_fitness = None
        elif algorithm == "Tabu Search":
            current_route, current_distance, best_route, best_distance, iteration, tabu_size = result
            temperature = None
            avg_fitness = None
        elif algorithm == "Genetic Algorithm":
            current_route, current_distance, best_route, best_distance, iteration, avg_fitness = result
            temperature = None
            tabu_size = None
        else:
            current_route, current_distance, best_route, best_distance, iteration = result
            temperature = None
            tabu_size = None
            avg_fitness = None
        
        # Store initial distance
        if initial_distance is None:
            initial_distance = best_distance
        
        # Only update UI periodically or on last iteration
        if iteration % update_interval == 0 or iteration == iterations:
            # Update progress
            progress = iteration / iterations
            progress_bar.progress(progress)
            
            # Update statistics
            improvement = ((initial_distance - best_distance) / initial_distance) * 100
            
            stat_iteration.metric("Iteration", f"{iteration:,} / {iterations:,}")
            stat_current_distance.metric("Current Distance", f"{current_distance:.2f} km")
            stat_best_distance.metric("Best Distance Found", f"{best_distance:.2f} km")
            stat_improvement.metric("Improvement", f"{improvement:.2f}%")
            
            if algorithm == "Simulated Annealing":
                stat_temperature.metric("Temperature", f"{temperature:.1f}")
            elif algorithm == "Tabu Search":
                stat_tabu_size.metric("Tabu List Size", f"{tabu_size}")
            elif algorithm == "Genetic Algorithm":
                stat_avg_fitness.metric("Avg Population Fitness", f"{avg_fitness:.6f}")
            
            # Update plot with best route
            fig = plot_route(cities_coords, city_names, best_route, "Best Route Found", best_distance, temperature)
            plot_placeholder.pyplot(fig)
            plt.close()
    
    st.sidebar.success("✅ Algorithm completed!")
    st.balloons()
