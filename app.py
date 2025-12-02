import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from algorithms import random_search, calculate_total_distance
import time

# Page configuration
st.set_page_config(page_title="TSP Solver", layout="wide")

# Title
st.title("🗺️ Traveling Salesman Problem Solver")

# Sidebar controls
st.sidebar.header("Configuration")

# Algorithm selection
algorithm = st.sidebar.selectbox(
    "Select Algorithm",
    ["Random Search"]  # We'll add more later
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
iterations = st.sidebar.slider("Iterations", min_value=100, max_value=10000, value=1000, step=100)
animation_speed = st.sidebar.slider("Animation Speed (updates/sec)", min_value=1, max_value=50, value=10)

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

with col1:
    st.subheader("🗺️ Route Visualization")
    plot_placeholder = st.empty()

# Function to plot the route
def plot_route(cities_coords, city_names, route, title="Current Route", distance=None):
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
    ax.set_title(f"{title}" + (f" - Distance: {distance:.2f} km" if distance else ""), fontsize=14, fontweight='bold')
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

# Run the algorithm
if run_algorithm:
    st.sidebar.info("🔄 Algorithm running...")
    
    # Progress bar
    progress_bar = st.sidebar.progress(0)
    
    initial_distance = None
    update_interval = 1.0 / animation_speed  # Time between updates
    last_update_time = 0
    
    # Run the selected algorithm
    if algorithm == "Random Search":
        algorithm_generator = random_search(cities_coords, iterations)
    
    for current_route, current_distance, best_route, best_distance, iteration in algorithm_generator:
        # Store initial distance
        if initial_distance is None:
            initial_distance = best_distance
        
        # Update progress
        progress = iteration / iterations
        progress_bar.progress(progress)
        
        # Control update frequency
        current_time = time.time()
        if current_time - last_update_time >= update_interval or iteration == iterations:
            # Update statistics
            improvement = ((initial_distance - best_distance) / initial_distance) * 100
            
            stat_iteration.metric("Iteration", f"{iteration:,} / {iterations:,}")
            stat_current_distance.metric("Current Distance", f"{current_distance:.2f} km")
            stat_best_distance.metric("Best Distance Found", f"{best_distance:.2f} km")
            stat_improvement.metric("Improvement", f"{improvement:.2f}%")
            
            # Update plot with best route
            fig = plot_route(cities_coords, city_names, best_route, "Best Route Found", best_distance)
            plot_placeholder.pyplot(fig)
            plt.close()
            
            last_update_time = current_time
    
    st.sidebar.success("✅ Algorithm completed!")
    st.balloons()
