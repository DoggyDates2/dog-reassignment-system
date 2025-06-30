"""
Configuration settings for the Dog Reassignment System
"""

import streamlit as st

# Application Configuration
APP_CONFIG = {
    "page_title": "🐶 Optimized Dog Reassignment System",
    "page_icon": "🐕",
    "layout": "wide",
    "sidebar_state": "expanded"
}

# Default Data URLs - UPDATE THESE WITH YOUR ACTUAL GOOGLE SHEETS URLS
DEFAULT_URLS = {
    "map_url": "https://docs.google.com/spreadsheets/d/1mg8d5CLxSR54KhNUL8SpL5jzrGN-bghTsC9vxSK8lR0/export?format=csv&gid=267803750",
    "matrix_url": "https://docs.google.com/spreadsheets/d/1421xCS86YH6hx0RcuZCyXkyBK_xl-VDSlXyDNvw09Pg/export?format=csv&gid=398422902"
}

# Optimization Parameters
OPTIMIZATION_CONFIG = {
    "max_domino_depth": 5,
    "eviction_distance_limit": 0.75,
    "fringe_distance_limit": 0.5,
    "reassignment_threshold": 1.5,
    "placement_goal_distance": 0.5,
    "max_chain_length": 3,
    "simulated_annealing_iterations": 1000,
    "cooling_rate": 0.995
}

# UI Configuration
UI_CONFIG = {
    "max_history_entries": 10,
    "default_distance_weight": 1.0,
    "default_balance_weight": 0.2,
    "default_max_distance": 1.5,
    "enable_simulated_annealing": True,
    "enable_constraints": True,
    "chart_color_scheme": "Set2"
}

# Data Validation Rules
VALIDATION_CONFIG = {
    "min_capacity": 1,
    "max_capacity": 20,
    "valid_groups": [1, 2, 3],
    "max_dogs_per_assignment": 10,
    "max_distance_threshold": 10.0
}

def get_secrets_or_default(key: str, default: str) -> str:
    """
    Get value from Streamlit secrets or return default
    Used for hiding sensitive URLs in production
    """
    try:
        if hasattr(st, 'secrets') and 'sheets' in st.secrets:
            return st.secrets['sheets'].get(key, default)
    except:
        pass
    return default

def get_map_url() -> str:
    """Get map data URL from secrets or default"""
    return get_secrets_or_default('map_url', DEFAULT_URLS['map_url'])

def get_matrix_url() -> str:
    """Get matrix data URL from secrets or default"""
    return get_secrets_or_default('matrix_url', DEFAULT_URLS['matrix_url'])
