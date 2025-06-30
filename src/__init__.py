"""
"""
Optimized Dog Reassignment System
Source package initialization
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Multi-algorithm optimization system for dog assignment logistics"

# Import main components for easy access
try:
    from .data_manager import DataManager, DogProfile, DriverProfile, DistanceMatrix, create_data_manager
    from .optimization_engine import MultiAlgorithmOptimizer, AssignmentSolution
    from .assignment_logic import AssignmentEngine, ReassignmentResult, create_assignment_engine
    from .ui_components import SessionState, DashboardUI

    __all__ = [
        'DataManager', 'DogProfile', 'DriverProfile', 'DistanceMatrix', 'create_data_manager',
        'MultiAlgorithmOptimizer', 'AssignmentSolution',
        'AssignmentEngine', 'ReassignmentResult', 'create_assignment_engine',
        'SessionState', 'DashboardUI'
    ]
except ImportError:
    # Files not uploaded yet, that's OK
    pass
