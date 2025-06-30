"""
Optimization Engine Module
Core algorithms for finding optimal dog reassignments
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import itertools
from scipy.optimize import linear_sum_assignment

@dataclass
class AssignmentSolution:
    """Represents a complete assignment solution with quality metrics"""
    assignments: Dict[str, str]  # dog_id -> driver_name
    total_distance: float
    load_balance_score: float
    constraint_violations: int
    strategy_used: str
    
    def quality_score(self) -> float:
        """Calculate overall quality score (lower is better)"""
        distance_weight = 1.0
        balance_weight = 0.2
        violation_penalty = 10.0
        
        return (self.total_distance * distance_weight + 
                self.load_balance_score * balance_weight + 
                self.constraint_violations * violation_penalty)

class HungarianOptimizer:
    """Hungarian Algorithm implementation for minimum cost assignment"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
    
    def optimize(self, dogs_to_reassign: List) -> Optional[AssignmentSolution]:
        """Find optimal assignment using Hungarian algorithm"""
        if not dogs_to_reassign:
            return None
        
        available_drivers = self._get_available_drivers(dogs_to_reassign)
        if not available_drivers:
            return None
        
        # Create cost matrix
        cost_matrix = self._build_cost_matrix(dogs_to_reassign, available_drivers)
        
        # Solve assignment problem
        try:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            assignments = {}
            total_distance = 0
            
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row][col] != np.inf:
                    dog = dogs_to_reassign[row]
                    driver_name = available_drivers[col]
                    assignments[dog.dog_id] = driver_name
                    total_distance += cost_matrix[row][col]
            
            if assignments:
                return AssignmentSolution(
                    assignments=assignments,
                    total_distance=total_distance,
                    load_balance_score=self._calculate_load_balance(assignments),
                    constraint_violations=self._count_violations(assignments),
                    strategy_used="Hungarian Algorithm"
                )
        
        except Exception:
            pass
        
        return None
    
    def _get_available_drivers(self, dogs_to_reassign: List) -> List[str]:
        """Get drivers that can potentially accept reassigned dogs"""
        available = []
        
        for driver_name, driver in self.data_manager.drivers.items():
            # Skip drivers that are completely called out
            if len(driver.callouts) >= 3:
                continue
            
            # Check if driver can accept at least one dog
            can_accept_any = False
            for dog in dogs_to_reassign:
                if not any(group in driver.callouts for group in dog.groups):
                    can_accept_any = True
                    break
            
            if can_accept_any:
                available.append(driver_name)
        
        return available
    
    def _build_cost_matrix(self, dogs: List, drivers: List[str]) -> np.ndarray:
        """Build cost matrix for Hungarian algorithm"""
        n_dogs = len(dogs)
        n_drivers = len(drivers)
        
        # Create square matrix (pad with dummy assignments if needed)
        size = max(n_dogs, n_drivers)
        cost_matrix = np.full((size, size), np.inf)
        
        for i, dog in enumerate(dogs):
            for j, driver_name in enumerate(drivers):
                driver = self.data_manager.drivers[driver_name]
                
                # Check if driver can accept this dog
                if self._can_driver_accept(driver, dog):
                    # Calculate assignment cost
                    distance_cost = self._calculate_distance_cost(dog, driver_name)
                    capacity_penalty = self._calculate_capacity_penalty(dog, driver_name)
                    group_penalty = self._calculate_group_penalty(dog, driver)
                    
                    total_cost = distance_cost + capacity_penalty + group_penalty
                    cost_matrix[i][j] = total_cost
        
        return cost_matrix
    
    def _can_driver_accept(self, driver, dog) -> bool:
        """Check if driver can theoretically accept dog"""
        # Check callouts
        if any(group in driver.callouts for group in dog.groups):
            return False
        
        # Check basic capacity
        current_loads = self.data_manager.get_driver_current_loads(driver.name)
        for group in dog.groups:
            available = driver.get_capacity(group) - current_loads.get(group, 0)
            if available < dog.num_dogs:
                return False
        
        return True
    
    def _calculate_distance_cost(self, dog, driver_name: str) -> float:
        """Calculate distance cost for assigning dog to driver"""
        if not self.data_manager.distance_matrix:
            return 1.0
        
        driver_dogs = self.data_manager.get_dogs_for_driver(driver_name)
        if not driver_dogs:
            return 0.5  # Small cost for new driver
        
        min_distance = float('inf')
        for other_dog_id in driver_dogs:
            distance = self.data_manager.distance_matrix.get_distance(dog.dog_id, other_dog_id)
            min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else 1.0
    
    def _calculate_capacity_penalty(self, dog, driver_name: str) -> float:
        """Calculate penalty for capacity utilization"""
        driver = self.data_manager.drivers[driver_name]
        current_loads = self.data_manager.get_driver_current_loads(driver_name)
        
        penalty = 0
        for group in dog.groups:
            utilization = (current_loads.get(group, 0) + dog.num_dogs) / driver.get_capacity(group)
            if utilization > 0.8:  # High utilization penalty
                penalty += (utilization - 0.8) * 2
        
        return penalty
    
    def _calculate_group_penalty(self, dog, driver) -> float:
        """Calculate penalty for group mismatches"""
        penalty = 0
        for group in dog.groups:
            if group in driver.callouts:
                penalty += 100  # High penalty for callout violations
        return penalty
    
    def _calculate_load_balance(self, assignments: Dict[str, str]) -> float:
        """Calculate load balance score across all drivers"""
        driver_loads = defaultdict(list)
        
        for dog_id, driver_name in assignments.items():
            if dog_id in self.data_manager.dogs:
                dog = self.data_manager.dogs[dog_id]
                current_loads = self.data_manager.get_driver_current_loads(driver_name)
                total_load = sum(current_loads.values()) + dog.num_dogs
                driver_loads[driver_name].append(total_load)
        
        if not driver_loads:
            return 0
        
        all_loads = [sum(loads) for loads in driver_loads.values()]
        return np.std(all_loads) if len(all_loads) > 1 else 0
    
    def _count_violations(self, assignments: Dict[str, str]) -> int:
        """Count constraint violations in assignment"""
        violations = 0
        
        for dog_id, driver_name in assignments.items():
            if dog_id not in self.data_manager.dogs or driver_name not in self.data_manager.drivers:
                violations += 1
                continue
            
            dog = self.data_manager.dogs[dog_id]
            driver = self.data_manager.drivers[driver_name]
            
            # Check callout violations
            if any(group in driver.callouts for group in dog.groups):
                violations += 1
        
        return violations

class SimulatedAnnealingOptimizer:
    """Simulated Annealing optimizer for global solution search"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
    
    def optimize(self, dogs_to_reassign: List, 
                initial_solution: Optional[AssignmentSolution] = None) -> Optional[AssignmentSolution]:
        """Optimize using simulated annealing"""
        if not dogs_to_reassign:
            return None
        
        # Get initial solution
        current_solution = initial_solution or self._generate_random_solution(dogs_to_reassign)
        if not current_solution:
            return None
        
        best_solution = current_solution
        temperature = 1.0
        cooling_rate = 0.995
        max_iterations = 1000
        
        for iteration in range(max_iterations):
            # Generate neighbor solution
            neighbor = self._generate_neighbor(current_solution, dogs_to_reassign)
            if not neighbor:
                continue
            
            current_quality = current_solution.quality_score()
            neighbor_quality = neighbor.quality_score()
            
            # Accept or reject neighbor
            delta = neighbor_quality - current_quality
            
            if delta < 0 or (temperature > 0 and np.random.random() < np.exp(-delta / temperature)):
                current_solution = neighbor
                
                if neighbor_quality < best_solution.quality_score():
                    best_solution = neighbor
            
            temperature *= cooling_rate
            
            # Early termination if temperature is very low
            if temperature < 1e-6:
                break
        
        best_solution.strategy_used = "Simulated Annealing"
        return best_solution
    
    def _generate_random_solution(self, dogs_to_reassign: List) -> Optional[AssignmentSolution]:
        """Generate a random valid solution"""
        assignments = {}
        
        for dog in dogs_to_reassign:
            valid_drivers = self._get_valid_drivers_for_dog(dog)
            if valid_drivers:
                assignments[dog.dog_id] = np.random.choice(valid_drivers)
        
        if assignments:
            return self._create_solution_from_assignments(assignments, "Random Initial")
        
        return None
    
    def _generate_neighbor(self, solution: AssignmentSolution, dogs: List) -> Optional[AssignmentSolution]:
        """Generate a neighbor solution by modifying current assignment"""
        if len(solution.assignments) < 2:
            return None
        
        # Strategy: Reassign single dog
        dog_ids = list(solution.assignments.keys())
        dog_id = np.random.choice(dog_ids)
        dog = next((d for d in dogs if d.dog_id == dog_id), None)
        
        if not dog:
            return None
        
        valid_drivers = self._get_valid_drivers_for_dog(dog)
        current_driver = solution.assignments[dog_id]
        
        # Remove current driver to force a different assignment
        valid_drivers = [d for d in valid_drivers if d != current_driver]
        
        if valid_drivers:
            new_driver = np.random.choice(valid_drivers)
            new_assignments = dict(solution.assignments)
            new_assignments[dog_id] = new_driver
            return self._create_solution_from_assignments(new_assignments, solution.strategy_used)
        
        return None
    
    def _get_valid_drivers_for_dog(self, dog) -> List[str]:
        """Get list of drivers that can accept this dog"""
        valid = []
        
        for driver_name, driver in self.data_manager.drivers.items():
            if driver_name == getattr(dog, 'current_driver', None):
                continue
            
            # Check callouts
            if any(group in driver.callouts for group in dog.groups):
                continue
            
            # Check capacity (simplified)
            current_loads = self.data_manager.get_driver_current_loads(driver_name)
            can_fit = True
            
            for group in dog.groups:
                available = driver.get_capacity(group) - current_loads.get(group, 0)
                if available < dog.num_dogs:
                    can_fit = False
                    break
            
            if can_fit:
                valid.append(driver_name)
        
        return valid
    
    def _create_solution_from_assignments(self, assignments: Dict[str, str], strategy: str) -> AssignmentSolution:
        """Create AssignmentSolution from assignments dict"""
        total_distance = 0
        
        for dog_id, driver_name in assignments.items():
            if dog_id in self.data_manager.dogs:
                distance = self._calculate_assignment_distance(dog_id, driver_name)
                total_distance += distance
        
        return AssignmentSolution(
            assignments=assignments,
            total_distance=total_distance,
            load_balance_score=self._calculate_load_balance(assignments),
            constraint_violations=self._count_violations(assignments),
            strategy_used=strategy
        )
    
    def _calculate_assignment_distance(self, dog_id: str, driver_name: str) -> float:
        """Calculate distance cost for a single assignment"""
        if not self.data_manager.distance_matrix:
            return 1.0
        
        driver_dogs = self.data_manager.get_dogs_for_driver(driver_name)
        if not driver_dogs:
            return 0.5
        
        min_distance = float('inf')
        for other_dog_id in driver_dogs:
            if other_dog_id != dog_id:
                distance = self.data_manager.distance_matrix.get_distance(dog_id, other_dog_id)
                min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else 1.0
    
    def _calculate_load_balance(self, assignments: Dict[str, str]) -> float:
        """Calculate load balance score"""
        # Simplified load balance calculation
        driver_loads = defaultdict(int)
        
        for dog_id, driver_name in assignments.items():
            if dog_id in self.data_manager.dogs:
                dog = self.data_manager.dogs[dog_id]
                driver_loads[driver_name] += dog.num_dogs
        
        loads = list(driver_loads.values())
        return np.std(loads) if len(loads) > 1 else 0
    
    def _count_violations(self, assignments: Dict[str, str]) -> int:
        """Count constraint violations"""
        violations = 0
        
        for dog_id, driver_name in assignments.items():
            if dog_id not in self.data_manager.dogs or driver_name not in self.data_manager.drivers:
                violations += 1
                continue
            
            dog = self.data_manager.dogs[dog_id]
            driver = self.data_manager.drivers[driver_name]
            
            # Check callouts
            if any(group in driver.callouts for group in dog.groups):
                violations += 1
        
        return violations

class MultiAlgorithmOptimizer:
    """Combines multiple optimization strategies to find the best solution"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.hungarian = HungarianOptimizer(data_manager)
        self.simulated_annealing = SimulatedAnnealingOptimizer(data_manager)
    
    def optimize(self, dogs_to_reassign: List) -> Optional[AssignmentSolution]:
        """Run multiple optimization strategies and return the best solution"""
        if not dogs_to_reassign:
            return None
        
        solutions = []
        
        # Strategy 1: Hungarian Algorithm
        try:
            hungarian_solution = self.hungarian.optimize(dogs_to_reassign)
            if hungarian_solution:
                solutions.append(hungarian_solution)
        except Exception:
            pass
        
        # Strategy 2: Simulated Annealing with Hungarian initial solution
        try:
            initial = hungarian_solution if 'hungarian_solution' in locals() else None
            sa_solution = self.simulated_annealing.optimize(dogs_to_reassign, initial)
            if sa_solution:
                solutions.append(sa_solution)
        except Exception:
            pass
        
        # Return best solution
        if solutions:
            best_solution = min(solutions, key=lambda s: s.quality_score())
            best_solution.strategy_used = f"Multi-Algorithm ({best_solution.strategy_used})"
            return best_solution
        
        return None
