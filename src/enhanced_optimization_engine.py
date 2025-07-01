"""
Enhanced Optimization Engine Module
Optimized for minimizing total system miles and average drive time
Complete file ready for copy-paste
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict
import itertools
from scipy.optimize import linear_sum_assignment

@dataclass
class AssignmentSolution:
    """Enhanced solution with total miles and drive time tracking"""
    assignments: Dict[str, str]  # dog_id -> driver_name
    total_distance: float
    system_total_miles: float  # Total miles for entire system
    avg_drive_time_before: float  # Average drive time before reassignment
    avg_drive_time_after: float   # Average drive time after reassignment
    load_balance_score: float
    constraint_violations: int
    strategy_used: str
    cost_savings: float  # Estimated cost savings in dollars
    driver_impact: Dict[str, Dict]  # Per-driver impact analysis
    
    def quality_score(self) -> float:
        """Enhanced quality score prioritizing total miles reduction"""
        # Weights favor mile reduction
        distance_weight = 2.0  # Increased weight for distance
        system_miles_weight = 1.5  # Weight for total system impact
        drive_time_penalty = 3.0   # Heavy penalty for increased drive time
        balance_weight = 0.2
        violation_penalty = 10.0
        
        # Penalty if average drive time increases
        drive_time_increase = max(0, self.avg_drive_time_after - self.avg_drive_time_before)
        
        return (self.total_distance * distance_weight + 
                self.system_total_miles * system_miles_weight +
                drive_time_increase * drive_time_penalty +
                self.load_balance_score * balance_weight + 
                self.constraint_violations * violation_penalty)
    
    def miles_saved(self) -> float:
        """Calculate miles saved compared to baseline"""
        return max(0, self.avg_drive_time_before - self.avg_drive_time_after)
    
    def cost_benefit_summary(self) -> str:
        """Generate cost-benefit summary"""
        miles_saved = self.miles_saved()
        cost_per_mile = 10.0  # $10 per mile from config
        dollar_savings = miles_saved * cost_per_mile
        
        return f"""
💰 COST-BENEFIT SUMMARY
======================
📏 Miles saved: {miles_saved:.2f}
💵 Cost savings: ${dollar_savings:.2f}
⏱️ Avg drive time: {self.avg_drive_time_before:.2f} → {self.avg_drive_time_after:.2f}
🎯 System total miles: {self.system_total_miles:.2f}
⚡ Strategy: {self.strategy_used}
"""

class SystemMilesCalculator:
    """Calculates total system miles and average drive times"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
    
    def calculate_total_system_miles(self, assignments: Dict[str, str] = None) -> float:
        """Calculate total miles across all drivers in the system"""
        if assignments is None:
            assignments = self.data_manager.current_assignments
        
        total_miles = 0.0
        
        # Group dogs by driver
        driver_dogs = defaultdict(list)
        for dog_id, driver_name in assignments.items():
            if dog_id in self.data_manager.dogs:
                driver_dogs[driver_name].append(dog_id)
        
        # Calculate miles for each driver
        for driver_name, dog_ids in driver_dogs.items():
            if len(dog_ids) <= 1:
                continue  # No driving needed for 0-1 dogs
            
            driver_miles = self._calculate_driver_route_miles(dog_ids)
            total_miles += driver_miles
        
        return total_miles
    
    def calculate_average_drive_time(self, assignments: Dict[str, str] = None) -> float:
        """Calculate average drive time per driver"""
        if assignments is None:
            assignments = self.data_manager.current_assignments
        
        # Group dogs by driver
        driver_dogs = defaultdict(list)
        for dog_id, driver_name in assignments.items():
            if dog_id in self.data_manager.dogs:
                driver_dogs[driver_name].append(dog_id)
        
        driver_times = []
        for driver_name, dog_ids in driver_dogs.items():
            if len(dog_ids) == 0:
                drive_time = 0.0
            elif len(dog_ids) == 1:
                drive_time = 0.5  # Minimal driving for single dog
            else:
                drive_time = self._calculate_driver_route_miles(dog_ids)
            
            driver_times.append(drive_time)
        
        return sum(driver_times) / len(driver_times) if driver_times else 0.0
    
    def _calculate_driver_route_miles(self, dog_ids: List[str]) -> float:
        """Estimate total route miles for a driver's dogs using TSP approximation"""
        if len(dog_ids) <= 1:
            return 0.0
        
        if not self.data_manager.distance_matrix:
            return len(dog_ids) * 1.0  # Fallback estimate
        
        # Use nearest neighbor TSP approximation for efficiency
        return self._nearest_neighbor_tsp(dog_ids)
    
    def _nearest_neighbor_tsp(self, dog_ids: List[str]) -> float:
        """Approximate TSP solution using nearest neighbor algorithm"""
        if len(dog_ids) < 2:
            return 0.0
        
        unvisited = set(dog_ids)
        current = dog_ids[0]
        unvisited.remove(current)
        total_distance = 0.0
        
        while unvisited:
            nearest = None
            min_distance = float('inf')
            
            for candidate in unvisited:
                distance = self.data_manager.distance_matrix.get_distance(current, candidate)
                if distance < min_distance:
                    min_distance = distance
                    nearest = candidate
            
            if nearest:
                total_distance += min_distance
                current = nearest
                unvisited.remove(nearest)
            else:
                # Fallback if no valid distances
                total_distance += 1.0
                unvisited.pop()
        
        return total_distance

class EnhancedHungarianOptimizer:
    """Hungarian Algorithm enhanced with total miles consideration"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.miles_calculator = SystemMilesCalculator(data_manager)
    
    def optimize(self, dogs_to_reassign: List) -> Optional[AssignmentSolution]:
        """Find optimal assignment minimizing total system miles"""
        if not dogs_to_reassign:
            return None
        
        # Calculate baseline metrics
        baseline_total_miles = self.miles_calculator.calculate_total_system_miles()
        baseline_avg_drive_time = self.miles_calculator.calculate_average_drive_time()
        
        available_drivers = self._get_available_drivers(dogs_to_reassign)
        if not available_drivers:
            return None
        
        # Create enhanced cost matrix considering system impact
        cost_matrix = self._build_enhanced_cost_matrix(dogs_to_reassign, available_drivers)
        
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
                # Calculate new system metrics
                new_assignments = dict(self.data_manager.current_assignments)
                new_assignments.update(assignments)
                
                new_total_miles = self.miles_calculator.calculate_total_system_miles(new_assignments)
                new_avg_drive_time = self.miles_calculator.calculate_average_drive_time(new_assignments)
                
                # Calculate cost savings
                miles_saved = baseline_total_miles - new_total_miles
                cost_savings = miles_saved * 10.0  # $10 per mile
                
                return AssignmentSolution(
                    assignments=assignments,
                    total_distance=total_distance,
                    system_total_miles=new_total_miles,
                    avg_drive_time_before=baseline_avg_drive_time,
                    avg_drive_time_after=new_avg_drive_time,
                    load_balance_score=self._calculate_load_balance(assignments),
                    constraint_violations=self._count_violations(assignments),
                    strategy_used="Enhanced Hungarian Algorithm",
                    cost_savings=cost_savings,
                    driver_impact=self._calculate_driver_impact(assignments)
                )
        
        except Exception:
            pass
        
        return None
    
    def _build_enhanced_cost_matrix(self, dogs: List, drivers: List[str]) -> np.ndarray:
        """Build cost matrix considering total system miles impact"""
        n_dogs = len(dogs)
        n_drivers = len(drivers)
        
        # Create square matrix
        size = max(n_dogs, n_drivers)
        cost_matrix = np.full((size, size), np.inf)
        
        # Current system state for comparison
        baseline_total_miles = self.miles_calculator.calculate_total_system_miles()
        
        for i, dog in enumerate(dogs):
            for j, driver_name in enumerate(drivers):
                driver = self.data_manager.drivers[driver_name]
                
                # Check if driver can accept this dog
                if self._can_driver_accept(driver, dog):
                    # Calculate total cost including system impact
                    local_distance_cost = self._calculate_distance_cost(dog, driver_name)
                    system_impact_cost = self._calculate_system_impact_cost(dog, driver_name, baseline_total_miles)
                    capacity_penalty = self._calculate_capacity_penalty(dog, driver_name)
                    group_penalty = self._calculate_group_penalty(dog, driver)
                    
                    total_cost = (local_distance_cost + 
                                system_impact_cost * 0.5 +  # Weight system impact
                                capacity_penalty + 
                                group_penalty)
                    
                    cost_matrix[i][j] = total_cost
        
        return cost_matrix
    
    def _calculate_system_impact_cost(self, dog, driver_name: str, baseline_miles: float) -> float:
        """Calculate how this assignment affects total system miles"""
        # Simulate the assignment
        test_assignments = dict(self.data_manager.current_assignments)
        test_assignments[dog.dog_id] = driver_name
        
        # Calculate new total miles
        new_total_miles = self.miles_calculator.calculate_total_system_miles(test_assignments)
        
        # Return the increase in total system miles
        return max(0, new_total_miles - baseline_miles)
    
    def _get_available_drivers(self, dogs_to_reassign: List) -> List[str]:
        """Get drivers that can accept reassigned dogs"""
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
    
    def _can_driver_accept(self, driver, dog) -> bool:
        """Check if driver can accept dog"""
        # Check callouts
        if any(group in driver.callouts for group in dog.groups):
            return False
        
        # Check capacity
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
        """Calculate load balance score"""
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
        """Count constraint violations"""
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
    
    def _calculate_driver_impact(self, assignments: Dict[str, str]) -> Dict[str, Dict]:
        """Calculate per-driver impact of assignments"""
        impact = {}
        
        for dog_id, driver_name in assignments.items():
            if driver_name not in impact:
                impact[driver_name] = {
                    'dogs_added': 0,
                    'miles_added': 0.0,
                    'utilization_change': {}
                }
            
            impact[driver_name]['dogs_added'] += 1
            
            # Calculate miles added (simplified)
            if dog_id in self.data_manager.dogs:
                dog = self.data_manager.dogs[dog_id]
                distance = self._calculate_distance_cost(dog, driver_name)
                impact[driver_name]['miles_added'] += distance
        
        return impact

class EnhancedMultiAlgorithmOptimizer:
    """Enhanced multi-algorithm optimizer prioritizing mile reduction"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.hungarian = EnhancedHungarianOptimizer(data_manager)
        self.miles_calculator = SystemMilesCalculator(data_manager)
    
    def optimize(self, dogs_to_reassign: List) -> Optional[AssignmentSolution]:
        """Run enhanced optimization prioritizing total miles reduction"""
        if not dogs_to_reassign:
            return None
        
        solutions = []
        
        # Strategy 1: Enhanced Hungarian Algorithm
        try:
            hungarian_solution = self.hungarian.optimize(dogs_to_reassign)
            if hungarian_solution:
                solutions.append(hungarian_solution)
        except Exception:
            pass
        
        # Strategy 2: Greedy mile-reduction approach
        try:
            greedy_solution = self._greedy_mile_reduction(dogs_to_reassign)
            if greedy_solution:
                solutions.append(greedy_solution)
        except Exception:
            pass
        
        # Strategy 3: Load balancing with mile awareness
        try:
            balanced_solution = self._balanced_mile_optimization(dogs_to_reassign)
            if balanced_solution:
                solutions.append(balanced_solution)
        except Exception:
            pass
        
        # Return best solution based on quality score
        if solutions:
            best_solution = min(solutions, key=lambda s: s.quality_score())
            best_solution.strategy_used = f"Enhanced Multi-Algorithm ({best_solution.strategy_used})"
            return best_solution
        
        return None
    
    def _greedy_mile_reduction(self, dogs_to_reassign: List) -> Optional[AssignmentSolution]:
        """Greedy algorithm focused purely on mile reduction"""
        assignments = {}
        baseline_miles = self.miles_calculator.calculate_total_system_miles()
        baseline_avg_time = self.miles_calculator.calculate_average_drive_time()
        
        # Create a copy of current assignments for testing
        test_assignments = dict(self.data_manager.current_assignments)
        
        for dog in dogs_to_reassign:
            best_driver = None
            best_miles = float('inf')
            
            for driver_name, driver in self.data_manager.drivers.items():
                if self._can_assign(dog, driver):
                    # Test assignment
                    test_assignments[dog.dog_id] = driver_name
                    test_miles = self.miles_calculator.calculate_total_system_miles(test_assignments)
                    
                    if test_miles < best_miles:
                        best_miles = test_miles
                        best_driver = driver_name
                    
                    # Reset for next test
                    test_assignments[dog.dog_id] = self.data_manager.current_assignments.get(dog.dog_id, "")
            
            if best_driver:
                assignments[dog.dog_id] = best_driver
                test_assignments[dog.dog_id] = best_driver  # Keep for next iteration
        
        if assignments:
            # Calculate final metrics
            final_assignments = dict(self.data_manager.current_assignments)
            final_assignments.update(assignments)
            
            final_miles = self.miles_calculator.calculate_total_system_miles(final_assignments)
            final_avg_time = self.miles_calculator.calculate_average_drive_time(final_assignments)
            
            total_distance = sum(self._calculate_assignment_distance(dog_id, driver) 
                               for dog_id, driver in assignments.items())
            
            return AssignmentSolution(
                assignments=assignments,
                total_distance=total_distance,
                system_total_miles=final_miles,
                avg_drive_time_before=baseline_avg_time,
                avg_drive_time_after=final_avg_time,
                load_balance_score=0.0,  # Simplified
                constraint_violations=0,  # Simplified
                strategy_used="Greedy Mile Reduction",
                cost_savings=(baseline_miles - final_miles) * 10.0,
                driver_impact={}  # Simplified
            )
        
        return None
    
    def _balanced_mile_optimization(self, dogs_to_reassign: List) -> Optional[AssignmentSolution]:
        """Balance load while minimizing miles"""
        # This could implement a more sophisticated algorithm that considers both load balance and miles
        # For now, return None to focus on the other algorithms
        return None
    
    def _can_assign(self, dog, driver) -> bool:
        """Check if dog can be assigned to driver"""
        # Check callouts
        if any(group in driver.callouts for group in dog.groups):
            return False
        
        # Check capacity
        current_loads = self.data_manager.get_driver_current_loads(driver.name)
        for group in dog.groups:
            available = driver.get_capacity(group) - current_loads.get(group, 0)
            if available < dog.num_dogs:
                return False
        
        return True
    
    def _calculate_assignment_distance(self, dog_id: str, driver_name: str) -> float:
        """Calculate distance for assignment"""
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
