"""
Assignment Logic Module
Clean, understandable assignment logic replacing complex fringe moves and domino evictions
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
from optimization_engine import MultiAlgorithmOptimizer, AssignmentSolution

@dataclass
class ReassignmentResult:
    """Results from a reassignment operation"""
    assignments: List[Dict]
    quality_score: float
    total_distance: float
    dogs_affected: int
    strategy_used: str
    success: bool
    error_message: Optional[str] = None

class AssignmentEngine:
    """
    Main assignment engine that coordinates optimization and produces results
    """
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.optimizer = MultiAlgorithmOptimizer(data_manager)
    
    def reassign_dogs(self, called_out_driver: str, affected_groups: List[int]) -> ReassignmentResult:
        """
        Main entry point for dog reassignment
        
        Args:
            called_out_driver: Name of driver being called out
            affected_groups: List of group numbers affected by callout
            
        Returns:
            ReassignmentResult with assignments and metrics
        """
        try:
            # Update driver callout status
            self._update_driver_callouts(called_out_driver, affected_groups)
            
            # Find dogs that need reassignment
            dogs_to_reassign = self._identify_dogs_to_reassign(called_out_driver, affected_groups)
            
            if not dogs_to_reassign:
                return ReassignmentResult(
                    assignments=[],
                    quality_score=0.0,
                    total_distance=0.0,
                    dogs_affected=0,
                    strategy_used="None needed",
                    success=True,
                    error_message="No dogs need reassignment"
                )
            
            # Run optimization
            solution = self.optimizer.optimize(dogs_to_reassign)
            
            if not solution:
                return ReassignmentResult(
                    assignments=[],
                    quality_score=float('inf'),
                    total_distance=float('inf'),
                    dogs_affected=len(dogs_to_reassign),
                    strategy_used="Failed",
                    success=False,
                    error_message="No valid reassignment solution found"
                )
            
            # Convert solution to assignment format
            assignments = self._solution_to_assignments(solution, dogs_to_reassign)
            
            # Update internal assignments if successful
            if assignments:
                self._apply_assignments(solution.assignments)
            
            return ReassignmentResult(
                assignments=assignments,
                quality_score=solution.quality_score(),
                total_distance=solution.total_distance,
                dogs_affected=len(assignments),
                strategy_used=solution.strategy_used,
                success=True
            )
            
        except Exception as e:
            return ReassignmentResult(
                assignments=[],
                quality_score=float('inf'),
                total_distance=float('inf'),
                dogs_affected=0,
                strategy_used="Error",
                success=False,
                error_message=f"Reassignment failed: {str(e)}"
            )
    
    def _update_driver_callouts(self, driver_name: str, groups: List[int]):
        """Update driver callout status"""
        if driver_name in self.data_manager.drivers:
            driver = self.data_manager.drivers[driver_name]
            for group in groups:
                driver.set_callout(group, True)
    
    def _identify_dogs_to_reassign(self, driver_name: str, groups: List[int]) -> List:
        """Identify dogs that need reassignment due to callout"""
        dogs_to_reassign = []
        
        for dog_id, assigned_driver in self.data_manager.current_assignments.items():
            if assigned_driver != driver_name:
                continue
            
            if dog_id not in self.data_manager.dogs:
                continue
            
            dog = self.data_manager.dogs[dog_id]
            
            # Check if any of the dog's groups are affected by callout
            if any(group in groups for group in dog.groups):
                dogs_to_reassign.append(dog)
        
        return dogs_to_reassign
    
    def _solution_to_assignments(self, solution: AssignmentSolution, dogs_to_reassign: List) -> List[Dict]:
        """Convert optimization solution to assignment format"""
        assignments = []
        
        for dog in dogs_to_reassign:
            if dog.dog_id in solution.assignments:
                new_driver = solution.assignments[dog.dog_id]
                distance = self._calculate_assignment_distance(dog, new_driver)
                
                assignments.append({
                    "Dog ID": dog.dog_id,
                    "Dog Name": dog.name,
                    "New Assignment": f"{new_driver}:{'&'.join(map(str, sorted(dog.groups)))}",
                    "New Driver": new_driver,
                    "Groups": sorted(list(dog.groups)),
                    "Distance": round(distance, 3),
                    "Number of Dogs": dog.num_dogs,
                    "Previous Driver": self.data_manager.current_assignments.get(dog.dog_id, "Unknown")
                })
        
        return assignments
    
    def _calculate_assignment_distance(self, dog, driver_name: str) -> float:
        """Calculate distance for assignment"""
        if not self.data_manager.distance_matrix:
            return 1.0
        
        driver_dogs = self.data_manager.get_dogs_for_driver(driver_name)
        if not driver_dogs:
            return 0.5  # Default distance for new driver
        
        min_distance = float('inf')
        for other_dog_id in driver_dogs:
            if other_dog_id != dog.dog_id:
                distance = self.data_manager.distance_matrix.get_distance(dog.dog_id, other_dog_id)
                min_distance = min(min_distance, distance)
        
        return min_distance if min_distance != float('inf') else 1.0
    
    def _apply_assignments(self, assignments: Dict[str, str]):
        """Apply assignments to data manager state"""
        for dog_id, new_driver in assignments.items():
            if dog_id in self.data_manager.current_assignments:
                self.data_manager.current_assignments[dog_id] = new_driver

class CapacityAnalyzer:
    """Analyzes driver capacity and load distribution"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
    
    def get_capacity_summary(self) -> Dict:
        """Get summary of driver capacities and utilization"""
        summary = {
            'drivers': {},
            'total_capacity': 0,
            'total_load': 0,
            'utilization': 0.0,
            'overloaded_drivers': [],
            'available_capacity': {}
        }
        
        for driver_name, driver in self.data_manager.drivers.items():
            current_loads = self.data_manager.get_driver_current_loads(driver_name)
            
            driver_summary = {
                'capacities': driver.group_capacities.copy(),
                'current_loads': current_loads,
                'utilization': {},
                'available': {},
                'is_overloaded': False,
                'callouts': list(driver.callouts)
            }
            
            for group in [1, 2, 3]:
                capacity = driver.get_capacity(group)
                load = current_loads.get(group, 0)
                utilization = (load / capacity * 100) if capacity > 0 else 0
                available = max(0, capacity - load)
                
                driver_summary['utilization'][group] = utilization
                driver_summary['available'][group] = available
                
                if load > capacity:
                    driver_summary['is_overloaded'] = True
                
                summary['total_capacity'] += capacity
                summary['total_load'] += load
            
            if driver_summary['is_overloaded']:
                summary['overloaded_drivers'].append(driver_name)
            
            summary['drivers'][driver_name] = driver_summary
        
        # Calculate overall utilization
        if summary['total_capacity'] > 0:
            summary['utilization'] = (summary['total_load'] / summary['total_capacity'] * 100)
        
        # Calculate available capacity by group
        for group in [1, 2, 3]:
            total_available = sum(
                driver_data['available'].get(group, 0) 
                for driver_data in summary['drivers'].values()
            )
            summary['available_capacity'][group] = total_available
        
        return summary
    
    def get_reassignment_candidates(self, target_group: int, min_distance: float = 0.5) -> List[Dict]:
        """Find potential candidates for reassignment within a group"""
        candidates = []
        
        if not self.data_manager.distance_matrix:
            return candidates
        
        # Find dogs in target group
        target_dogs = []
        for dog_id, dog in self.data_manager.dogs.items():
            if target_group in dog.groups:
                target_dogs.append((dog_id, dog))
        
        # Analyze potential swaps
        for dog_id, dog in target_dogs:
            current_driver = self.data_manager.current_assignments.get(dog_id)
            if not current_driver:
                continue
            
            # Find nearby dogs with different drivers
            neighbors = self.data_manager.distance_matrix.get_neighbors(dog_id, max_distance=2.0)
            
            for neighbor_id, distance in neighbors:
                if neighbor_id not in self.data_manager.dogs:
                    continue
                
                neighbor_dog = self.data_manager.dogs[neighbor_id]
                neighbor_driver = self.data_manager.current_assignments.get(neighbor_id)
                
                if neighbor_driver and neighbor_driver != current_driver:
                    if target_group in neighbor_dog.groups and distance >= min_distance:
                        candidates.append({
                            'dog_id': dog_id,
                            'neighbor_id': neighbor_id,
                            'distance': distance,
                            'current_driver': current_driver,
                            'neighbor_driver': neighbor_driver,
                            'swap_potential': self._can_swap(dog, neighbor_dog)
                        })
        
        return sorted(candidates, key=lambda x: x['distance'])
    
    def _can_swap(self, dog1, dog2) -> bool:
        """Check if two dogs can be swapped between drivers"""
        driver1_name = self.data_manager.current_assignments.get(dog1.dog_id)
        driver2_name = self.data_manager.current_assignments.get(dog2.dog_id)
        
        if not driver1_name or not driver2_name:
            return False
        
        driver1 = self.data_manager.drivers.get(driver1_name)
        driver2 = self.data_manager.drivers.get(driver2_name)
        
        if not driver1 or not driver2:
            return False
        
        # Check if each driver can accept the other's dog
        can_1_accept_2 = self._can_driver_accept_replacing(driver1, dog2, dog1)
        can_2_accept_1 = self._can_driver_accept_replacing(driver2, dog1, dog2)
        
        return can_1_accept_2 and can_2_accept_1
    
    def _can_driver_accept_replacing(self, driver, new_dog, old_dog) -> bool:
        """Check if driver can accept new dog by replacing old dog"""
        # Check callouts
        if any(group in driver.callouts for group in new_dog.groups):
            return False
        
        # Simulate capacity after replacement
        current_loads = self.data_manager.get_driver_current_loads(driver.name)
        
        # Remove old dog's load
        for group in old_dog.groups:
            current_loads[group] = current_loads.get(group, 0) - old_dog.num_dogs
        
        # Add new dog's load
        for group in new_dog.groups:
            new_load = current_loads.get(group, 0) + new_dog.num_dogs
            if new_load > driver.get_capacity(group):
                return False
        
        return True

def create_assignment_engine(data_manager) -> AssignmentEngine:
    """Factory function to create assignment engine"""
    return AssignmentEngine(data_manager)
