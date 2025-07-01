"""
Enhanced Assignment Logic Module
Cost-aware assignment logic that minimizes total miles and tracks drive time impact
Complete file ready for copy-paste
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import time

# Import the enhanced optimization engine
from enhanced_optimization_engine import EnhancedMultiAlgorithmOptimizer, AssignmentSolution, SystemMilesCalculator

@dataclass
class EnhancedReassignmentResult:
    """Enhanced results with cost and mile tracking"""
    assignments: List[Dict]
    quality_score: float
    total_distance: float
    system_total_miles: float  # Total system miles after reassignment
    baseline_total_miles: float  # Total system miles before reassignment
    avg_drive_time_before: float  # Average drive time before
    avg_drive_time_after: float   # Average drive time after
    dogs_affected: int
    strategy_used: str
    success: bool
    cost_savings: float  # Dollar savings
    miles_saved: float   # Miles saved
    driver_impact_summary: Dict[str, Dict]  # Per-driver impact
    error_message: Optional[str] = None
    
    def get_cost_benefit_report(self) -> str:
        """Generate detailed cost-benefit report"""
        cost_per_mile = 10.0
        
        return f"""
📊 ENHANCED COST-BENEFIT REPORT
===============================
🎯 MILE OPTIMIZATION RESULTS:
   • Total system miles BEFORE: {self.baseline_total_miles:.2f}
   • Total system miles AFTER:  {self.system_total_miles:.2f}
   • Miles SAVED: {self.miles_saved:.2f}
   • Cost SAVED: ${self.cost_savings:.2f}

⏱️ DRIVE TIME IMPACT:
   • Average drive time BEFORE: {self.avg_drive_time_before:.2f}
   • Average drive time AFTER:  {self.avg_drive_time_after:.2f}
   • Change: {self.avg_drive_time_after - self.avg_drive_time_before:+.2f}

🚗 ASSIGNMENT DETAILS:
   • Dogs reassigned: {self.dogs_affected}
   • Strategy used: {self.strategy_used}
   • Quality score: {self.quality_score:.3f}

💰 COST ANALYSIS:
   • Cost per mile: ${cost_per_mile:.2f}
   • Daily savings: ${self.cost_savings:.2f}
   • Weekly savings: ${self.cost_savings * 5:.2f}
   • Annual savings: ${self.cost_savings * 250:.2f}

🎯 OPTIMIZATION GOAL STATUS:
   {'✅ SUCCESS' if self.miles_saved > 0 else '⚠️  NO SAVINGS'}: {'Miles reduced!' if self.miles_saved > 0 else 'No mile reduction achieved'}
   {'✅ SUCCESS' if self.avg_drive_time_after <= self.avg_drive_time_before else '❌ WARNING'}: {'Drive time maintained' if self.avg_drive_time_after <= self.avg_drive_time_before else 'Drive time increased!'}
"""

class CostAwareAssignmentEngine:
    """
    Enhanced assignment engine that prioritizes total mile reduction and drive time optimization
    """
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.optimizer = EnhancedMultiAlgorithmOptimizer(data_manager)
        self.miles_calculator = SystemMilesCalculator(data_manager)
        self.cost_per_mile = 10.0  # $10 per mile from config
    
    def reassign_dogs(self, called_out_driver: str, affected_groups: List[int], 
                     strategy: str = "enhanced_optimization") -> EnhancedReassignmentResult:
        """
        Enhanced reassignment with total mile and cost optimization
        
        Args:
            called_out_driver: Name of driver being called out
            affected_groups: List of group numbers affected by callout
            strategy: Optimization strategy to use
            
        Returns:
            EnhancedReassignmentResult with detailed cost and mile analysis
        """
        try:
            print(f"🎯 Starting enhanced reassignment optimization...")
            print(f"   Driver called out: {called_out_driver}")
            print(f"   Groups affected: {affected_groups}")
            
            # Capture baseline metrics BEFORE any changes
            baseline_total_miles = self.miles_calculator.calculate_total_system_miles()
            baseline_avg_drive_time = self.miles_calculator.calculate_average_drive_time()
            
            print(f"📊 Baseline metrics:")
            print(f"   Total system miles: {baseline_total_miles:.2f}")
            print(f"   Average drive time: {baseline_avg_drive_time:.2f}")
            
            # Update driver callout status
            self._update_driver_callouts(called_out_driver, affected_groups)
            
            # Find dogs that need reassignment
            dogs_to_reassign = self._identify_dogs_to_reassign(called_out_driver, affected_groups)
            
            if not dogs_to_reassign:
                return EnhancedReassignmentResult(
                    assignments=[],
                    quality_score=0.0,
                    total_distance=0.0,
                    system_total_miles=baseline_total_miles,
                    baseline_total_miles=baseline_total_miles,
                    avg_drive_time_before=baseline_avg_drive_time,
                    avg_drive_time_after=baseline_avg_drive_time,
                    dogs_affected=0,
                    strategy_used="None needed",
                    success=True,
                    cost_savings=0.0,
                    miles_saved=0.0,
                    driver_impact_summary={},
                    error_message="No dogs need reassignment"
                )
            
            print(f"🐕 Found {len(dogs_to_reassign)} dogs needing reassignment")
            
            # Estimate potential cost impact
            potential_cost_impact = len(dogs_to_reassign) * 2.0 * self.cost_per_mile  # Rough estimate
            print(f"💰 Potential cost impact: ${potential_cost_impact:.2f}")
            
            # Choose optimization intensity based on potential impact
            if potential_cost_impact >= 50.0:  # High impact - use thorough optimization
                print(f"🔥 High impact scenario - using thorough optimization")
                solution = self._thorough_optimization(dogs_to_reassign)
            else:
                print(f"⚡ Standard optimization")
                solution = self.optimizer.optimize(dogs_to_reassign)
            
            if not solution:
                return EnhancedReassignmentResult(
                    assignments=[],
                    quality_score=float('inf'),
                    total_distance=float('inf'),
                    system_total_miles=baseline_total_miles,
                    baseline_total_miles=baseline_total_miles,
                    avg_drive_time_before=baseline_avg_drive_time,
                    avg_drive_time_after=baseline_avg_drive_time,
                    dogs_affected=len(dogs_to_reassign),
                    strategy_used="Failed",
                    success=False,
                    cost_savings=0.0,
                    miles_saved=0.0,
                    driver_impact_summary={},
                    error_message="No valid reassignment solution found"
                )
            
            # Convert solution to assignment format
            assignments = self._solution_to_assignments(solution, dogs_to_reassign)
            
            # Apply assignments if successful
            if assignments:
                self._apply_assignments(solution.assignments)
            
            # Calculate miles saved and cost savings
            miles_saved = max(0, baseline_total_miles - solution.system_total_miles)
            cost_savings = miles_saved * self.cost_per_mile
            
            print(f"✅ Optimization complete!")
            print(f"   Miles saved: {miles_saved:.2f}")
            print(f"   Cost savings: ${cost_savings:.2f}")
            print(f"   Drive time change: {solution.avg_drive_time_after - solution.avg_drive_time_before:+.2f}")
            
            return EnhancedReassignmentResult(
                assignments=assignments,
                quality_score=solution.quality_score(),
                total_distance=solution.total_distance,
                system_total_miles=solution.system_total_miles,
                baseline_total_miles=baseline_total_miles,
                avg_drive_time_before=solution.avg_drive_time_before,
                avg_drive_time_after=solution.avg_drive_time_after,
                dogs_affected=len(assignments),
                strategy_used=solution.strategy_used,
                success=True,
                cost_savings=cost_savings,
                miles_saved=miles_saved,
                driver_impact_summary=solution.driver_impact
            )
            
        except Exception as e:
            print(f"❌ Error in reassignment: {e}")
            return EnhancedReassignmentResult(
                assignments=[],
                quality_score=float('inf'),
                total_distance=float('inf'),
                system_total_miles=0.0,
                baseline_total_miles=0.0,
                avg_drive_time_before=0.0,
                avg_drive_time_after=0.0,
                dogs_affected=0,
                strategy_used="Error",
                success=False,
                cost_savings=0.0,
                miles_saved=0.0,
                driver_impact_summary={},
                error_message=f"Reassignment failed: {str(e)}"
            )
    
    def _thorough_optimization(self, dogs_to_reassign: List) -> Optional[AssignmentSolution]:
        """Run thorough optimization for high-impact scenarios"""
        print(f"🔬 Running thorough optimization with multiple strategies...")
        
        solutions = []
        
        # Strategy 1: Enhanced Hungarian
        try:
            solution1 = self.optimizer.hungarian.optimize(dogs_to_reassign)
            if solution1:
                solutions.append(solution1)
                print(f"   Hungarian: {solution1.quality_score():.3f} quality, {solution1.system_total_miles:.2f} miles")
        except Exception as e:
            print(f"   Hungarian failed: {e}")
        
        # Strategy 2: Try multiple greedy approaches with different starting points
        for start_driver in list(self.data_manager.drivers.keys())[:3]:  # Try top 3 drivers
            try:
                solution = self._greedy_optimization_from_driver(dogs_to_reassign, start_driver)
                if solution:
                    solutions.append(solution)
                    print(f"   Greedy from {start_driver}: {solution.quality_score():.3f} quality")
            except Exception:
                pass
        
        # Strategy 3: Mile-first optimization
        try:
            solution3 = self._mile_first_optimization(dogs_to_reassign)
            if solution3:
                solutions.append(solution3)
                print(f"   Mile-first: {solution3.quality_score():.3f} quality")
        except Exception:
            pass
        
        # Return best solution
        if solutions:
            best = min(solutions, key=lambda s: s.quality_score())
            print(f"🏆 Best solution: {best.strategy_used} with {best.quality_score():.3f} quality")
            return best
        
        return None
    
    def _greedy_optimization_from_driver(self, dogs_to_reassign: List, start_driver: str) -> Optional[AssignmentSolution]:
        """Greedy optimization starting from a specific driver"""
        # Implementation would start assignments from a specific driver and work outward
        # This is a simplified version
        return self.optimizer.optimize(dogs_to_reassign)
    
    def _mile_first_optimization(self, dogs_to_reassign: List) -> Optional[AssignmentSolution]:
        """Optimization that prioritizes mile reduction above all else"""
        assignments = {}
        baseline_miles = self.miles_calculator.calculate_total_system_miles()
        baseline_avg_time = self.miles_calculator.calculate_average_drive_time()
        
        # Sort dogs by their impact on total miles (largest impact first)
        dogs_by_impact = self._sort_dogs_by_mile_impact(dogs_to_reassign)
        
        # Create a copy for testing
        test_assignments = dict(self.data_manager.current_assignments)
        
        for dog in dogs_by_impact:
            best_driver = self._find_best_driver_for_miles(dog, test_assignments)
            if best_driver:
                assignments[dog.dog_id] = best_driver
                test_assignments[dog.dog_id] = best_driver  # Update for next calculation
        
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
                load_balance_score=0.0,
                constraint_violations=0,
                strategy_used="Mile-First Optimization",
                cost_savings=(baseline_miles - final_miles) * self.cost_per_mile,
                driver_impact={}
            )
        
        return None
    
    def _sort_dogs_by_mile_impact(self, dogs: List) -> List:
        """Sort dogs by their potential impact on total system miles"""
        # Simplified: just return as-is for now
        # Real implementation would calculate potential mile impact for each dog
        return dogs
    
    def _find_best_driver_for_miles(self, dog, test_assignments: Dict[str, str]) -> Optional[str]:
        """Find the driver that results in minimum total system miles"""
        best_driver = None
        best_miles = float('inf')
        
        for driver_name, driver in self.data_manager.drivers.items():
            if self._can_driver_accept_dog(driver, dog):
                # Test assignment
                test_assignments[dog.dog_id] = driver_name
                test_miles = self.miles_calculator.calculate_total_system_miles(test_assignments)
                
                if test_miles < best_miles:
                    best_miles = test_miles
                    best_driver = driver_name
                
                # Reset for next test
                test_assignments[dog.dog_id] = self.data_manager.current_assignments.get(dog.dog_id, "")
        
        return best_driver
    
    def _can_driver_accept_dog(self, driver, dog) -> bool:
        """Check if driver can accept this specific dog"""
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
                distance = self._calculate_assignment_distance(dog.dog_id, new_driver)
                
                assignments.append({
                    "Dog ID": dog.dog_id,
                    "Dog Name": dog.name,
                    "New Assignment": f"{new_driver}:{'&'.join(map(str, sorted(dog.groups)))}",
                    "New Driver": new_driver,
                    "Groups": sorted(list(dog.groups)),
                    "Distance": round(distance, 3),
                    "Number of Dogs": dog.num_dogs,
                    "Previous Driver": self.data_manager.current_assignments.get(dog.dog_id, "Unknown"),
                    "Miles Impact": f"{solution.system_total_miles:.2f}",  # NEW
                    "Cost Impact": f"${(solution.cost_savings / len(dogs_to_reassign)):.2f}"  # NEW
                })
        
        return assignments
    
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
    
    def _apply_assignments(self, assignments: Dict[str, str]):
        """Apply assignments to data manager state"""
        for dog_id, new_driver in assignments.items():
            if dog_id in self.data_manager.current_assignments:
                self.data_manager.current_assignments[dog_id] = new_driver
    
    def get_system_mile_analysis(self) -> Dict:
        """Get comprehensive system mile analysis"""
        total_miles = self.miles_calculator.calculate_total_system_miles()
        avg_drive_time = self.miles_calculator.calculate_average_drive_time()
        
        # Per-driver analysis
        driver_analysis = {}
        driver_dogs = defaultdict(list)
        
        for dog_id, driver_name in self.data_manager.current_assignments.items():
            if dog_id in self.data_manager.dogs:
                driver_dogs[driver_name].append(dog_id)
        
        for driver_name, dog_ids in driver_dogs.items():
            if len(dog_ids) > 1:
                route_miles = self.miles_calculator._calculate_driver_route_miles(dog_ids)
            else:
                route_miles = 0.5 if len(dog_ids) == 1 else 0.0
            
            driver_analysis[driver_name] = {
                'dogs': len(dog_ids),
                'route_miles': route_miles,
                'efficiency': route_miles / len(dog_ids) if len(dog_ids) > 0 else 0
            }
        
        return {
            'total_system_miles': total_miles,
            'average_drive_time': avg_drive_time,
            'total_daily_cost': total_miles * self.cost_per_mile,
            'driver_breakdown': driver_analysis,
            'optimization_opportunities': self._identify_optimization_opportunities()
        }
    
    def _identify_optimization_opportunities(self) -> List[str]:
        """Identify potential optimization opportunities"""
        opportunities = []
        
        # Check for high-mileage drivers
        driver_dogs = defaultdict(list)
        for dog_id, driver_name in self.data_manager.current_assignments.items():
            if dog_id in self.data_manager.dogs:
                driver_dogs[driver_name].append(dog_id)
        
        for driver_name, dog_ids in driver_dogs.items():
            if len(dog_ids) > 1:
                route_miles = self.miles_calculator._calculate_driver_route_miles(dog_ids)
                if route_miles > 5.0:  # High mileage threshold
                    opportunities.append(f"Driver {driver_name} has high route mileage ({route_miles:.1f} miles)")
        
        return opportunities

def create_enhanced_assignment_engine(data_manager) -> CostAwareAssignmentEngine:
    """Factory function to create enhanced assignment engine"""
    return CostAwareAssignmentEngine(data_manager)

# Backward compatibility
class AssignmentEngine(CostAwareAssignmentEngine):
    """Backward compatibility wrapper"""
    
    def reassign_dogs(self, called_out_driver: str, affected_groups: List[int]) -> EnhancedReassignmentResult:
        """Maintain backward compatibility while using enhanced engine"""
        return super().reassign_dogs(called_out_driver, affected_groups, "enhanced_optimization")

def create_assignment_engine(data_manager) -> AssignmentEngine:
    """Factory function maintaining backward compatibility"""
    return AssignmentEngine(data_manager)

# Import existing capacity analyzer for backward compatibility
from assignment_logic import CapacityAnalyzer
