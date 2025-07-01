"""
FIXED Enhanced Assignment Logic Module
Fixes NaN drivers and provides better debugging
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import time

# Import the FIXED optimization engine
from enhanced_optimization_engine import EnhancedMultiAlgorithmOptimizer, AssignmentSolution, SystemMilesCalculator

@dataclass
class EnhancedReassignmentResult:
    """Enhanced results with cost and mile tracking"""
    assignments: List[Dict]
    quality_score: float
    total_distance: float
    system_total_miles: float
    baseline_total_miles: float
    avg_drive_time_before: float
    avg_drive_time_after: float
    dogs_affected: int
    strategy_used: str
    success: bool
    cost_savings: float
    miles_saved: float
    driver_impact_summary: Dict[str, Dict]
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

🎯 OPTIMIZATION GOAL STATUS:
   {'✅ SUCCESS' if self.miles_saved > 0 else '⚠️  NO SAVINGS'}: {'Miles reduced!' if self.miles_saved > 0 else 'No mile reduction achieved'}
   {'✅ SUCCESS' if self.avg_drive_time_after <= self.avg_drive_time_before else '❌ WARNING'}: {'Drive time maintained' if self.avg_drive_time_after <= self.avg_drive_time_before else 'Drive time increased!'}
"""

class CostAwareAssignmentEngine:
    """FIXED Enhanced assignment engine with better error handling"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.optimizer = EnhancedMultiAlgorithmOptimizer(data_manager)
        self.miles_calculator = SystemMilesCalculator(data_manager)
        self.cost_per_mile = 10.0
    
    def reassign_dogs(self, called_out_driver: str, affected_groups: List[int], 
                     strategy: str = "enhanced_optimization") -> EnhancedReassignmentResult:
        """FIXED: Enhanced reassignment with proper error handling"""
        try:
            print(f"\n🎯 === STARTING DOG REASSIGNMENT ===")
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
                print("ℹ️  No dogs need reassignment")
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
            
            print(f"🐕 Found {len(dogs_to_reassign)} dogs needing reassignment:")
            for dog in dogs_to_reassign:
                print(f"   • {dog.dog_id} ({dog.name}) - Groups: {list(dog.groups)}")
            
            # FIXED: Check driver availability before optimization
            available_drivers = self._check_driver_availability(dogs_to_reassign)
            if not available_drivers:
                print("🚨 NO AVAILABLE DRIVERS - Cannot proceed with optimization")
                return self._create_failed_result(dogs_to_reassign, baseline_total_miles, baseline_avg_drive_time, 
                                                "No available drivers found")
            
            print(f"🚗 Available drivers: {available_drivers}")
            
            # Run optimization
            print(f"\n🔄 Running optimization...")
            solution = self.optimizer.optimize(dogs_to_reassign)
            
            if not solution or not solution.assignments:
                print("❌ Optimization failed - trying manual assignment...")
                solution = self._manual_assignment_fallback(dogs_to_reassign)
            
            if not solution or not solution.assignments:
                return self._create_failed_result(dogs_to_reassign, baseline_total_miles, baseline_avg_drive_time,
                                                "All optimization strategies failed")
            
            # FIXED: Validate all assignments
            validated_assignments = self._validate_assignments(solution.assignments)
            if not validated_assignments:
                return self._create_failed_result(dogs_to_reassign, baseline_total_miles, baseline_avg_drive_time,
                                                "No valid assignments after validation")
            
            # Update solution with validated assignments
            solution.assignments = validated_assignments
            
            # Convert solution to assignment format
            assignments = self._solution_to_assignments(solution, dogs_to_reassign)
            
            # Apply assignments if successful
            if assignments:
                self._apply_assignments(solution.assignments)
            
            # Calculate miles saved and cost savings
            miles_saved = max(0, baseline_total_miles - solution.system_total_miles)
            cost_savings = miles_saved * self.cost_per_mile
            
            print(f"\n✅ === OPTIMIZATION COMPLETE ===")
            print(f"   Dogs reassigned: {len(assignments)}")
            print(f"   Strategy used: {solution.strategy_used}")
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
            print(f"❌ CRITICAL ERROR in reassignment: {e}")
            import traceback
            traceback.print_exc()
            
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
    
    def _check_driver_availability(self, dogs_to_reassign: List) -> List[str]:
        """FIXED: Thorough driver availability check"""
        available_drivers = []
        
        print(f"\n🔍 === DRIVER AVAILABILITY CHECK ===")
        
        for driver_name, driver in self.data_manager.drivers.items():
            print(f"\n🚗 Checking {driver_name}:")
            
            # Check if completely called out
            if len(driver.callouts) >= 3:
                print(f"   ❌ Called out all groups: {driver.callouts}")
                continue
            
            # Check capacity for each dog
            can_accept_any = False
            current_loads = self.data_manager.get_driver_current_loads(driver_name)
            
            print(f"   📊 Current loads: {current_loads}")
            print(f"   📊 Capacities: {driver.group_capacities}")
            print(f"   📊 Callouts: {driver.callouts}")
            
            for dog in dogs_to_reassign:
                can_accept_this_dog = True
                issues = []
                
                # Check group conflicts
                for group in dog.groups:
                    if group in driver.callouts:
                        can_accept_this_dog = False
                        issues.append(f"Called out group {group}")
                        break
                
                # Check capacity
                if can_accept_this_dog:
                    for group in dog.groups:
                        capacity = driver.get_capacity(group)
                        current_load = current_loads.get(group, 0)
                        available = capacity - current_load
                        
                        if available < dog.num_dogs:
                            can_accept_this_dog = False
                            issues.append(f"Group {group}: need {dog.num_dogs}, only {available} available")
                            break
                
                if can_accept_this_dog:
                    can_accept_any = True
                    print(f"   ✅ Can accept {dog.dog_id}")
                    break
                else:
                    print(f"   ❌ Cannot accept {dog.dog_id}: {', '.join(issues)}")
            
            if can_accept_any:
                available_drivers.append(driver_name)
                print(f"   ✅ {driver_name} is AVAILABLE")
            else:
                print(f"   ❌ {driver_name} is UNAVAILABLE")
        
        print(f"\n📋 Available drivers: {available_drivers}")
        return available_drivers
    
    def _validate_assignments(self, assignments: Dict[str, str]) -> Dict[str, str]:
        """FIXED: Validate all assignments to prevent NaN drivers"""
        validated = {}
        
        print(f"\n🔍 === VALIDATING ASSIGNMENTS ===")
        
        for dog_id, driver_name in assignments.items():
            print(f"🔍 Validating {dog_id} → {driver_name}")
            
            # Check driver name is valid
            if not driver_name or driver_name == "nan" or driver_name == "NaN":
                print(f"   ❌ Invalid driver name: '{driver_name}'")
                continue
            
            # Check driver exists
            if driver_name not in self.data_manager.drivers:
                print(f"   ❌ Driver {driver_name} not found in system")
                continue
            
            # Check dog exists
            if dog_id not in self.data_manager.dogs:
                print(f"   ❌ Dog {dog_id} not found in system")
                continue
            
            dog = self.data_manager.dogs[dog_id]
            driver = self.data_manager.drivers[driver_name]
            
            # Check capacity
            current_loads = self.data_manager.get_driver_current_loads(driver_name)
            can_assign = True
            
            for group in dog.groups:
                if group in driver.callouts:
                    print(f"   ❌ Driver {driver_name} called out for group {group}")
                    can_assign = False
                    break
                
                capacity = driver.get_capacity(group)
                current_load = current_loads.get(group, 0)
                
                if current_load + dog.num_dogs > capacity:
                    print(f"   ❌ Capacity exceeded for group {group}: {current_load + dog.num_dogs} > {capacity}")
                    can_assign = False
                    break
            
            if can_assign:
                validated[dog_id] = driver_name
                print(f"   ✅ Valid assignment: {dog_id} → {driver_name}")
            else:
                print(f"   ❌ Invalid assignment: {dog_id} → {driver_name}")
        
        print(f"\n📋 Validated {len(validated)}/{len(assignments)} assignments")
        return validated
    
    def _manual_assignment_fallback(self, dogs_to_reassign: List) -> Optional[AssignmentSolution]:
        """FIXED: Manual assignment fallback when optimization fails"""
        print(f"\n🛠️  === MANUAL ASSIGNMENT FALLBACK ===")
        
        assignments = {}
        baseline_miles = self.miles_calculator.calculate_total_system_miles()
        baseline_avg_time = self.miles_calculator.calculate_average_drive_time()
        
        # Get available drivers
        available_drivers = []
        for driver_name, driver in self.data_manager.drivers.items():
            if len(driver.callouts) < 3:
                available_drivers.append(driver_name)
        
        if not available_drivers:
            print("❌ No drivers available for manual assignment")
            return None
        
        print(f"🚗 Available drivers for manual assignment: {available_drivers}")
        
        # Try to assign each dog
        for dog in dogs_to_reassign:
            assigned = False
            
            for driver_name in available_drivers:
                driver = self.data_manager.drivers[driver_name]
                
                # Check if this driver can take this dog
                if self._can_driver_accept_dog_simple(driver, dog):
                    assignments[dog.dog_id] = driver_name
                    print(f"✅ Manual: {dog.dog_id} → {driver_name}")
                    assigned = True
                    break
            
            if not assigned:
                print(f"❌ Could not manually assign {dog.dog_id}")
        
        if assignments:
            # Calculate final metrics
            final_assignments = dict(self.data_manager.current_assignments)
            final_assignments.update(assignments)
            
            final_miles = self.miles_calculator.calculate_total_system_miles(final_assignments)
            final_avg_time = self.miles_calculator.calculate_average_drive_time(final_assignments)
            
            return AssignmentSolution(
                assignments=assignments,
                total_distance=len(assignments) * 1.5,
                system_total_miles=final_miles,
                avg_drive_time_before=baseline_avg_time,
                avg_drive_time_after=final_avg_time,
                load_balance_score=0.0,
                constraint_violations=0,
                strategy_used="Manual Assignment Fallback",
                cost_savings=(baseline_miles - final_miles) * self.cost_per_mile,
                driver_impact={}
            )
        
        return None
    
    def _can_driver_accept_dog_simple(self, driver, dog) -> bool:
        """Simple check if driver can accept dog"""
        # Check callouts
        for group in dog.groups:
            if group in driver.callouts:
                return False
        
        # Check capacity
        current_loads = self.data_manager.get_driver_current_loads(driver.name)
        for group in dog.groups:
            capacity = driver.get_capacity(group)
            current_load = current_loads.get(group, 0)
            if current_load + dog.num_dogs > capacity:
                return False
        
        return True
    
    def _create_failed_result(self, dogs_to_reassign: List, baseline_miles: float, 
                             baseline_time: float, error_msg: str) -> EnhancedReassignmentResult:
        """Create a failed result with proper error message"""
        return EnhancedReassignmentResult(
            assignments=[],
            quality_score=float('inf'),
            total_distance=float('inf'),
            system_total_miles=baseline_miles,
            baseline_total_miles=baseline_miles,
            avg_drive_time_before=baseline_time,
            avg_drive_time_after=baseline_time,
            dogs_affected=len(dogs_to_reassign),
            strategy_used="Failed",
            success=False,
            cost_savings=0.0,
            miles_saved=0.0,
            driver_impact_summary={},
            error_message=error_msg
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
                
                # FIXED: Better distance calculation
                distance = self._calculate_assignment_distance(dog.dog_id, new_driver)
                
                assignments.append({
                    "Dog ID": dog.dog_id,
                    "Dog Name": dog.name,
                    "New Assignment": f"{new_driver}:{'&'.join(map(str, sorted(dog.groups)))}",
                    "New Driver": new_driver,
                    "Groups": sorted(list(dog.groups)),
                    "Distance": round(distance, 2),  # FIXED: Round to 2 decimals
                    "Number of Dogs": dog.num_dogs,
                    "Previous Driver": self.data_manager.current_assignments.get(dog.dog_id, "Unknown"),
                    "System Miles": f"{solution.system_total_miles:.1f}",
                    "Cost Savings": f"${solution.cost_savings:.2f}"
                })
        
        return assignments
    
    def _calculate_assignment_distance(self, dog_id: str, driver_name: str) -> float:
        """FIXED: More realistic distance calculation"""
        if not self.data_manager.distance_matrix:
            return 1.5  # Reasonable default
        
        driver_dogs = self.data_manager.get_dogs_for_driver(driver_name)
        if not driver_dogs:
            return 1.0  # New driver
        
        # Find closest dog in driver's route
        min_distance = float('inf')
        valid_distances = 0
        
        for other_dog_id in driver_dogs:
            if other_dog_id != dog_id:
                distance = self.data_manager.distance_matrix.get_distance(dog_id, other_dog_id)
                if distance != float('inf') and distance > 0:
                    min_distance = min(min_distance, distance)
                    valid_distances += 1
        
        # FIXED: Better fallback
        if valid_distances == 0 or min_distance == float('inf'):
            return 1.2  # Reasonable estimate when no valid distances
        
        return min_distance
    
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
                route_miles = 1.0 if len(dog_ids) == 1 else 0.0  # FIXED: Realistic single dog
            
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
try:
    from assignment_logic import CapacityAnalyzer
except ImportError:
    # Create a simple fallback if the original doesn't exist
    class CapacityAnalyzer:
        def __init__(self, data_manager):
            self.data_manager = data_manager
        
        def get_capacity_summary(self) -> Dict:
            return {
                'drivers': {},
                'total_capacity': 0,
                'total_load': 0,
                'utilization': 0.0,
                'overloaded_drivers': [],
                'available_capacity': {}
            }
