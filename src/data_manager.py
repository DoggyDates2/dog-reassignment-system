"""
Data Management Module
Handles loading, validation, and caching of dog assignment data
"""

import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Set, List, Optional, Tuple
import re
from functools import lru_cache

@dataclass(frozen=True)
class DogProfile:
    """Immutable dog profile for better caching and consistency"""
    dog_id: str
    name: str
    groups: frozenset[int]
    num_dogs: int
    address: str
    
    def __post_init__(self):
        if not self.dog_id.strip():
            raise ValueError("Dog ID cannot be empty")
        if self.num_dogs <= 0:
            raise ValueError("Number of dogs must be positive")
        if not self.groups:
            raise ValueError("Dog must belong to at least one group")

@dataclass
class DriverProfile:
    """Driver profile with capacity management"""
    name: str
    group_capacities: Dict[int, int]
    callouts: Set[int]
    
    def __post_init__(self):
        # Ensure all groups 1-3 have capacities
        for i in [1, 2, 3]:
            if i not in self.group_capacities:
                self.group_capacities[i] = 9
    
    def is_called_out_for_group(self, group: int) -> bool:
        return group in self.callouts
    
    def get_capacity(self, group: int) -> int:
        return self.group_capacities.get(group, 9)
    
    def set_callout(self, group: int, called_out: bool = True):
        if called_out:
            self.callouts.add(group)
        else:
            self.callouts.discard(group)

class DistanceMatrix:
    """Optimized distance matrix with caching and validation"""
    
    def __init__(self, raw_matrix_df: pd.DataFrame):
        self._validate_matrix(raw_matrix_df)
        self.dog_ids = [str(col).strip() for col in raw_matrix_df.columns[1:]]
        self._matrix = self._build_numpy_matrix(raw_matrix_df)
        self._id_to_index = {dog_id: i for i, dog_id in enumerate(self.dog_ids)}
    
    def _validate_matrix(self, df: pd.DataFrame):
        """Validate distance matrix format"""
        if df.empty:
            raise ValueError("Distance matrix cannot be empty")
        if len(df.columns) < 2:
            raise ValueError("Distance matrix must have at least 2 columns")
    
    def _build_numpy_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Convert to optimized numpy matrix"""
        n = len(self.dog_ids)
        matrix = np.full((n, n), np.inf)
        
        for i, row in df.iterrows():
            if i >= n:
                break
            row_id = str(row.iloc[0]).strip()
            if row_id in self._id_to_index:
                row_idx = self._id_to_index[row_id]
                for j, dog_id in enumerate(self.dog_ids):
                    try:
                        val = float(row.iloc[j + 1])
                        if 0 <= val <= 10.0:
                            matrix[row_idx][j] = val
                    except (ValueError, IndexError):
                        continue
        
        # Ensure diagonal is zero
        np.fill_diagonal(matrix, 0.0)
        return matrix
    
    @lru_cache(maxsize=10000)
    def get_distance(self, dog1_id: str, dog2_id: str) -> float:
        """Get distance between two dogs with caching"""
        if dog1_id == dog2_id:
            return 0.0
        
        idx1 = self._id_to_index.get(dog1_id)
        idx2 = self._id_to_index.get(dog2_id)
        
        if idx1 is None or idx2 is None:
            return float('inf')
        
        return float(self._matrix[idx1][idx2])
    
    def get_neighbors(self, dog_id: str, max_distance: float = 1.0) -> List[Tuple[str, float]]:
        """Get all neighbors within max_distance, sorted by distance"""
        if dog_id not in self._id_to_index:
            return []
        
        idx = self._id_to_index[dog_id]
        neighbors = []
        
        distances = self._matrix[idx]
        valid_distances = np.where((distances > 0) & (distances <= max_distance))[0]
        
        for other_idx in valid_distances:
            other_id = self.dog_ids[other_idx]
            distance = float(distances[other_idx])
            neighbors.append((other_id, distance))
        
        return sorted(neighbors, key=lambda x: x[1])

class DataManager:
    """Centralized data management with validation and caching"""
    
    def __init__(self):
        self.dogs: Dict[str, DogProfile] = {}
        self.drivers: Dict[str, DriverProfile] = {}
        self.current_assignments: Dict[str, str] = {}  # dog_id -> driver_name
        self.distance_matrix: Optional[DistanceMatrix] = None
        self._load_errors: List[str] = []
    
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def load_from_urls(_self, map_url: str, matrix_url: str) -> bool:
        """Load and parse data from Google Sheets URLs"""
try:
            with st.spinner("📥 Loading map data..."):
                map_df = pd.read_csv(map_url, dtype=str, on_bad_lines='skip', encoding='utf-8')
            
            with st.spinner("📥 Loading distance matrix..."):
                matrix_df = pd.read_csv(matrix_url, dtype=str, on_bad_lines='skip', encoding='utf-8')
            
            return _self._process_raw_data(map_df, matrix_df)
                
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            _self._load_errors.append(f"URL loading error: {e}")
            return False
    
    def _process_raw_data(self, map_df: pd.DataFrame, matrix_df: pd.DataFrame) -> bool:
        """Process raw dataframes into optimized structures"""
        self._load_errors.clear()
        
        try:
            with st.spinner("🔄 Processing distance matrix..."):
                self.distance_matrix = DistanceMatrix(matrix_df)
            
            with st.spinner("🔄 Processing dog and driver data..."):
                # Clean map data
                map_df = map_df.dropna(subset=['Dog ID', 'Name', 'Group'])
                map_df['Dog ID'] = map_df['Dog ID'].astype(str).str.strip()
                map_df['Name'] = map_df['Name'].astype(str).str.strip()
                map_df['Group'] = map_df['Group'].astype(str).str.strip()
                
                # Process dogs and assignments
                self._process_dogs(map_df)
                self._process_drivers(map_df)
            
            return True
            
        except Exception as e:
            st.error(f"Data processing error: {e}")
            self._load_errors.append(f"Processing error: {e}")
            return False
    
    def _process_dogs(self, df: pd.DataFrame):
        """Extract dog profiles and current assignments"""
        self.dogs.clear()
        self.current_assignments.clear()
        
        for _, row in df.iterrows():
            try:
                dog_id = row['Dog ID']
                driver_name = row['Name']
                group_str = row['Group']
                
                # Skip invalid rows
                if not all([dog_id, driver_name, group_str]):
                    continue
                
                # Parse groups
                groups = self._parse_groups(group_str)
                if not groups:
                    continue
                
                # Parse number of dogs
                try:
                    num_dogs = int(float(row.get('Number of dogs', 1)))
                except (ValueError, TypeError):
                    num_dogs = 1
                
                # Create dog profile
                dog = DogProfile(
                    dog_id=dog_id,
                    name=row.get('Dog Name', ''),
                    groups=frozenset(groups),
                    num_dogs=num_dogs,
                    address=row.get('Address', '')
                )
                
                self.dogs[dog_id] = dog
                self.current_assignments[dog_id] = driver_name
                
            except Exception as e:
                continue
    
    def _process_drivers(self, df: pd.DataFrame):
        """Extract driver profiles and capacities"""
        self.drivers.clear()
        processed_drivers = set()
        
        for _, row in df.iterrows():
            driver_name = str(row.get('Driver', '')).strip()
            
            if not driver_name or driver_name in processed_drivers:
                continue
            
            try:
                # Parse capacities
                capacities = {}
                for i, col in enumerate(['Group 1', 'Group 2', 'Group 3'], 1):
                    cap_val = str(row.get(col, '')).strip().upper()
                    if cap_val in ['', 'X', 'NAN']:
                        capacities[i] = 9
                    else:
                        try:
                            capacities[i] = int(cap_val)
                        except ValueError:
                            capacities[i] = 9
                
                # Parse callouts (X means called out)
                callouts = set()
                for i, col in enumerate(['Group 1', 'Group 2', 'Group 3'], 1):
                    if str(row.get(col, '')).strip().upper() == 'X':
                        callouts.add(i)
                
                self.drivers[driver_name] = DriverProfile(
                    name=driver_name,
                    group_capacities=capacities,
                    callouts=callouts
                )
                
                processed_drivers.add(driver_name)
                
            except Exception as e:
                continue
    
    @lru_cache(maxsize=1000)
    def _parse_groups(self, group_str: str) -> Tuple[int, ...]:
        """Parse group string into sorted tuple of integers"""
        if not group_str:
            return tuple()
        
        # Remove LM prefix and extract digits
        clean_str = group_str.replace("LM", "")
        groups = []
        for char in clean_str:
            if char.isdigit():
                group_num = int(char)
                if 1 <= group_num <= 3:
                    groups.append(group_num)
        
        return tuple(sorted(set(groups)))
    
    def get_driver_current_loads(self, driver_name: str) -> Dict[int, int]:
        """Calculate current load for a driver"""
        loads = {1: 0, 2: 0, 3: 0}
        
        for dog_id, assigned_driver in self.current_assignments.items():
            if assigned_driver == driver_name and dog_id in self.dogs:
                dog = self.dogs[dog_id]
                for group in dog.groups:
                    loads[group] += dog.num_dogs
        
        return loads
    
    def get_dogs_for_driver(self, driver_name: str) -> Set[str]:
        """Get all dog IDs currently assigned to a driver"""
        return {dog_id for dog_id, driver in self.current_assignments.items() 
                if driver == driver_name}
    
    def validate_data_integrity(self) -> List[str]:
        """Validate data consistency and return list of issues"""
        issues = []
        
        # Check for dogs without drivers
        for dog_id in self.dogs:
            if dog_id not in self.current_assignments:
                issues.append(f"Dog {dog_id} has no driver assignment")
        
        # Check for assignments to non-existent drivers
        for dog_id, driver_name in self.current_assignments.items():
            if driver_name not in self.drivers:
                issues.append(f"Dog {dog_id} assigned to unknown driver {driver_name}")
        
        return issues

def create_data_manager(map_url: str, matrix_url: str) -> Optional[DataManager]:
    """Factory function to create and initialize data manager"""
    manager = DataManager()
    
    if manager.load_from_urls(map_url, matrix_url):
        return manager
    
    return None
