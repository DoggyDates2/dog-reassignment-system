import streamlit as st
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Set, List, Optional, Tuple
import re
from functools import lru_cache

@dataclass(frozen=True)
class DogProfile:
    dog_id: str
    name: str
    groups: frozenset[int]
    num_dogs: int
    address: str

@dataclass
class DriverProfile:
    name: str
    group_capacities: Dict[int, int]
    callouts: Set[int]
    
    def __post_init__(self):
        for i in [1, 2, 3]:
            if i not in self.group_capacities:
                self.group_capacities[i] = 9
    
    def get_capacity(self, group: int) -> int:
        return self.group_capacities.get(group, 9)
    
    def set_callout(self, group: int, called_out: bool = True):
        if called_out:
            self.callouts.add(group)
        else:
            self.callouts.discard(group)

class DistanceMatrix:
    def __init__(self, raw_matrix_df: pd.DataFrame):
        self.dog_ids = [str(col).strip() for col in raw_matrix_df.columns[1:]]
        self._id_to_index = {dog_id: i for i, dog_id in enumerate(self.dog_ids)}
        self._matrix = self._build_numpy_matrix(raw_matrix_df)
    
    def _build_numpy_matrix(self, df: pd.DataFrame) -> np.ndarray:
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
        
        np.fill_diagonal(matrix, 0.0)
        return matrix
    
    def get_distance(self, dog1_id: str, dog2_id: str) -> float:
        """Get distance between two dogs with caching - 0 means not allowed"""  # ← 4 spaces
        if dog1_id == dog2_id:  # ← 4 spaces
    
    idx1 = self._id_to_index.get(dog1_id)
    idx2 = self._id_to_index.get(dog2_id)
    
    if idx1 is None or idx2 is None:
        return float('inf')
    
    distance = float(self._matrix[idx1][idx2])
    
    # If distance is 0, it means "not allowed" - return infinity
    if distance == 0.0:
        return float('inf')
    
    return distance
    
    def get_neighbors(self, dog_id: str, max_distance: float = 1.0) -> List[Tuple[str, float]]:
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
    def __init__(self):
        self.dogs: Dict[str, DogProfile] = {}
        self.drivers: Dict[str, DriverProfile] = {}
        self.current_assignments: Dict[str, str] = {}
        self.distance_matrix: Optional[DistanceMatrix] = None
    
    def load_from_urls(self, map_url: str, matrix_url: str) -> bool:
        try:
            map_df = pd.read_csv(map_url, dtype=str, on_bad_lines='skip', encoding='utf-8')
            matrix_df = pd.read_csv(matrix_url, dtype=str, on_bad_lines='skip', encoding='utf-8')
            return self._process_raw_data(map_df, matrix_df)
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            return False
    
    def _process_raw_data(self, map_df: pd.DataFrame, matrix_df: pd.DataFrame) -> bool:
        try:
            self.distance_matrix = DistanceMatrix(matrix_df)
            
            map_df = map_df.dropna(subset=['Dog ID', 'Name', 'Group'])
            map_df['Dog ID'] = map_df['Dog ID'].astype(str).str.strip()
            map_df['Name'] = map_df['Name'].astype(str).str.strip()
            map_df['Group'] = map_df['Group'].astype(str).str.strip()
            
            self._process_dogs(map_df)
            self._process_drivers(map_df)
            return True
        except Exception as e:
            st.error(f"Data processing error: {e}")
            return False
    
    def _process_dogs(self, df: pd.DataFrame):
        self.dogs.clear()
        self.current_assignments.clear()
        
        for _, row in df.iterrows():
            try:
                dog_id = row['Dog ID']
                driver_name = row['Name']
                group_str = row['Group']
                
                if not all([dog_id, driver_name, group_str]):
                    continue
                
                groups = self._parse_groups(group_str)
                if not groups:
                    continue
                
                try:
                    num_dogs = int(float(row.get('Number of dogs', 1)))
                except:
                    num_dogs = 1
                
                dog = DogProfile(
                    dog_id=dog_id,
                    name=row.get('Dog Name', ''),
                    groups=frozenset(groups),
                    num_dogs=num_dogs,
                    address=row.get('Address', '')
                )
                
                self.dogs[dog_id] = dog
                self.current_assignments[dog_id] = driver_name
            except:
                continue
    
    def _process_drivers(self, df: pd.DataFrame):
        self.drivers.clear()
        processed_drivers = set()
        
        for _, row in df.iterrows():
            driver_name = str(row.get('Driver', '')).strip()
            
            if not driver_name or driver_name in processed_drivers:
                continue
            
            try:
                capacities = {}
                for i, col in enumerate(['Group 1', 'Group 2', 'Group 3'], 1):
                    cap_val = str(row.get(col, '')).strip().upper()
                    if cap_val in ['', 'X', 'NAN']:
                        capacities[i] = 9
                    else:
                        try:
                            capacities[i] = int(cap_val)
                        except:
                            capacities[i] = 9
                
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
            except:
                continue
    
    def _parse_groups(self, group_str: str) -> Tuple[int, ...]:
        if not group_str:
            return tuple()
        
        clean_str = group_str.replace("LM", "")
        groups = []
        for char in clean_str:
            if char.isdigit():
                group_num = int(char)
                if 1 <= group_num <= 3:
                    groups.append(group_num)
        
        return tuple(sorted(set(groups)))
    
    def get_driver_current_loads(self, driver_name: str) -> Dict[int, int]:
        loads = {1: 0, 2: 0, 3: 0}
        
        for dog_id, assigned_driver in self.current_assignments.items():
            if assigned_driver == driver_name and dog_id in self.dogs:
                dog = self.dogs[dog_id]
                for group in dog.groups:
                    loads[group] += dog.num_dogs
        
        return loads
    
    def get_dogs_for_driver(self, driver_name: str) -> Set[str]:
        return {dog_id for dog_id, driver in self.current_assignments.items() 
                if driver == driver_name}
    
    def validate_data_integrity(self) -> List[str]:
        issues = []
        
        for dog_id in self.dogs:
            if dog_id not in self.current_assignments:
                issues.append(f"Dog {dog_id} has no driver assignment")
        
        for dog_id, driver_name in self.current_assignments.items():
            if driver_name not in self.drivers:
                issues.append(f"Dog {dog_id} assigned to unknown driver {driver_name}")
        
        return issues

def create_data_manager(map_url: str, matrix_url: str) -> Optional[DataManager]:
    manager = DataManager()
    
    if manager.load_from_urls(map_url, matrix_url):
        return manager
    
    return None
