"""
UI Components Module
Modern Streamlit interface with state management and analytics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

from data_manager import create_data_manager
from assignment_logic import create_assignment_engine, ReassignmentResult, CapacityAnalyzer
from config import get_map_url, get_matrix_url

@dataclass
class HistoryEntry:
    """Entry in reassignment history"""
    timestamp: datetime
    result: ReassignmentResult
    driver_called_out: str
    groups_affected: List[int]

class SessionState:
    """Centralized session state management"""
    
    def __init__(self):
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize session state variables"""
        defaults = {
            'data_manager': None,
            'assignment_engine': None,
            'reassignment_history': [],
            'current_scenario': None,
            'data_last_loaded': None,
            'last_error': None
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @property
    def data_manager(self):
        return st.session_state.data_manager
    
    @data_manager.setter
    def data_manager(self, value):
        st.session_state.data_manager = value
        st.session_state.assignment_engine = create_assignment_engine(value) if value else None
        st.session_state.data_last_loaded = datetime.now()
    
    @property
    def assignment_engine(self):
        return st.session_state.assignment_engine
    
    def add_history_entry(self, entry: HistoryEntry):
        """Add entry to reassignment history"""
        st.session_state.reassignment_history.append(entry)
        # Keep only recent entries
        max_entries = 10
        if len(st.session_state.reassignment_history) > max_entries:
            st.session_state.reassignment_history = st.session_state.reassignment_history[-max_entries:]
    
    def get_history(self) -> List[HistoryEntry]:
        """Get reassignment history"""
        return st.session_state.reassignment_history.copy()
    
    def clear_history(self):
        """Clear reassignment history"""
        st.session_state.reassignment_history = []
    
    def set_error(self, error: str):
        """Set last error message"""
        st.session_state.last_error = error
    
    def get_error(self) -> Optional[str]:
        """Get and clear last error"""
        error = st.session_state.last_error
        st.session_state.last_error = None
        return error

class DashboardUI:
    """Main dashboard UI with optimized layout"""
    
    def __init__(self, session_state: SessionState):
        self.session_state = session_state
        self._setup_styling()
    
    def _setup_styling(self):
        """Setup custom CSS styling"""
        st.markdown("""
        <style>
        .metric-container {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            border-left: 4px solid #007bff;
        }
        .success-metric {
            border-left-color: #28a745;
            background-color: #d4f6d4;
        }
        .warning-metric {
            border-left-color: #ffc107;
            background-color: #fff3cd;
        }
        .error-metric {
            border-left-color: #dc3545;
            background-color: #f8d7da;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
            border-radius: 5px 5px 0 0;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render(self):
        """Render the main dashboard"""
        st.title("🐶 Optimized Dog Reassignment System")
        
        # Show any errors
        error = self.session_state.get_error()
        if error:
            st.error(f"❌ {error}")
        
        # Sidebar for data management
        self._render_sidebar()
        
        # Main content area
        if self.session_state.data_manager is None:
            self._render_welcome_screen()
        else:
            self._render_main_content()
    
    def _render_sidebar(self):
        """Render sidebar with data management"""
        with st.sidebar:
            st.header("📊 Data Management")
            
            # Data loading section
            with st.expander("🔄 Load Data", expanded=True):
                map_url = st.text_input(
                    "Map Data URL",
                    value=get_map_url(),
                    help="Google Sheets CSV export URL for dog/driver mapping"
                )
                
                matrix_url = st.text_input(
                    "Distance Matrix URL", 
                    value=get_matrix_url(),
                    help="Google Sheets CSV export URL for distance matrix"
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Load", type="primary"):
                        self._load_data(map_url, matrix_url)
                
                with col2:
                    if st.button("🗑️ Clear"):
                        self._clear_data()
            
            # Data status
            if self.session_state.data_manager:
                self._render_data_status()
            
            # Settings
            self._render_settings()
    
    def _render_data_status(self):
        """Render current data status"""
        dm = self.session_state.data_manager
        
        st.success(f"✅ Data loaded")
        st.metric("🐕 Dogs", len(dm.dogs))
        st.metric("🚗 Drivers", len(dm.drivers))
        
        if st.session_state.data_last_loaded:
            st.caption(f"Updated: {st.session_state.data_last_loaded.strftime('%H:%M:%S')}")
        
        # Data integrity check
        issues = dm.validate_data_integrity()
        if issues:
            st.warning(f"⚠️ {len(issues)} data issues")
            with st.expander("View Issues"):
                for issue in issues[:5]:
                    st.text(f"• {issue}")
                if len(issues) > 5:
                    st.text(f"... and {len(issues) - 5} more")
    
    def _render_settings(self):
        """Render optimization settings"""
        st.header("⚙️ Settings")
        
        with st.expander("Optimization Parameters"):
            st.slider(
                "Distance Weight", 
                0.1, 2.0, 
                1.0, 
                0.1, 
                key="distance_weight",
                help="Weight for distance in optimization cost function"
            )
            
            st.slider(
                "Load Balance Weight", 
                0.0, 1.0, 
                0.2, 
                0.1, 
                key="balance_weight",
                help="Weight for load balancing in cost function"
            )
            
            st.slider(
                "Max Distance Threshold", 
                0.5, 3.0, 
                1.5, 
                0.25, 
                key="max_distance",
                help="Maximum distance for considering assignments"
            )
    
    def _render_welcome_screen(self):
        """Render welcome screen when no data is loaded"""
        st.info("👈 Please load data using the sidebar to get started")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📋 How to Use
            1. **Load Data**: Enter Google Sheets URLs in the sidebar
            2. **Select Driver**: Choose which driver to call out  
            3. **Choose Groups**: Select affected groups (1, 2, 3)
            4. **Run Optimization**: Click "Optimize Reassignments"
            5. **Review Results**: See optimized assignments and quality metrics
            """)
        
        with col2:
            st.markdown("""
            ### 🎯 Features
            - **Multi-Algorithm Optimization**: Hungarian, Simulated Annealing, Constraints
            - **Quality Scoring**: Distance + Load Balancing + Group Compatibility  
            - **Visual Analytics**: Charts and metrics for solution quality
            - **History Tracking**: Keep track of previous reassignments
            - **Data Validation**: Automatic checks for data integrity
            """)
    
    def _render_main_content(self):
        """Render main content with tabs"""
        # Overview metrics
        self._render_overview_metrics()
        
        # Main tabs - ADDED DEBUG TAB
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Reassignment", "📊 Analytics", "📈 History", "🔍 Data Explorer", "🔧 Debug"])
        
        with tab1:
            self._render_reassignment_tab()
        
        with tab2:
            self._render_analytics_tab()
        
        with tab3:
            self._render_history_tab()
        
        with tab4:
            self._render_data_explorer_tab()
        
        with tab5:
            render_debug_tab(self.session_state.data_manager)
    
    def _render_overview_metrics(self):
        """Render overview metrics cards"""
        dm = self.session_state.data_manager
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("🐕 Total Dogs", len(dm.dogs))
        
        with col2:
            st.metric("🚗 Active Drivers", len(dm.drivers))
        
        with col3:
            analyzer = CapacityAnalyzer(dm)
            summary = analyzer.get_capacity_summary()
            st.metric("📊 Utilization", f"{summary['utilization']:.1f}%")
        
        with col4:
            avg_distance = self._calculate_average_distance()
            st.metric("📏 Avg Distance", f"{avg_distance:.2f}")
        
        with col5:
            issues = dm.validate_data_integrity()
            delta_color = "normal" if len(issues) == 0 else "inverse"
            st.metric("⚠️ Issues", len(issues), delta_color=delta_color)
    
    def _render_reassignment_tab(self):
        """Render reassignment interface"""
        st.header("🎯 Driver Callout & Reassignment")
        
        # ===== DEBUG SECTION =====
        st.subheader("🔧 Debug Analysis")
        dm = self.session_state.data_manager
        
        # Check data overlap
        todays_dogs = set(dm.dogs.keys())
        matrix_dogs = set(dm.distance_matrix.dog_ids) if dm.distance_matrix else set()
        missing_from_matrix = todays_dogs - matrix_dogs
        overlap = len(todays_dogs & matrix_dogs)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Today's Dogs", len(todays_dogs))
        with col2:
            st.metric("Matrix Dogs", len(matrix_dogs))
        with col3:
            st.metric("Overlap", overlap)
        with col4:
            st.metric("Missing", len(missing_from_matrix))
        
        if missing_from_matrix:
            st.error(f"❌ {len(missing_from_matrix)} dogs missing from distance matrix (causes 0.5 distances)")
            with st.expander("Missing Dogs"):
                for i, dog_id in enumerate(sorted(missing_from_matrix)):
                    if i < 10:
                        dog = dm.dogs.get(dog_id)
                        st.write(f"• {dog_id} - {dog.name if dog else 'Unknown'}")
                    elif i == 10:
                        st.write(f"... and {len(missing_from_matrix) - 10} more")
                        break
        else:
            st.success("✅ All dogs found in distance matrix")
        
        # Check driver availability for Aidan callout
        st.subheader("🚗 Driver Availability Check")
        
        # Simulate calling out Aidan for groups 1,2,3
        called_out_driver = "Aidan"
        affected_groups = [1, 2, 3]
        
        available_drivers = []
        unavailable_reasons = {}
        
        for driver_name, driver in dm.drivers.items():
            if driver_name == called_out_driver:
                continue
                
            reasons = []
            
            # Check callouts
            if len(driver.callouts) >= 3:
                reasons.append("Called out all groups")
            elif any(group in driver.callouts for group in affected_groups):
                called_out_groups = [g for g in affected_groups if g in driver.callouts]
                reasons.append(f"Called out groups {called_out_groups}")
            
            # Check capacity
            current_loads = dm.get_driver_current_loads(driver_name)
            capacity_issues = []
            for group in affected_groups:
                capacity = driver.get_capacity(group)
                load = current_loads.get(group, 0)
                available = capacity - load
                if available <= 0:
                    capacity_issues.append(f"Group {group}: {available}/{capacity}")
            
            if capacity_issues:
                reasons.append(f"No capacity: {', '.join(capacity_issues)}")
            
            if not reasons:
                available_drivers.append(driver_name)
            else:
                unavailable_reasons[driver_name] = reasons
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Available Drivers", len(available_drivers))
            if available_drivers:
                st.success("Available: " + ", ".join(available_drivers[:5]))
                if len(available_drivers) > 5:
                    st.write(f"... and {len(available_drivers) - 5} more")
        
        with col2:
            st.metric("Unavailable Drivers", len(unavailable_reasons))
            if unavailable_reasons:
                st.error("❌ Reasons drivers unavailable:")
                for driver, reasons in list(unavailable_reasons.items())[:3]:
                    st.write(f"• {driver}: {', '.join(reasons)}")
        
        if len(available_drivers) == 0:
            st.error("🚨 NO AVAILABLE DRIVERS - This is why you get 'nan' assignments!")
        
        st.divider()
        # ===== END DEBUG SECTION =====
        
        # Driver and group selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            available_drivers = sorted(dm.drivers.keys())
            selected_driver = st.selectbox(
                "👤 Select Driver to Call Out",
                available_drivers,
                help="Choose the driver who needs to be called out"
            )
        
        with col2:
            affected_groups = st.multiselect(
                "📋 Groups Affected",
                [1, 2, 3],
                default=[1, 2, 3],
                help="Select which groups are affected by the callout"
            )
        
        if selected_driver and affected_groups:
            # Show impact preview
            self._render_impact_preview(selected_driver, affected_groups)
            
            # Optimization controls
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Optimize Reassignments", type="primary", use_container_width=True):
                    self._run_optimization(selected_driver, affected_groups)
    
    def _render_impact_preview(self, driver_name: str, groups: List[int]):
        """Render preview of affected dogs"""
        dm = self.session_state.data_manager
        
        affected_dogs = []
        for dog_id, assigned_driver in dm.current_assignments.items():
            if assigned_driver == driver_name and dog_id in dm.dogs:
                dog = dm.dogs[dog_id]
                if any(group in groups for group in dog.groups):
                    affected_dogs.append(dog)
        
        if affected_dogs:
            st.info(f"📋 {len(affected_dogs)} dogs will need reassignment")
            
            with st.expander("👀 Preview Affected Dogs"):
                preview_data = []
                for dog in affected_dogs:
                    preview_data.append({
                        "Dog ID": dog.dog_id,
                        "Name": dog.name,
                        "Groups": ", ".join(map(str, sorted(dog.groups))),
                        "Number of Dogs": dog.num_dogs,
                        "Address": dog.address
                    })
                
                st.dataframe(pd.DataFrame(preview_data), use_container_width=True)
        else:
            st.warning("ℹ️ No dogs need reassignment for the selected driver and groups")
    
    def _run_optimization(self, driver_name: str, groups: List[int]):
        """Run optimization and display results"""
        engine = self.session_state.assignment_engine
        
        with st.spinner("🔄 Running optimization algorithms..."):
            result = engine.reassign_dogs(driver_name, groups)
        
        # Store in history
        history_entry = HistoryEntry(
            timestamp=datetime.now(),
            result=result,
            driver_called_out=driver_name,
            groups_affected=groups
        )
        self.session_state.add_history_entry(history_entry)
        
        # Display results
        if result.success:
            self._display_optimization_results(result)
        else:
            st.error(f"❌ Optimization failed: {result.error_message}")
    
    def _display_optimization_results(self, result: ReassignmentResult):
        """Display optimization results"""
        st.success(f"✅ Optimization Complete! {result.dogs_affected} dogs reassigned")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Quality Score", f"{result.quality_score:.2f}")
        with col2:
            st.metric("📏 Total Distance", f"{result.total_distance:.2f}")
        with col3:
            st.metric("🐕 Dogs Affected", result.dogs_affected)
        with col4:
            st.metric("⚡ Strategy", result.strategy_used)
        
        # Results table
        if result.assignments:
            st.subheader("📋 New Assignments")
            df = pd.DataFrame(result.assignments)
            st.dataframe(df, use_container_width=True)
            
            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"reassignments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    def _render_analytics_tab(self):
        """Render analytics and visualizations"""
        st.header("📊 System Analytics")
        
        dm = self.session_state.data_manager
        analyzer = CapacityAnalyzer(dm)
        
        # Driver utilization chart
        col1, col2 = st.columns(2)
        
        with col1:
            self._render_utilization_chart(analyzer)
        
        with col2:
            self._render_distance_distribution()
        
        # Capacity summary table
        st.subheader("🚗 Driver Capacity Summary")
        summary = analyzer.get_capacity_summary()
        
        summary_data = []
        for driver_name, driver_data in summary['drivers'].items():
            for group in [1, 2, 3]:
                summary_data.append({
                    "Driver": driver_name,
                    "Group": f"Group {group}",
                    "Capacity": driver_data['capacities'][group],
                    "Current Load": driver_data['current_loads'].get(group, 0),
                    "Utilization (%)": round(driver_data['utilization'][group], 1),
                    "Available": driver_data['available'][group],
                    "Status": "🔴 Overloaded" if driver_data['utilization'][group] > 100 
                             else "🟡 High" if driver_data['utilization'][group] > 80 
                             else "🟢 Normal"
                })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            st.dataframe(df, use_container_width=True)
    
    def _render_utilization_chart(self, analyzer):
        """Render driver utilization chart"""
        st.subheader("🚗 Driver Utilization by Group")
        
        summary = analyzer.get_capacity_summary()
        chart_data = []
        
        for driver_name, driver_data in summary['drivers'].items():
            for group in [1, 2, 3]:
                chart_data.append({
                    "Driver": driver_name,
                    "Group": f"Group {group}",
                    "Utilization": driver_data['utilization'][group]
                })
        
        if chart_data:
            df = pd.DataFrame(chart_data)
            fig = px.bar(
                df, 
                x="Driver", 
                y="Utilization", 
                color="Group",
                title="Driver Utilization by Group (%)",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.add_hline(y=100, line_dash="dash", line_color="red", annotation_text="100% Capacity")
            fig.update_layout(yaxis_title="Utilization (%)", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_distance_distribution(self):
        """Render distance distribution chart"""
        st.subheader("🗺️ Distance Distribution")
        
        dm = self.session_state.data_manager
        if not dm.distance_matrix:
            st.info("Distance matrix not available")
            return
        
        # Sample distances for performance
        distances = []
        sample_dogs = list(dm.dogs.keys())[:50]
        
        for dog_id in sample_dogs:
            neighbors = dm.distance_matrix.get_neighbors(dog_id, max_distance=3.0)
            distances.extend([dist for _, dist in neighbors[:10]])  # Top 10 neighbors
        
        if distances:
            fig = px.histogram(
                x=distances,
                title="Distribution of Dog-to-Dog Distances",
                labels={"x": "Distance", "y": "Count"},
                nbins=20
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No distance data available for visualization")
    
    def _render_history_tab(self):
        """Render reassignment history"""
        st.header("📈 Reassignment History")
        
        history = self.session_state.get_history()
        
        if not history:
            st.info("📋 No reassignment history yet. Run some optimizations to see results here.")
            
            if st.button("🗑️ Clear History"):
                self.session_state.clear_history()
                st.rerun()
            return
        
        # History metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_quality = sum(h.result.quality_score for h in history) / len(history)
            st.metric("📊 Avg Quality Score", f"{avg_quality:.2f}")
        
        with col2:
            total_dogs = sum(h.result.dogs_affected for h in history)
            st.metric("🐕 Total Dogs Reassigned", total_dogs)
        
        with col3:
            avg_distance = sum(h.result.total_distance for h in history) / len(history)
            st.metric("📏 Avg Total Distance", f"{avg_distance:.2f}")
        
        with col4:
            success_rate = sum(1 for h in history if h.result.success) / len(history) * 100
            st.metric("✅ Success Rate", f"{success_rate:.1f}%")
        
        # History table
        history_data = []
        for i, entry in enumerate(reversed(history)):  # Most recent first
            history_data.append({
                "Run": len(history) - i,
                "Timestamp": entry.timestamp.strftime("%m/%d %H:%M:%S"),
                "Driver": entry.driver_called_out,
                "Groups": ", ".join(map(str, entry.groups_affected)),
                "Strategy": entry.result.strategy_used,
                "Dogs Affected": entry.result.dogs_affected,
                "Quality Score": round(entry.result.quality_score, 3),
                "Total Distance": round(entry.result.total_distance, 3),
                "Success": "✅" if entry.result.success else "❌"
            })
        
        st.dataframe(pd.DataFrame(history_data), use_container_width=True)
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            self.session_state.clear_history()
            st.rerun()
    
    def _render_data_explorer_tab(self):
        """Render data exploration interface"""
        st.header("🔍 Data Explorer")
        
        dm = self.session_state.data_manager
        
        tab1, tab2, tab3 = st.tabs(["🐕 Dogs", "🚗 Drivers", "🗺️ Distance Matrix"])
        
        with tab1:
            self._render_dogs_explorer(dm)
        
        with tab2:
            self._render_drivers_explorer(dm)
        
        with tab3:
            self._render_distance_explorer(dm)
    
    def _render_dogs_explorer(self, dm):
        """Render dogs data explorer"""
        dogs_data = []
        for dog_id, dog in dm.dogs.items():
            current_driver = dm.current_assignments.get(dog_id, "Unknown")
            dogs_data.append({
                "Dog ID": dog.dog_id,
                "Name": dog.name,
                "Groups": ", ".join(map(str, sorted(dog.groups))),
                "Number of Dogs": dog.num_dogs,
                "Current Driver": current_driver,
                "Address": dog.address
            })
        
        if dogs_data:
            st.dataframe(pd.DataFrame(dogs_data), use_container_width=True)
            
            # Export option
            csv = pd.DataFrame(dogs_data).to_csv(index=False)
            st.download_button(
                "📥 Download Dogs Data",
                csv,
                "dogs_data.csv",
                "text/csv"
            )
    
    def _render_drivers_explorer(self, dm):
        """Render drivers data explorer"""
        drivers_data = []
        for driver_name, driver in dm.drivers.items():
            loads = dm.get_driver_current_loads(driver_name)
            callouts = ", ".join(map(str, sorted(driver.callouts))) if driver.callouts else "None"
            
            drivers_data.append({
                "Driver": driver_name,
                "Group 1 Cap": driver.get_capacity(1),
                "Group 1 Load": loads[1],
                "Group 1 Util%": round(loads[1] / driver.get_capacity(1) * 100, 1),
                "Group 2 Cap": driver.get_capacity(2),
                "Group 2 Load": loads[2],
                "Group 2 Util%": round(loads[2] / driver.get_capacity(2) * 100, 1),
                "Group 3 Cap": driver.get_capacity(3),
                "Group 3 Load": loads[3],
                "Group 3 Util%": round(loads[3] / driver.get_capacity(3) * 100, 1),
                "Called Out Groups": callouts,
                "Total Dogs": sum(loads.values())
            })
        
        if drivers_data:
            st.dataframe(pd.DataFrame(drivers_data), use_container_width=True)
    
    def _render_distance_explorer(self, dm):
        """Render distance matrix explorer"""
        if not dm.distance_matrix:
            st.info("Distance matrix not available")
            return
        
        st.write("🗺️ Distance Matrix Sample (first 10x10)")
        
        sample_ids = dm.distance_matrix.dog_ids[:10]
        if len(sample_ids) < 10:
            sample_ids = dm.distance_matrix.dog_ids
        
        sample_matrix = []
        for id1 in sample_ids:
            row = []
            for id2 in sample_ids:
                distance = dm.distance_matrix.get_distance(id1, id2)
                row.append(distance if distance != float('inf') else '∞')
            sample_matrix.append(row)
        
        sample_df = pd.DataFrame(sample_matrix, index=sample_ids, columns=sample_ids)
        st.dataframe(sample_df, use_container_width=True)
    
    def _load_data(self, map_url: str, matrix_url: str):
        """Load data from URLs"""
        try:
            with st.spinner("🔄 Loading data..."):
                data_manager = create_data_manager(map_url, matrix_url)
                
                if data_manager:
                    self.session_state.data_manager = data_manager
                    st.success("✅ Data loaded successfully!")
                    st.rerun()
                else:
                    self.session_state.set_error("Failed to load data. Please check URLs and try again.")
        
        except Exception as e:
            self.session_state.set_error(f"Error loading data: {e}")
    
    def _clear_data(self):
        """Clear all data and reset state"""
        self.session_state.data_manager = None
        self.session_state.clear_history()
        st.success("🗑️ Data cleared successfully!")
        st.rerun()
    
    def _calculate_average_distance(self) -> float:
        """Calculate average distance to nearest neighbor"""
        dm = self.session_state.data_manager
        
        if not dm.distance_matrix or not dm.dogs:
            return 0.0
        
        distances = []
        sample_dogs = list(dm.dogs.keys())[:20]  # Sample for performance
        
        for dog_id in sample_dogs:
            neighbors = dm.distance_matrix.get_neighbors(dog_id, max_distance=2.0)
            if neighbors:
                distances.append(neighbors[0][1])  # Closest neighbor
        
        return sum(distances) / len(distances) if distances else 0.0


# DEBUG FUNCTIONS ADDED BELOW

def debug_live_data_mismatch(data_manager):
    """Debug mismatches between today's dogs and master distance matrix"""
    
    st.header("🔄 Live Data Subset Analysis")
    
    if not data_manager:
        st.error("No data manager loaded")
        return
    
    # Get today's dogs vs matrix dogs
    todays_dogs = set(data_manager.dogs.keys())
    matrix_dogs = set(data_manager.distance_matrix.dog_ids) if data_manager.distance_matrix else set()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Today's Dogs", len(todays_dogs))
    with col2:
        st.metric("Matrix Dogs", len(matrix_dogs))
    with col3:
        overlap = len(todays_dogs & matrix_dogs)
        st.metric("Overlap", overlap)
    
    # Find mismatches
    st.subheader("📊 Data Overlap Analysis")
    
    # Dogs in today's assignment but NOT in distance matrix
    missing_from_matrix = todays_dogs - matrix_dogs
    if missing_from_matrix:
        st.error(f"❌ {len(missing_from_matrix)} dogs scheduled today are MISSING from distance matrix:")
        with st.expander("Missing Dogs (this causes 0.5 distance fallbacks)"):
            for i, dog_id in enumerate(sorted(missing_from_matrix)):
                if i < 20:  # Show first 20
                    dog = data_manager.dogs.get(dog_id)
                    st.write(f"• {dog_id} - {dog.name if dog else 'Unknown'}")
                elif i == 20:
                    st.write(f"... and {len(missing_from_matrix) - 20} more")
                    break
    else:
        st.success("✅ All today's dogs found in distance matrix!")
    
    # Show sample matches to verify ID formats
    matches = todays_dogs & matrix_dogs
    if matches:
        st.subheader("✅ Sample Matching Dogs")
        sample_matches = list(matches)[:5]
        for dog_id in sample_matches:
            dog = data_manager.dogs[dog_id]
            st.write(f"• {dog_id} - {dog.name} ✓")
    
    return missing_from_matrix

def debug_driver_availability(data_manager, called_out_driver: str, affected_groups):
    """Debug why drivers might be showing as NaN"""
    
    st.subheader(f"🚗 Driver Availability for {called_out_driver} callout")
    
    if called_out_driver not in data_manager.drivers:
        st.error(f"❌ Driver '{called_out_driver}' not found!")
        st.write("Available drivers:", list(data_manager.drivers.keys())[:10])
        return
    
    # Find dogs that need reassignment
    dogs_needing_reassignment = []
    for dog_id, assigned_driver in data_manager.current_assignments.items():
        if assigned_driver == called_out_driver and dog_id in data_manager.dogs:
            dog = data_manager.dogs[dog_id]
            if any(group in affected_groups for group in dog.groups):
                dogs_needing_reassignment.append(dog)
    
    st.write(f"Dogs needing reassignment: {len(dogs_needing_reassignment)}")
    
    if not dogs_needing_reassignment:
        st.warning("No dogs need reassignment - check if driver name or groups are correct")
        return
    
    # Analyze each driver's availability
    st.subheader("Driver Capacity Analysis")
    
    available_count = 0
    driver_analysis = []
    
    for driver_name, driver in data_manager.drivers.items():
        if driver_name == called_out_driver:
            continue
        
        # Get current loads
        current_loads = data_manager.get_driver_current_loads(driver_name)
        
        analysis = {
            'name': driver_name,
            'total_callouts': len(driver.callouts),
            'can_accept_any': False,
            'issues': []
        }
        
        # Check if completely called out
        if len(driver.callouts) >= 3:
            analysis['issues'].append("Called out all groups")
        
        # Check capacity for sample dog
        if dogs_needing_reassignment:
            sample_dog = dogs_needing_reassignment[0]
            can_accept = True
            
            for group in sample_dog.groups:
                if group in driver.callouts:
                    analysis['issues'].append(f"Called out group {group}")
                    can_accept = False
                    break
                
                capacity = driver.get_capacity(group)
                current_load = current_loads.get(group, 0)
                available = capacity - current_load
                
                if available < sample_dog.num_dogs:
                    analysis['issues'].append(f"Group {group}: {available}/{capacity} available, need {sample_dog.num_dogs}")
                    can_accept = False
            
            analysis['can_accept_any'] = can_accept
            if can_accept:
                available_count += 1
        
        driver_analysis.append(analysis)
    
    # Display results
    st.write(f"**Available drivers: {available_count}/{len(driver_analysis)}**")
    
    if available_count == 0:
        st.error("🚨 NO AVAILABLE DRIVERS - This is why you're getting NaN!")
        st.write("**Reasons:**")
        
        issue_summary = {}
        for analysis in driver_analysis:
            for issue in analysis['issues']:
                issue_summary[issue] = issue_summary.get(issue, 0) + 1
        
        for issue, count in sorted(issue_summary.items(), key=lambda x: x[1], reverse=True):
            st.write(f"• {issue}: {count} drivers")
    
    # Show detailed driver status
    with st.expander("Detailed Driver Status"):
        for analysis in driver_analysis:
            status = "✅ Available" if analysis['can_accept_any'] else "❌ Unavailable"
            issues = ", ".join(analysis['issues']) if analysis['issues'] else "None"
            st.write(f"**{analysis['name']}**: {status} - {issues}")

def render_debug_tab(data_manager):
    """Debug tab for the main dashboard"""
    
    st.header("🔧 System Debug")
    
    if not data_manager:
        st.warning("Load data first to debug")
        return
    
    # Data mismatch analysis
    missing_dogs = debug_live_data_mismatch(data_manager)
    
    st.divider()
    
    # Driver availability test
    st.subheader("🎯 Test Driver Scenario")
    
    col1, col2 = st.columns(2)
    with col1:
        drivers = list(data_manager.drivers.keys())
        test_driver = st.selectbox("Test calling out driver:", drivers if drivers else ["No drivers"])
    
    with col2:
        test_groups = st.multiselect("Affected groups:", [1, 2, 3], default=[1, 2, 3])
    
    if test_driver and test_groups and st.button("🔍 Analyze This Scenario"):
        debug_driver_availability(data_manager, test_driver, test_groups)
    
    # Recommendations
    st.divider()
    st.subheader("💡 Recommendations")
    
    if missing_dogs:
        st.warning("**Distance Matrix Issue**: Some of today's dogs aren't in your distance matrix")
        st.write("**Solutions:**")
        st.write("1. Update your master distance matrix to include all current dogs")
        st.write("2. Or modify the code to handle missing dogs more gracefully")
        st.write("3. Check if dog IDs have different formats between sheets")
    
    st.info("**For NaN drivers**: Usually means no drivers available due to callouts or capacity limits")
