"""
Enhanced UI Components Module
Complete Streamlit interface with cost tracking and mile optimization focus
Complete file ready for copy-paste
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict

from data_manager import create_data_manager
from enhanced_assignment_logic import create_enhanced_assignment_engine, EnhancedReassignmentResult, CostAwareAssignmentEngine
from assignment_logic import CapacityAnalyzer  # Keep existing analyzer
from config import get_map_url, get_matrix_url

@dataclass
class EnhancedHistoryEntry:
    """Enhanced history entry with cost tracking"""
    timestamp: datetime
    result: EnhancedReassignmentResult
    driver_called_out: str
    groups_affected: List[int]

class EnhancedSessionState:
    """Enhanced session state with cost tracking"""
    
    def __init__(self):
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize enhanced session state variables"""
        defaults = {
            'data_manager': None,
            'assignment_engine': None,
            'reassignment_history': [],
            'current_scenario': None,
            'data_last_loaded': None,
            'last_error': None,
            'total_cost_savings': 0.0,  # Track cumulative savings
            'total_miles_saved': 0.0,   # Track cumulative miles saved
            'optimization_mode': 'enhanced'  # Optimization mode
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
        # Use enhanced assignment engine
        st.session_state.assignment_engine = create_enhanced_assignment_engine(value) if value else None
        st.session_state.data_last_loaded = datetime.now()
    
    @property
    def assignment_engine(self) -> CostAwareAssignmentEngine:
        return st.session_state.assignment_engine
    
    def add_history_entry(self, entry: EnhancedHistoryEntry):
        """Add enhanced history entry and update cumulative metrics"""
        st.session_state.reassignment_history.append(entry)
        
        # Update cumulative savings
        if entry.result.success:
            st.session_state.total_cost_savings += entry.result.cost_savings
            st.session_state.total_miles_saved += entry.result.miles_saved
        
        # Keep only recent entries
        max_entries = 10
        if len(st.session_state.reassignment_history) > max_entries:
            st.session_state.reassignment_history = st.session_state.reassignment_history[-max_entries:]
    
    def get_history(self) -> List[EnhancedHistoryEntry]:
        """Get enhanced reassignment history"""
        return st.session_state.reassignment_history.copy()
    
    def get_cumulative_savings(self) -> Tuple[float, float]:
        """Get cumulative cost and mile savings"""
        return st.session_state.total_cost_savings, st.session_state.total_miles_saved
    
    def clear_history(self):
        """Clear history and reset cumulative metrics"""
        st.session_state.reassignment_history = []
        st.session_state.total_cost_savings = 0.0
        st.session_state.total_miles_saved = 0.0
    
    def set_error(self, error: str):
        """Set last error message"""
        st.session_state.last_error = error
    
    def get_error(self) -> Optional[str]:
        """Get and clear last error"""
        error = st.session_state.last_error
        st.session_state.last_error = None
        return error

class EnhancedDashboardUI:
    """Enhanced dashboard UI with cost optimization focus"""
    
    def __init__(self, session_state: EnhancedSessionState):
        self.session_state = session_state
        self._setup_enhanced_styling()
    
    def _setup_enhanced_styling(self):
        """Setup enhanced CSS styling with cost focus"""
        st.markdown("""
        <style>
        .cost-savings-metric {
            background: linear-gradient(90deg, #d4f6d4 0%, #a8e6a8 100%);
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            border-left: 4px solid #28a745;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .miles-metric {
            background: linear-gradient(90deg, #cce7ff 0%, #99d6ff 100%);
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            border-left: 4px solid #007bff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .warning-metric {
            background: linear-gradient(90deg, #fff3cd 0%, #ffe69c 100%);
            border-left-color: #ffc107;
        }
        .error-metric {
            background: linear-gradient(90deg, #f8d7da 0%, #f5c6cb 100%);
            border-left-color: #dc3545;
        }
        .big-number {
            font-size: 2.5rem;
            font-weight: bold;
            color: #28a745;
            text-align: center;
        }
        .cost-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render(self):
        """Render the enhanced dashboard"""
        st.title("🐶💰 Cost-Optimized Dog Reassignment System")
        
        # Cost savings header
        self._render_cost_savings_header()
        
        # Show any errors
        error = self.session_state.get_error()
        if error:
            st.error(f"❌ {error}")
        
        # Sidebar for data management
        self._render_enhanced_sidebar()
        
        # Main content area
        if self.session_state.data_manager is None:
            self._render_welcome_screen()
        else:
            self._render_enhanced_main_content()
    
    def _render_cost_savings_header(self):
        """Render cost savings header"""
        total_cost_savings, total_miles_saved = self.session_state.get_cumulative_savings()
        
        if total_cost_savings > 0:
            st.markdown(f"""
            <div class="cost-header">
                <h2>💰 Total Savings This Session: ${total_cost_savings:.2f}</h2>
                <h3>📏 Total Miles Saved: {total_miles_saved:.1f} miles</h3>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_enhanced_sidebar(self):
        """Render enhanced sidebar with optimization settings"""
        with st.sidebar:
            st.header("📊 Data & Optimization")
            
            # Cost optimization settings
            with st.expander("💰 Cost Optimization Settings", expanded=True):
                cost_per_mile = st.number_input(
                    "Cost per Mile ($)",
                    min_value=1.0,
                    max_value=50.0,
                    value=10.0,
                    step=1.0,
                    help="Cost assigned to each mile of driving"
                )
                
                optimization_mode = st.selectbox(
                    "Optimization Mode",
                    ["enhanced", "thorough", "speed"],
                    index=0,
                    help="Enhanced: Best balance, Thorough: Maximum quality, Speed: Fastest"
                )
                
                st.session_state.optimization_mode = optimization_mode
            
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
            
            # Enhanced data status
            if self.session_state.data_manager:
                self._render_enhanced_data_status()
    
    def _render_enhanced_data_status(self):
        """Render enhanced data status with cost metrics"""
        dm = self.session_state.data_manager
        engine = self.session_state.assignment_engine
        
        st.success(f"✅ Data loaded")
        
        # Basic metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🐕 Dogs", len(dm.dogs))
        with col2:
            st.metric("🚗 Drivers", len(dm.drivers))
        
        # System mile analysis
        if engine:
            analysis = engine.get_system_mile_analysis()
            
            st.markdown('<div class="miles-metric">', unsafe_allow_html=True)
            st.metric(
                "📏 System Total Miles", 
                f"{analysis['total_system_miles']:.1f}",
                help="Total miles all drivers need to travel"
            )
            st.metric(
                "💰 Daily Cost", 
                f"${analysis['total_daily_cost']:.2f}",
                help="Total daily cost at $10/mile"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Optimization opportunities
            opportunities = analysis.get('optimization_opportunities', [])
            if opportunities:
                st.warning(f"💡 {len(opportunities)} optimization opportunities")
                with st.expander("View Opportunities"):
                    for opp in opportunities[:3]:
                        st.text(f"• {opp}")
        
        if st.session_state.data_last_loaded:
            st.caption(f"Updated: {st.session_state.data_last_loaded.strftime('%H:%M:%S')}")
    
    def _render_enhanced_main_content(self):
        """Render enhanced main content with cost focus"""
        # Enhanced overview metrics
        self._render_enhanced_overview_metrics()
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🎯 Smart Reassignment", 
            "💰 Cost Analytics", 
            "📈 Savings History", 
            "🔍 Data Explorer", 
            "🔧 Debug"
        ])
        
        with tab1:
            self._render_enhanced_reassignment_tab()
        
        with tab2:
            self._render_cost_analytics_tab()
        
        with tab3:
            self._render_enhanced_history_tab()
        
        with tab4:
            self._render_data_explorer_tab()
        
        with tab5:
            self._render_debug_tab()
    
    def _render_enhanced_overview_metrics(self):
        """Render enhanced overview metrics with cost focus"""
        dm = self.session_state.data_manager
        engine = self.session_state.assignment_engine
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("🐕 Total Dogs", len(dm.dogs))
        
        with col2:
            st.metric("🚗 Active Drivers", len(dm.drivers))
        
        with col3:
            if engine:
                analysis = engine.get_system_mile_analysis()
                st.metric("📏 System Miles", f"{analysis['total_system_miles']:.1f}")
        
        with col4:
            if engine:
                analysis = engine.get_system_mile_analysis()
                st.metric("💰 Daily Cost", f"${analysis['total_daily_cost']:.0f}")
        
        with col5:
            total_savings, _ = self.session_state.get_cumulative_savings()
            delta_color = "normal" if total_savings > 0 else "off"
            st.metric("💵 Saved Today", f"${total_savings:.0f}", delta_color=delta_color)
    
    def _render_enhanced_reassignment_tab(self):
        """Render enhanced reassignment interface with cost preview"""
        st.header("🎯 Smart Cost-Optimized Reassignment")
        
        # Cost impact preview
        st.markdown("""
        <div class="cost-header">
            <h4>💡 Smart Optimization</h4>
            <p>This system minimizes total miles and prevents increased drive times while maintaining capacity constraints.</p>
        </div>
        """, unsafe_allow_html=True)
        
        dm = self.session_state.data_manager
        
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
            # Enhanced impact preview with cost estimates
            self._render_enhanced_impact_preview(selected_driver, affected_groups)
            
            # Optimization controls
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Optimize with Cost Focus", type="primary", use_container_width=True):
                    self._run_enhanced_optimization(selected_driver, affected_groups)
    
    def _render_enhanced_impact_preview(self, driver_name: str, groups: List[int]):
        """Render enhanced preview with cost estimates"""
        dm = self.session_state.data_manager
        engine = self.session_state.assignment_engine
        
        # Find affected dogs
        affected_dogs = []
        for dog_id, assigned_driver in dm.current_assignments.items():
            if assigned_driver == driver_name and dog_id in dm.dogs:
                dog = dm.dogs[dog_id]
                if any(group in groups for group in dog.groups):
                    affected_dogs.append(dog)
        
        if affected_dogs:
            # Cost impact estimate
            estimated_impact = len(affected_dogs) * 2.0 * 10.0  # Rough estimate
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🐕 Dogs to Reassign", len(affected_dogs))
            with col2:
                st.metric("💰 Potential Savings", f"${estimated_impact:.0f}")
            with col3:
                optimization_level = "🔥 Thorough" if estimated_impact >= 50 else "⚡ Standard"
                st.metric("🎯 Optimization Level", optimization_level)
            
            # Current system metrics
            if engine:
                analysis = engine.get_system_mile_analysis()
                st.info(f"📊 Current system: {analysis['total_system_miles']:.1f} miles, ${analysis['total_daily_cost']:.0f} daily cost")
            
            with st.expander("👀 Preview Affected Dogs"):
                preview_data = []
                for dog in affected_dogs:
                    preview_data.append({
                        "Dog ID": dog.dog_id,
                        "Name": dog.name,
                        "Groups": ", ".join(map(str, sorted(dog.groups))),
                        "Number of Dogs": dog.num_dogs,
                        "Current Driver": driver_name
                    })
                
                st.dataframe(pd.DataFrame(preview_data), use_container_width=True)
        else:
            st.warning("ℹ️ No dogs need reassignment for the selected driver and groups")
    
    def _run_enhanced_optimization(self, driver_name: str, groups: List[int]):
        """Run enhanced optimization and display cost-focused results"""
        engine = self.session_state.assignment_engine
        
        with st.spinner("🔄 Running cost-optimized algorithms..."):
            result = engine.reassign_dogs(driver_name, groups, st.session_state.optimization_mode)
        
        # Store in enhanced history
        history_entry = EnhancedHistoryEntry(
            timestamp=datetime.now(),
            result=result,
            driver_called_out=driver_name,
            groups_affected=groups
        )
        self.session_state.add_history_entry(history_entry)
        
        # Display enhanced results
        if result.success:
            self._display_enhanced_optimization_results(result)
        else:
            st.error(f"❌ Optimization failed: {result.error_message}")
    
    def _display_enhanced_optimization_results(self, result: EnhancedReassignmentResult):
        """Display enhanced optimization results with cost focus"""
        # Success message with savings highlight
        if result.cost_savings > 0:
            st.success(f"✅ Optimization Complete! Saved ${result.cost_savings:.2f} and {result.miles_saved:.1f} miles")
        else:
            st.info(f"✅ Optimization Complete! {result.dogs_affected} dogs reassigned efficiently")
        
        # Enhanced metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            delta_miles = result.miles_saved
            st.metric(
                "💰 Cost Savings", 
                f"${result.cost_savings:.2f}",
                delta=f"${delta_miles * 10:.2f}" if delta_miles > 0 else None
            )
        
        with col2:
            st.metric(
                "📏 Miles Saved", 
                f"{result.miles_saved:.2f}",
                delta=f"-{result.miles_saved:.2f}" if result.miles_saved > 0 else None
            )
        
        with col3:
            drive_time_change = result.avg_drive_time_after - result.avg_drive_time_before
            st.metric(
                "⏱️ Drive Time Change",
                f"{drive_time_change:+.2f}",
                delta=f"{drive_time_change:+.2f}"
            )
        
        with col4:
            st.metric("🎯 Quality Score", f"{result.quality_score:.2f}")
        
        # System before/after comparison
        st.subheader("📊 System Impact Comparison")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**BEFORE Optimization:**")
            st.write(f"• Total miles: {result.baseline_total_miles:.2f}")
            st.write(f"• Average drive time: {result.avg_drive_time_before:.2f}")
            st.write(f"• Daily cost: ${result.baseline_total_miles * 10:.2f}")
        
        with col2:
            st.markdown("**AFTER Optimization:**")
            st.write(f"• Total miles: {result.system_total_miles:.2f}")
            st.write(f"• Average drive time: {result.avg_drive_time_after:.2f}")
            st.write(f"• Daily cost: ${result.system_total_miles * 10:.2f}")
        
        # Enhanced results table
        if result.assignments:
            st.subheader("📋 Optimized Assignments")
            df = pd.DataFrame(result.assignments)
            st.dataframe(df, use_container_width=True)
            
            # Cost-benefit report
            with st.expander("💰 Detailed Cost-Benefit Report"):
                st.text(result.get_cost_benefit_report())
            
            # Download button with enhanced filename
            csv = df.to_csv(index=False)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"cost_optimized_reassignments_{timestamp}.csv"
            st.download_button(
                label="📥 Download Cost-Optimized Results",
                data=csv,
                file_name=filename,
                mime="text/csv"
            )
    
    def _render_cost_analytics_tab(self):
        """Render cost analytics and optimization insights"""
        st.header("💰 Cost Analytics & Optimization Insights")
        
        dm = self.session_state.data_manager
        engine = self.session_state.assignment_engine
        
        if not engine:
            st.warning("Load data to view cost analytics")
            return
        
        # System cost analysis
        analysis = engine.get_system_mile_analysis()
        
        # Cost overview cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="cost-savings-metric">', unsafe_allow_html=True)
            st.metric("💰 Daily Cost", f"${analysis['total_daily_cost']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            weekly_cost = analysis['total_daily_cost'] * 5
            st.metric("📅 Weekly Cost", f"${weekly_cost:.2f}")
        
        with col3:
            annual_cost = analysis['total_daily_cost'] * 250
            st.metric("📆 Annual Cost", f"${annual_cost:,.0f}")
        
        with col4:
            total_savings, _ = self.session_state.get_cumulative_savings()
            st.metric("💵 Session Savings", f"${total_savings:.2f}")
        
        # Driver efficiency analysis
        st.subheader("🚗 Driver Efficiency Analysis")
        
        driver_data = []
        for driver_name, driver_info in analysis['driver_breakdown'].items():
            efficiency = driver_info['efficiency']
            daily_cost = driver_info['route_miles'] * 10.0
            
            # Efficiency rating
            if efficiency <= 1.0:
                rating = "🟢 Excellent"
            elif efficiency <= 1.5:
                rating = "🟡 Good"
            else:
                rating = "🔴 Needs Optimization"
            
            driver_data.append({
                "Driver": driver_name,
                "Dogs": driver_info['dogs'],
                "Route Miles": round(driver_info['route_miles'], 2),
                "Daily Cost": f"${daily_cost:.2f}",
                "Efficiency": round(efficiency, 2),
                "Rating": rating
            })
        
        if driver_data:
            df = pd.DataFrame(driver_data)
            st.dataframe(df, use_container_width=True)
            
            # Efficiency chart
            fig = px.bar(
                df, 
                x="Driver", 
                y="Efficiency",
                color="Efficiency",
                title="Driver Efficiency (Miles per Dog)",
                color_continuous_scale="RdYlGn_r"
            )
            fig.add_hline(y=1.5, line_dash="dash", line_color="red", annotation_text="Optimization Threshold")
            st.plotly_chart(fig, use_container_width=True)
        
        # Optimization opportunities
        opportunities = analysis.get('optimization_opportunities', [])
        if opportunities:
            st.subheader("💡 Optimization Opportunities")
            for i, opp in enumerate(opportunities):
                st.write(f"{i+1}. {opp}")
    
    def _render_enhanced_history_tab(self):
        """Render enhanced history with cost tracking"""
        st.header("📈 Cost Savings History")
        
        history = self.session_state.get_history()
        
        if not history:
            st.info("📋 No optimization history yet. Run some cost optimizations to see savings here.")
            return
        
        # Enhanced history metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_cost_savings = sum(h.result.cost_savings for h in history)
        total_miles_saved = sum(h.result.miles_saved for h in history)
        avg_quality = sum(h.result.quality_score for h in history) / len(history)
        
        with col1:
            st.markdown('<div class="cost-savings-metric">', unsafe_allow_html=True)
            st.metric("💰 Total Saved", f"${total_cost_savings:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="miles-metric">', unsafe_allow_html=True)
            st.metric("📏 Miles Saved", f"{total_miles_saved:.1f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            success_count = sum(1 for h in history if h.result.success and h.result.cost_savings > 0)
            st.metric("🎯 Successful Optimizations", f"{success_count}/{len(history)}")
        
        with col4:
            st.metric("📊 Avg Quality Score", f"{avg_quality:.2f}")
        
        # Savings trend chart
        if len(history) > 1:
            st.subheader("💰 Cost Savings Trend")
            
            trend_data = []
            cumulative_savings = 0
            
            for i, entry in enumerate(history):
                cumulative_savings += entry.result.cost_savings
                trend_data.append({
                    "Run": i + 1,
                    "Session Savings": entry.result.cost_savings,
                    "Cumulative Savings": cumulative_savings,
                    "Miles Saved": entry.result.miles_saved
                })
            
            df = pd.DataFrame(trend_data)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["Run"], 
                y=df["Cumulative Savings"],
                name="Cumulative Savings ($)",
                line=dict(color="green", width=3)
            ))
            fig.add_trace(go.Bar(
                x=df["Run"], 
                y=df["Session Savings"],
                name="Per-Session Savings ($)",
                opacity=0.6
            ))
            
            fig.update_layout(
                title="Cost Savings Over Time",
                xaxis_title="Optimization Run",
                yaxis_title="Savings ($)",
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced history table
        history_data = []
        for i, entry in enumerate(reversed(history)):  # Most recent first
            history_data.append({
                "Run": len(history) - i,
                "Time": entry.timestamp.strftime("%m/%d %H:%M"),
                "Driver": entry.driver_called_out,
                "Groups": ", ".join(map(str, entry.groups_affected)),
                "Dogs": entry.result.dogs_affected,
                "Cost Saved": f"${entry.result.cost_savings:.2f}",
                "Miles Saved": f"{entry.result.miles_saved:.1f}",
                "Drive Time Δ": f"{entry.result.avg_drive_time_after - entry.result.avg_drive_time_before:+.2f}",
                "Strategy": entry.result.strategy_used,
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
    
    def _render_debug_tab(self):
        """Render debug interface"""
        st.header("🔧 System Debug")
        
        dm = self.session_state.data_manager
        
        if not dm:
            st.warning("Load data first to debug")
            return
        
        # Data mismatch analysis
        self._debug_live_data_mismatch(dm)
        
        st.divider()
        
        # Driver availability test
        st.subheader("🎯 Test Driver Scenario")
        
        col1, col2 = st.columns(2)
        with col1:
            drivers = list(dm.drivers.keys())
            test_driver = st.selectbox("Test calling out driver:", drivers if drivers else ["No drivers"])
        
        with col2:
            test_groups = st.multiselect("Affected groups:", [1, 2, 3], default=[1, 2, 3])
        
        if test_driver and test_groups and st.button("🔍 Analyze This Scenario"):
            self._debug_driver_availability(dm, test_driver, test_groups)
    
    def _debug_live_data_mismatch(self, data_manager):
        """Debug mismatches between today's dogs and master distance matrix"""
        
        st.header("🔄 Live Data Subset Analysis")
        
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
        
        return missing_from_matrix
    
    def _debug_driver_availability(self, data_manager, called_out_driver: str, affected_groups):
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
    
    def _render_welcome_screen(self):
        """Render enhanced welcome screen"""
        st.info("👈 Please load data using the sidebar to get started")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 💰 Cost-Optimized Features
            1. **Smart Mile Reduction**: Minimizes total system miles
            2. **Drive Time Protection**: Prevents increased average drive times
            3. **Cost Tracking**: Real-time cost savings calculation
            4. **Multiple Algorithms**: Hungarian, Greedy, Mile-First optimization
            5. **Impact Analysis**: Before/after system comparison
            """)
        
        with col2:
            st.markdown("""
            ### 🎯 Optimization Goals
            - ✅ **Minimize total miles** across all drivers
            - ✅ **Maintain or reduce** average drive time per driver
            - ✅ **Track cost savings** in real-time ($10/mile)
            - ✅ **Respect capacity** and callout constraints
            - ✅ **Provide transparency** with detailed reporting
            """)
    
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

# Factory functions for backward compatibility
SessionState = EnhancedSessionState
DashboardUI = EnhancedDashboardUI
