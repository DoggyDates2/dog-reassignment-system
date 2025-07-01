"""
🐶💰 Enhanced Cost-Optimized Dog Reassignment System
Main Streamlit Application with Advanced Mile and Cost Optimization
Complete file ready for copy-paste
"""

import streamlit as st
import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

# Enhanced imports
try:
    from enhanced_ui_components import EnhancedSessionState, EnhancedDashboardUI
    from config import APP_CONFIG
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please check that all enhanced files are correctly placed in src/")
    st.error("Required files: enhanced_optimization_engine.py, enhanced_assignment_logic.py, enhanced_ui_components.py")
    st.stop()

def main():
    """Main application entry point with enhanced cost optimization"""
    
    # Configure page with enhanced title
    st.set_page_config(
        page_title="🐶💰 Cost-Optimized Dog Reassignment System",
        page_icon="💰",
        layout=APP_CONFIG["layout"],
        initial_sidebar_state=APP_CONFIG["sidebar_state"]
    )
    
    # Initialize enhanced session state and UI
    try:
        session_state = EnhancedSessionState()
        dashboard = EnhancedDashboardUI(session_state)
    except Exception as e:
        st.error(f"❌ Failed to initialize enhanced components: {e}")
        st.error("Make sure all enhanced files are correctly placed in src/ directory")
        st.stop()
    
    # Render the enhanced application
    try:
        dashboard.render()
    except Exception as e:
        st.error(f"❌ Application Error: {e}")
        st.info("Please check the logs and try refreshing the page.")
        
        # Show error details in expander for debugging
        with st.expander("🔍 Error Details (for debugging)"):
            st.exception(e)
            
        # Show troubleshooting tips
        with st.expander("🛠️ Troubleshooting Tips"):
            st.markdown("""
            **Common Issues:**
            1. **Import errors**: Make sure all enhanced files are in src/ directory
            2. **Data loading errors**: Check your Google Sheets URLs
            3. **Optimization failures**: Verify driver capacity and callout constraints
            
            **Required Files in src/ directory:**
            - `enhanced_optimization_engine.py`
            - `enhanced_assignment_logic.py` 
            - `enhanced_ui_components.py`
            - `data_manager.py` (existing)
            - `assignment_logic.py` (existing)
            - `config.py` (existing)
            """)

if __name__ == "__main__":
    main()
