 """
🐶 Optimized Dog Reassignment System
Main Streamlit Application

Author: Your Name
Description: Multi-algorithm optimization system for dog assignment logistics
"""

import streamlit as st
import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

try:
    from ui_components import SessionState, DashboardUI
    from config import APP_CONFIG
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure all files are uploaded correctly and refresh the page.")
    st.stop()

def main():
    """Main application entry point"""
    
    # Configure page
    st.set_page_config(
        page_title=APP_CONFIG["page_title"],
        page_icon=APP_CONFIG["page_icon"],
        layout=APP_CONFIG["layout"],
        initial_sidebar_state=APP_CONFIG["sidebar_state"]
    )
    
    # Initialize session state and UI
    session_state = SessionState()
    dashboard = DashboardUI(session_state)
    
    # Render the application
    try:
        dashboard.render()
    except Exception as e:
        st.error(f"❌ Application Error: {e}")
        st.info("Please check the logs and try refreshing the page.")
        
        # Show error details in expander for debugging
        with st.expander("🔍 Error Details (for debugging)"):
            st.exception(e)

if __name__ == "__main__":
    main()
