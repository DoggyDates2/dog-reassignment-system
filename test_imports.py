import streamlit as st

try:
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent / "src"
    sys.path.append(str(src_path))
    
    st.write("Testing imports...")
    
    try:
        import config
        st.write("✅ config imported successfully")
    except Exception as e:
        st.write(f"❌ config error: {e}")
    
    try:
        import data_manager
        st.write("✅ data_manager imported successfully")
    except Exception as e:
        st.write(f"❌ data_manager error: {e}")
    
    try:
        import optimization_engine
        st.write("✅ optimization_engine imported successfully")
    except Exception as e:
        st.write(f"❌ optimization_engine error: {e}")
    
    try:
        import assignment_logic
        st.write("✅ assignment_logic imported successfully")
    except Exception as e:
        st.write(f"❌ assignment_logic error: {e}")
    
    try:
        import ui_components
        st.write("✅ ui_components imported successfully")
    except Exception as e:
        st.write(f"❌ ui_components error: {e}")

except Exception as e:
    st.write(f"Setup error: {e}")
