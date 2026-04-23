import subprocess
import sys
import time

# Define your apps and their target ports
apps = [
    {"file": "feature_extraction.py", "port": "8501"},
    {"file": "cluster_analysis.py", "port": "8502"},
    {"file": "hierarchical_analysis.py", "port": "8503"},
    {"file": "tree_visualization.py", "port": "8504"},
]

processes = []

def launch_apps():
    print("🚀 Launching CNN Embedding Analysis Suite...\n")
    
    for app in apps:
        print(f"Starting {app['file']} on port {app['port']}...")
        # Use Popen to run the process in the background
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", app["file"], "--server.port", app["port"]],
            stdout=subprocess.DEVNULL, # Hides the default Streamlit terminal output
            stderr=subprocess.DEVNULL  # Hides the default Streamlit error output
        )
        processes.append(process)
        
    print("\n✅ All pages are running independently in the background!")
    print(f"📍 Main Entry Point: http://localhost:{apps[0]['port']}")
    print("\nPress Ctrl+C to shut down all servers.")

def shutdown_apps():
    print("\n\n🛑 Shutting down all Streamlit servers...")
    for p in processes:
        p.terminate()
    print("✅ All servers stopped. Goodbye!")

if __name__ == "__main__":
    try:
        launch_apps()
        # Keep the main script alive so we can catch the Ctrl+C shutdown
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # This catches the Ctrl+C command in the terminal
        shutdown_apps()