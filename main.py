import subprocess
import sys
import time

apps = [
    {"file": "feature_extraction.py", "port": "8501"},
    {"file": "cluster_analysis.py", "port": "8502"},
    {"file": "hierarchical_analysis.py", "port": "8503"},
    {"file": "tree_visualization.py", "port": "8504"},
]

processes = []

def launch_apps():
    print("Launching CNN Embedding Analysis Suite...\n")
    
    for app in apps:
        print(f"Starting {app['file']} on port {app['port']}...")
        
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", app["file"], "--server.port", app["port"]],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        processes.append(process)
        
        time.sleep(1)
        
    print("\nAll pages are running independently in the background!")
    print(f"Main Entry Point: http://localhost:{apps[0]['port']}")
    print("\nPress Ctrl+C to shut down all servers.")

def shutdown_apps():
    print("\n\nShutting down all Streamlit servers...")
    for p in processes:
        p.terminate()
    print("All servers stopped. Goodbye!")

if __name__ == "__main__":
    try:
        launch_apps()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown_apps()