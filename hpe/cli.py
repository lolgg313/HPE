import sys
import os
import shutil
import importlib.util
import importlib.resources
import subprocess

def show_help():
    help_text = """
Usage: hpe [command]

Commands:
  help, (empty)   Show this help message
  run             Run the HPE engine
  get             Copy hpe.py to current directory

Examples:
  hpe
  hpe help
  hpe run
  hpe get
"""
    print(help_text)

def run_code():
    try:
        with importlib.resources.path("hpe.code", "hpe.py") as hpe_path:
            result = subprocess.run([sys.executable, str(hpe_path)])
            if result.returncode != 0:
                print("HPE exited with error code:", result.returncode)
    except Exception as e:
        print(f"Error running HPE: {e}")

def get_code():
    try:
        with importlib.resources.path("hpe.code", "hpe.py") as hpe_path:
            dest_path = os.path.join(os.getcwd(), "hpe.py")
            shutil.copyfile(hpe_path, dest_path)
            print(f"hpe.py copied to: {dest_path}")
    except Exception as e:
        print(f"Error copying hpe.py: {e}")

def main():
    if len(sys.argv) < 2 or sys.argv[1] == "help":
        show_help()
    elif sys.argv[1] == "run":
        run_code()
    elif sys.argv[1] == "get":
        get_code()
    else:
        print(f"Unknown command: {sys.argv[1]}")
        show_help()

if __name__ == "__main__":
    main()
