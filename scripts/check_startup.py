import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# Dedicated verification ports avoid interrupting servers already used by the user.
ROOT=Path(__file__).resolve().parents[1]
def main():
    children=[]
    creationflags=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0
    try:
        commands=[([sys.executable,"-m","uvicorn","backend.main:app","--host","127.0.0.1","--port","18000"],"http://127.0.0.1:18000/health"),
            ([sys.executable,"-m","streamlit","run","frontend/app.py","--server.address","127.0.0.1","--server.port","18501","--server.headless","true","--browser.gatherUsageStats","false"],"http://127.0.0.1:18501/_stcore/health")]
        for args,url in commands:
            child=subprocess.Popen(args,cwd=ROOT,creationflags=creationflags,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            children.append(child)
            deadline=time.monotonic()+60
            while time.monotonic()<deadline:
                if child.poll() is not None:raise RuntimeError(f"Server exited with {child.returncode}: {args}")
                try:
                    with urllib.request.urlopen(url,timeout=2) as response:
                        assert response.status==200
                        print(url,response.read().decode())
                        break
                except (OSError,TimeoutError):time.sleep(.5)
            else:raise RuntimeError(f"Timed out waiting for {url}")
        print("FastAPI and Streamlit startup checks passed")
    finally:
        for child in children:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()

if __name__=="__main__":main()
