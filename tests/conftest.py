import os
import tempfile
# Use a fresh database and never modify user documents during tests.
os.environ["DOCU_DATA"]=tempfile.mkdtemp(prefix="docuconflict-tests-")
os.environ["DOCU_OFFLINE"]="true"
