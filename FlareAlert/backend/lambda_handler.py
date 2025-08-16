import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from mangum import Mangum
from app.main import app

handler = Mangum(app, lifespan="off")
