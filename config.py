import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

SENSOR_COLS = ["volt", "rotate", "pressure", "vibration"]
FAILURE_COMPONENTS = ["comp1", "comp2", "comp3", "comp4"]
ROLLING_WINDOWS = [3, 24, 72]
FAILURE_HORIZON = 24
RANDOM_STATE = 42
