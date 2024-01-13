import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import requests
import json
import csv
import sklearn.ensemble
import sklearn.metrics
from datetime import datetime

modules = [pd, np, joblib, plt, requests, json, csv, sklearn.ensemble, sklearn.metrics, datetime]

for module in modules:
    try:
        print(f"{module.__name__} version: {module.__version__}")
    except AttributeError:
        print(f"{module.__name__} doesn't have a __version__ attribute.")

# pandas version: 2.0.2
# numpy version: 1.24.3
# joblib version: 1.2.0
# requests version: 2.31.0
# json version: 2.0.9
# csv version: 1.0
# sklearn version 1.2.2