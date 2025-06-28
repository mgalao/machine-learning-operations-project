import sys
from pathlib import Path
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
# import folium
# from folium.plugins import MarkerCluster
from difflib import get_close_matches
import plotly.express as px
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
from kneed import KneeLocator
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pathlib import Path
from typing import List, Tuple, Union