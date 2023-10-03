import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import random

from sklearn.model_selection import train_test_split
import torch
import sys

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import plotly.express as px

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

import os
import ramanspy as rp
from ramanspy.preprocessing import baseline, normalise
from ramanspy.preprocessing.denoise import SavGol 
from ramanspy import Spectrum

from alive_progress import alive_bar