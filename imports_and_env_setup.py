# imports_and_env_setup.py

import os, json, platform
import joblib
import os
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
import lightgbm as lgb

from datetime import datetime, timezone
from sklearn import set_config
from sklearn.datasets import fetch_openml
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.base import BaseEstimator, TransformerMixin

import category_encoders as ce

set_config(transform_output="pandas")
