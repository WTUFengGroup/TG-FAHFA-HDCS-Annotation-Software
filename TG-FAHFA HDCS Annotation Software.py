#############################################################################
#  TG-FAHFA HDCS Annotation Software
#############################################################################
#  Single-file refactored build — merges candidate generation,
#  EAD fragment prediction, spectrum matching, scoring, and GUI.
#############################################################################

import os
import csv
import json
import re
import itertools
import math
from copy import deepcopy
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Set, Sequence, Iterable

# Release metadata
APP_NAME = "TG-FAHFA HDCS Annotation Software"
APP_VERSION = "1.0.0"
APP_VERSION_TAG = f"v{APP_VERSION}"
TARGET_HDCS_SCORE = 1.1
DISPLAY_MIN_HDCS_SCORE = 1.0
DISPLAY_MAX_HDCS_SCORE = 1.1
NEP_ZERO_TOLERANCE = 1e-9


# Some portable Python runtimes omit Tcl's script library.  When setup_env.bat
# has copied it into the project environment, point tkinter at that local copy.
_PROJECT_TCL_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "tcl")
if os.path.isdir(_PROJECT_TCL_ROOT):
    os.environ.setdefault("TCL_LIBRARY", os.path.join(_PROJECT_TCL_ROOT, "tcl8.6"))
    os.environ.setdefault("TK_LIBRARY", os.path.join(_PROJECT_TCL_ROOT, "tk8.6"))

import numpy as np
import pandas as pd
from scipy.optimize import nnls as scipy_nnls
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

#############################################################################
#  CONFIGURATION AND CONSTANTS
#############################################################################
# Precise atomic masses (IUPAC 2021)
ATOMIC_MASSES: Dict[str, float] = {
    "H": 1.00782503223, "C": 12.00000000000, "N": 14.00307400443,
    "O": 15.99491461957, "P": 30.97376199840, "S": 31.97207117440,
    "Na": 22.98976928200, "Cl": 34.96885268200, "Br": 78.91833760000,
}
ELECTRON_MASS = 0.000548579909
PROTON_MASS = ATOMIC_MASSES["H"] - ELECTRON_MASS
NA_MASS = ATOMIC_MASSES["Na"]

# Backward-compatible mass alias used by candidate generation
MASS = {k: ATOMIC_MASSES[k] for k in ("C", "H", "O", "Na")}


def calculate_fa3_protonated_mz(carbon: int, double_bonds: int) -> float:
    """Return exact [FA3+H]+ m/z for a neutral Cn:H(2n-2db)O2 fatty acid."""
    carbon = int(carbon)
    double_bonds = int(double_bonds)
    neutral_mass = (
        carbon * ATOMIC_MASSES["C"]
        + (2 * carbon - 2 * double_bonds) * ATOMIC_MASSES["H"]
        + 2 * ATOMIC_MASSES["O"]
    )
    return neutral_mass + PROTON_MASS

# Fragment matching tolerance (ppm)
PPM = 25
PPM_PARENT = 20

# EAD fragmentation offsets (Da)
EAD_OFFSETS = {
    "cho2_loss":  -44.997654,
    "c2h2o":       42.010565,
    "c3h4":        40.031300,
    "ch_radical":  13.007825,
    "c2h2":        26.015650,
}

SUBSCRIPT_TRANSLATION = str.maketrans({
    "0": "₀", "1": "₁", "2": "₂", "3": "₃", "4": "₄",
    "5": "₅", "6": "₆", "7": "₇", "8": "₈", "9": "₉", "x": "ₓ",
})

FRAGMENT_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "precursor": {
        "type": "precursor", "color": "#0BB4F2", "default_intensity": 100.0,
        "is_key": False, "annotation_template": "[M+Na]+",
    },
    "neutral_loss_fahfa": {
        "type": "neutral_loss_fahfa", "color": "#0BB4F2", "default_intensity": 35.0,
        "intensity_by_sn": {"sn2": 12.5}, "is_key": True,
        "annotation_template": "[M-{label}+Na]+",
    },
    "neutral_loss": {
        "type": "neutral_loss", "color": "#0BB4F2", "default_intensity": 35.0,
        "intensity_by_sn": {"sn3": 17.5}, "is_key": True,
        "annotation_template": "[M-{label}+Na]+",
    },
    "neutral_loss_branch": {
        "type": "neutral_loss_branch", "color": "#0BB4F2", "default_intensity": 40.0,
        "is_key": True, "annotation_template": "[M-{branch}+Na]+",
    },
    "neutral_loss_combined": {
        "type": "neutral_loss_combined", "color": "#0BB4F2", "default_intensity": 15.0,
        "is_key": True, "annotation_template": "[M-{branch}-{label}+Na]+",
    },
    "fahfa_precursor": {
        "type": "fahfa_precursor", "color": "#0BB4F2", "default_intensity": 30.0,
        "is_key": True, "annotation_template": "[FA3HFA+Na]+",
    },
    "fa3_protonated": {
        "type": "fa3_protonated", "color": "#E54A5B", "default_intensity": 50.0,
        "is_key": True, "annotation_template": "[FA3+H]+",
    },
    "hfa_dehydrated_adduct_c2h2o": {
        "type": "hfa_dehydrated_adduct_c2h2o", "color": "#E54A5B",
        "default_intensity": 7.5, "is_key": True,
        "annotation_template": "[HFA-H2O+C2H2O+Na]+", "offset": "c2h2o",
    },
    "hfa_dehydrated_adduct_c3h4": {
        "type": "hfa_dehydrated_adduct_c3h4", "color": "#E54A5B",
        "default_intensity": 5.0, "is_key": True,
        "annotation_template": "[HFA-H2O+C3H4+Na]+", "offset": "c3h4",
    },
    "hfa_dehydrated_sn2_c2h2": {
        "type": "hfa_dehydrated_sn2_c2h2", "color": "#FF683A",
        "default_intensity": 7.5, "is_key": True,
        "annotation_template": "[{label}-H2O+C2H2+Na]+", "offset": "c2h2",
    },
    "ead_diagnostic": {
        "type": "ead_diagnostic", "color": "#E54A5B", "color_by_fahfa_sn": {"sn2": "#FF683A"},
        "default_intensity": 27.5, "is_key": True, "annotation_template": "[FA3HFA-CHO2•+Na]+",
        "offset": "cho2_loss",
    },
    "ead_adduct_c2h2o": {
        "type": "ead_adduct", "color": "#E54A5B", "default_intensity": 17.5,
        "is_key": True, "annotation_template": "[FA3HFA+C2H2O+Na]+", "offset": "c2h2o",
    },
    "ead_adduct_c3h4": {
        "type": "ead_adduct", "color": "#E54A5B", "default_intensity": 10.0,
        "is_key": True, "annotation_template": "[FA3HFA+C3H4+Na]+", "offset": "c3h4",
    },
    "ead_adduct_ch_radical": {
        "type": "ead_adduct", "color": "#E54A5B", "default_intensity": 7.5,
        "is_key": True, "annotation_template": "[FA3HFA+CH•+Na]+", "offset": "ch_radical",
    },
    "ead_sn2_diagnostic": {
        "type": "ead_sn2_diagnostic", "color": "#FF683A", "default_intensity": 22.5,
        "is_key": True, "annotation_template": "[FA3HFA+C2H2+Na]+", "offset": "c2h2",
    },
    "ead_sn2": {
        "type": "ead_sn2", "color": "#FF683A", "default_intensity": 10.0,
        "is_key": False, "annotation_template": "[FA1+C2H2+Na]+", "offset": "c2h2",
    },
    "ead_diagnostic_ester_high": {
        "type": "ead_diagnostic_ester", "color": "#FFC000", "default_intensity": 13.0,
        "is_key": False, "annotation_template": "[M-FA3-C{x}H{h}+O+Na]+",
    },
    "ead_diagnostic_ester_low": {
        "type": "ead_diagnostic_ester", "color": "#FFC000", "default_intensity": 10.0,
        "is_key": False, "annotation_template": "[M-FA3-C{z}H{w}•+Na]+",
    },
}

FRAGMENT_TYPE_METADATA: Dict[str, Dict[str, Any]] = {}
for fragment_def in FRAGMENT_DEFINITIONS.values():
    FRAGMENT_TYPE_METADATA.setdefault(fragment_def["type"], fragment_def)

NEUTRAL_FRAGMENT_TYPES = {
    meta["type"] for meta in FRAGMENT_DEFINITIONS.values()
    if meta.get("color") == "#0BB4F2"
}

FRAGMENT_LEVEL_DISPLAY_MAP = {
    "neutral_loss": "L1",
    "neutral_loss_branch": "L1",
    "neutral_loss_combined": "L1",
    "neutral_loss_fahfa": "L1",
    "ead_sn2": "L1",
    "fahfa_precursor": "L2",
    "fa3_protonated": "L2",
    "ead_diagnostic": "L2",
    "hfa_dehydrated_adduct_c2h2o": "L3",
    "hfa_dehydrated_adduct_c3h4": "L3",
    "hfa_dehydrated_sn2_c2h2": "L3",
    "ead_adduct": "L3",
    "ead_adduct_c2h2o": "L3",
    "ead_adduct_c3h4": "L3",
    "ead_adduct_ch_radical": "L3",
    "ead_sn2_diagnostic": "L3",
    "ead_diagnostic_ester": "L4",
    "ead_diagnostic_ester_high": "L4",
    "ead_diagnostic_ester_low": "L4",
}

SPECTRUM_LEVEL_COLORS: Dict[str, str] = {
    "L1": "#0B78A8",
    "L2": "#147A35",
    "L3": "#C94820",
    "L4": "#A16A00",
    "no_level": "#334155",
}

HDCS_FRAGMENT_MAP: Dict[str, Dict[str, Any]] = {
    "neutral_loss": {"level": 1, "group": None},
    "neutral_loss_branch": {"level": 1, "group": None},
    "neutral_loss_combined": {"level": 1, "group": None},
    "neutral_loss_fahfa": {"level": 1, "group": None},
    "ead_sn2": {"level": 1, "group": None},
    "fahfa_precursor": {"level": 2, "group": None},
    "fa3_protonated": {"level": 2, "group": None},
    "ead_diagnostic": {"level": 2, "group": None},
    "hfa_dehydrated_adduct_c2h2o": {"level": 3, "group": "sn-1/3"},
    "hfa_dehydrated_adduct_c3h4": {"level": 3, "group": "sn-1/3"},
    "hfa_dehydrated_sn2_c2h2": {"level": 3, "group": "sn-2"},
    "ead_adduct_c2h2o": {"level": 3, "group": "sn-1/3"},
    "ead_adduct_c3h4": {"level": 3, "group": "sn-1/3"},
    "ead_adduct_ch_radical": {"level": 3, "group": "sn-1/3"},
    "ead_sn2_diagnostic": {"level": 3, "group": "sn-2"},
    "ead_diagnostic_ester_high": {"level": 4, "group_from_ester_position": True},
    "ead_diagnostic_ester_low": {"level": 4, "group_from_ester_position": True},
}


@dataclass
class HDCSConfig:
    """HDCS-MD parameters inherited from the original scoring program."""

    ppm_tolerance: float = 20.0
    l1_threshold: float = 150.0
    l2_threshold: float = 150.0
    l3_threshold: float = 150.0
    l4_threshold: float = 150.0
    shared_ion_tolerance_ppm: float = 20.0
    level_weights: Dict[int, float] = field(
        default_factory=lambda: {1: 1.0, 2: 3.0, 3: 5.0, 4: 0.0}
    )
    nep_lambda: float = 0.5
    l4_bonus_max: float = 0.10
    wdic_pass_threshold: float = 0.70
    sn_mixture_ratio: float = 0.15
    sn_mixture_min_intensity: float = 1500.0
    ester_mixture_ratio: float = 0.10
    ester_mixture_min_intensity: float = 200.0
    nnls_min_final_score: float = TARGET_HDCS_SCORE  # Exact target, not a lower bound.
    nnls_enabled: bool = True
    mixture_detection_enabled: bool = True


#############################################################################
#  CHEMICAL DATABASE
#############################################################################
# Basic fatty acid building blocks (even 14–20 carbons, 0–3 unsaturation)
FA_INFO: Dict[str, Tuple[int, int]] = {}
for c in range(14, 21, 2):
    for db in range(4):
        FA_INFO[f"{c}:{db}"] = (c, db)

# Common hydroxyl (ester bond) positions for hydroxy fatty acids
OH_POSITIONS_MAP: Dict[int, List[int]] = {
    14: [5, 7], 16: [5, 7, 9, 12],
    18: [5, 7, 9, 10, 12, 13], 20: [5, 9, 12, 14],
}

def get_oh_positions(c_num: int) -> List[int]:
    """Return common OH positions for a given HFA carbon number."""
    return OH_POSITIONS_MAP.get(c_num, [int(c_num / 2)])

# Fatty-acid abbreviations
FA_NAMES: Dict[Tuple[int, int], str] = {
    (4, 0): 'BA', (6, 0): 'HA', (8, 0): 'OcA', (10, 0): 'DA',
    (12, 0): 'DDA', (14, 0): 'MA', (16, 0): 'PA', (18, 0): 'SA',
    (20, 0): 'AdA', (22, 0): 'BeA', (24, 0): 'LiA', (26, 0): 'CeA',
    (14, 1): 'MOA', (16, 1): 'POA',
    (18, 1): 'OA', (20, 1): 'EA', (22, 1): 'ErA',
    (24, 1): 'NA', (18, 2): 'LA', (18, 3): 'ALA', (18, 4): 'SDA',
    (20, 2): 'EDA', (20, 3): 'DGLA', (20, 4): 'AA', (20, 5): 'EPA',
    (22, 2): 'DcDA', (22, 3): 'DTA', (22, 4): 'AdrA', (22, 5): 'DPA',
    (22, 6): 'DHA', (24, 2): 'TDA', (24, 4): 'TeA', (24, 5): 'TPA',
    (24, 6): 'THA',
}

HFA_ABBRS: Dict[Tuple[int, int], str] = {
    (14, 0): 'HMA', (16, 0): 'HPA', (18, 0): 'HSA',
    (14, 1): 'HMOA', (16, 1): 'HPOA', (18, 1): 'HOA',
    (18, 2): 'HLA', (18, 3): 'HALA', (18, 4): 'HSDA',
    (20, 0): 'HEA', (20, 1): 'HEA', (20, 2): 'HEDA',
    (20, 3): 'HDGLA', (20, 4): 'HAA', (20, 5): 'HEPA',
    (22, 0): 'HBeA', (22, 4): 'HAdrA',
    (22, 5): 'HDPA', (22, 6): 'HDHA',
    (24, 1): 'HNA',
}

VALENCES = {"C": 4, "N": 3, "O": 2, "S": 2, "P": 3, "F": 1, "Cl": 1, "Br": 1, "I": 1}

TARGET_LIBRARY = [
    {"name": "TG-FAHFA 56:0", "c": 56, "db": 0, "mz": 971.8249}, {"name": "TG-FAHFA 56:1", "c": 56, "db": 1, "mz": 969.8093}, {"name": "TG-FAHFA 56:2", "c": 56, "db": 2, "mz": 967.7936}, {"name": "TG-FAHFA 56:3", "c": 56, "db": 3, "mz": 965.778}, {"name": "TG-FAHFA 56:4", "c": 56, "db": 4, "mz": 963.7623}, {"name": "TG-FAHFA 56:5", "c": 56, "db": 5, "mz": 961.7467}, {"name": "TG-FAHFA 56:6", "c": 56, "db": 6, "mz": 959.731}, {"name": "TG-FAHFA 56:7", "c": 56, "db": 7, "mz": 957.7154}, {"name": "TG-FAHFA 56:8", "c": 56, "db": 8, "mz": 955.6997},
    {"name": "TG-FAHFA 58:0", "c": 58, "db": 0, "mz": 999.8562}, {"name": "TG-FAHFA 58:1", "c": 58, "db": 1, "mz": 997.8406}, {"name": "TG-FAHFA 58:2", "c": 58, "db": 2, "mz": 995.8249}, {"name": "TG-FAHFA 58:3", "c": 58, "db": 3, "mz": 993.8093}, {"name": "TG-FAHFA 58:4", "c": 58, "db": 4, "mz": 991.7936}, {"name": "TG-FAHFA 58:5", "c": 58, "db": 5, "mz": 989.778}, {"name": "TG-FAHFA 58:6", "c": 58, "db": 6, "mz": 987.7623}, {"name": "TG-FAHFA 58:7", "c": 58, "db": 7, "mz": 985.7467}, {"name": "TG-FAHFA 58:8", "c": 58, "db": 8, "mz": 983.731},
    {"name": "TG-FAHFA 60:0", "c": 60, "db": 0, "mz": 1027.8875}, {"name": "TG-FAHFA 60:1", "c": 60, "db": 1, "mz": 1025.8719}, {"name": "TG-FAHFA 60:2", "c": 60, "db": 2, "mz": 1023.8562}, {"name": "TG-FAHFA 60:3", "c": 60, "db": 3, "mz": 1021.8406}, {"name": "TG-FAHFA 60:4", "c": 60, "db": 4, "mz": 1019.8249}, {"name": "TG-FAHFA 60:5", "c": 60, "db": 5, "mz": 1017.8093}, {"name": "TG-FAHFA 60:6", "c": 60, "db": 6, "mz": 1015.7936}, {"name": "TG-FAHFA 60:7", "c": 60, "db": 7, "mz": 1013.778}, {"name": "TG-FAHFA 60:8", "c": 60, "db": 8, "mz": 1011.7623},
    {"name": "TG-FAHFA 62:0", "c": 62, "db": 0, "mz": 1055.9188}, {"name": "TG-FAHFA 62:1", "c": 62, "db": 1, "mz": 1053.9032}, {"name": "TG-FAHFA 62:2", "c": 62, "db": 2, "mz": 1051.8875}, {"name": "TG-FAHFA 62:3", "c": 62, "db": 3, "mz": 1049.8719}, {"name": "TG-FAHFA 62:4", "c": 62, "db": 4, "mz": 1047.8562}, {"name": "TG-FAHFA 62:5", "c": 62, "db": 5, "mz": 1045.8406}, {"name": "TG-FAHFA 62:6", "c": 62, "db": 6, "mz": 1043.8249}, {"name": "TG-FAHFA 62:7", "c": 62, "db": 7, "mz": 1041.8093}, {"name": "TG-FAHFA 62:8", "c": 62, "db": 8, "mz": 1039.7936},
    {"name": "TG-FAHFA 64:0", "c": 64, "db": 0, "mz": 1083.9501}, {"name": "TG-FAHFA 64:1", "c": 64, "db": 1, "mz": 1081.9345}, {"name": "TG-FAHFA 64:2", "c": 64, "db": 2, "mz": 1079.9188}, {"name": "TG-FAHFA 64:3", "c": 64, "db": 3, "mz": 1077.9032}, {"name": "TG-FAHFA 64:4", "c": 64, "db": 4, "mz": 1075.8875}, {"name": "TG-FAHFA 64:5", "c": 64, "db": 5, "mz": 1073.8719}, {"name": "TG-FAHFA 64:6", "c": 64, "db": 6, "mz": 1071.8562}, {"name": "TG-FAHFA 64:7", "c": 64, "db": 7, "mz": 1069.8406}, {"name": "TG-FAHFA 64:8", "c": 64, "db": 8, "mz": 1067.8249},
    {"name": "TG-FAHFA 66:0", "c": 66, "db": 0, "mz": 1111.9814}, {"name": "TG-FAHFA 66:1", "c": 66, "db": 1, "mz": 1109.9658}, {"name": "TG-FAHFA 66:2", "c": 66, "db": 2, "mz": 1107.9501}, {"name": "TG-FAHFA 66:3", "c": 66, "db": 3, "mz": 1105.9345}, {"name": "TG-FAHFA 66:4", "c": 66, "db": 4, "mz": 1103.9188}, {"name": "TG-FAHFA 66:5", "c": 66, "db": 5, "mz": 1101.9032}, {"name": "TG-FAHFA 66:6", "c": 66, "db": 6, "mz": 1099.8875}, {"name": "TG-FAHFA 66:7", "c": 66, "db": 7, "mz": 1097.8719}, {"name": "TG-FAHFA 66:8", "c": 66, "db": 8, "mz": 1095.8562},
    {"name": "TG-FAHFA 68:0", "c": 68, "db": 0, "mz": 1140.0127}, {"name": "TG-FAHFA 68:1", "c": 68, "db": 1, "mz": 1137.9971}, {"name": "TG-FAHFA 68:2", "c": 68, "db": 2, "mz": 1135.9814}, {"name": "TG-FAHFA 68:3", "c": 68, "db": 3, "mz": 1133.9658}, {"name": "TG-FAHFA 68:4", "c": 68, "db": 4, "mz": 1131.9501}, {"name": "TG-FAHFA 68:5", "c": 68, "db": 5, "mz": 1129.9345}, {"name": "TG-FAHFA 68:6", "c": 68, "db": 6, "mz": 1127.9188}, {"name": "TG-FAHFA 68:7", "c": 68, "db": 7, "mz": 1125.9032}, {"name": "TG-FAHFA 68:8", "c": 68, "db": 8, "mz": 1123.8875},
    {"name": "TG-FAHFA 70:0", "c": 70, "db": 0, "mz": 1168.044}, {"name": "TG-FAHFA 70:1", "c": 70, "db": 1, "mz": 1166.0284}, {"name": "TG-FAHFA 70:2", "c": 70, "db": 2, "mz": 1164.0127}, {"name": "TG-FAHFA 70:3", "c": 70, "db": 3, "mz": 1161.9971}, {"name": "TG-FAHFA 70:4", "c": 70, "db": 4, "mz": 1159.9814}, {"name": "TG-FAHFA 70:5", "c": 70, "db": 5, "mz": 1157.9658}, {"name": "TG-FAHFA 70:6", "c": 70, "db": 6, "mz": 1155.9501}, {"name": "TG-FAHFA 70:7", "c": 70, "db": 7, "mz": 1153.9345}, {"name": "TG-FAHFA 70:8", "c": 70, "db": 8, "mz": 1151.9188},
    {"name": "TG-FAHFA 72:0", "c": 72, "db": 0, "mz": 1196.0753}, {"name": "TG-FAHFA 72:1", "c": 72, "db": 1, "mz": 1194.0597}, {"name": "TG-FAHFA 72:2", "c": 72, "db": 2, "mz": 1192.044}, {"name": "TG-FAHFA 72:3", "c": 72, "db": 3, "mz": 1190.0284}, {"name": "TG-FAHFA 72:4", "c": 72, "db": 4, "mz": 1188.0127}, {"name": "TG-FAHFA 72:5", "c": 72, "db": 5, "mz": 1185.9971}, {"name": "TG-FAHFA 72:6", "c": 72, "db": 6, "mz": 1183.9814}, {"name": "TG-FAHFA 72:7", "c": 72, "db": 7, "mz": 1181.9658}, {"name": "TG-FAHFA 72:8", "c": 72, "db": 8, "mz": 1179.9501},
    {"name": "TG-FAHFA 74:0", "c": 74, "db": 0, "mz": 1224.1066}, {"name": "TG-FAHFA 74:1", "c": 74, "db": 1, "mz": 1222.091}, {"name": "TG-FAHFA 74:2", "c": 74, "db": 2, "mz": 1220.0753}, {"name": "TG-FAHFA 74:3", "c": 74, "db": 3, "mz": 1218.0597}, {"name": "TG-FAHFA 74:4", "c": 74, "db": 4, "mz": 1216.044}, {"name": "TG-FAHFA 74:5", "c": 74, "db": 5, "mz": 1214.0284}, {"name": "TG-FAHFA 74:6", "c": 74, "db": 6, "mz": 1212.0127}, {"name": "TG-FAHFA 74:7", "c": 74, "db": 7, "mz": 1209.9971}, {"name": "TG-FAHFA 74:8", "c": 74, "db": 8, "mz": 1207.9814},
    {"name": "TG-FAHFA 76:0", "c": 76, "db": 0, "mz": 1252.1379}, {"name": "TG-FAHFA 76:1", "c": 76, "db": 1, "mz": 1250.1223}, {"name": "TG-FAHFA 76:2", "c": 76, "db": 2, "mz": 1248.1066}, {"name": "TG-FAHFA 76:3", "c": 76, "db": 3, "mz": 1246.091}, {"name": "TG-FAHFA 76:4", "c": 76, "db": 4, "mz": 1244.0753}, {"name": "TG-FAHFA 76:5", "c": 76, "db": 5, "mz": 1242.0597}, {"name": "TG-FAHFA 76:6", "c": 76, "db": 6, "mz": 1240.044}, {"name": "TG-FAHFA 76:7", "c": 76, "db": 7, "mz": 1238.0284}, {"name": "TG-FAHFA 76:8", "c": 76, "db": 8, "mz": 1236.0127},
    {"name": "TG-FAHFA 78:0", "c": 78, "db": 0, "mz": 1280.1692}, {"name": "TG-FAHFA 78:1", "c": 78, "db": 1, "mz": 1278.1536}, {"name": "TG-FAHFA 78:2", "c": 78, "db": 2, "mz": 1276.1379}, {"name": "TG-FAHFA 78:3", "c": 78, "db": 3, "mz": 1274.1223}, {"name": "TG-FAHFA 78:4", "c": 78, "db": 4, "mz": 1272.1066}, {"name": "TG-FAHFA 78:5", "c": 78, "db": 5, "mz": 1270.091}, {"name": "TG-FAHFA 78:6", "c": 78, "db": 6, "mz": 1268.0753}, {"name": "TG-FAHFA 78:7", "c": 78, "db": 7, "mz": 1266.0597}, {"name": "TG-FAHFA 78:8", "c": 78, "db": 8, "mz": 1264.044},
    {"name": "TG-FAHFA 80:0", "c": 80, "db": 0, "mz": 1308.2005}, {"name": "TG-FAHFA 80:1", "c": 80, "db": 1, "mz": 1306.1849}, {"name": "TG-FAHFA 80:2", "c": 80, "db": 2, "mz": 1304.1692}, {"name": "TG-FAHFA 80:3", "c": 80, "db": 3, "mz": 1302.1536}, {"name": "TG-FAHFA 80:4", "c": 80, "db": 4, "mz": 1300.1379}, {"name": "TG-FAHFA 80:5", "c": 80, "db": 5, "mz": 1298.1223}, {"name": "TG-FAHFA 80:6", "c": 80, "db": 6, "mz": 1296.1066}, {"name": "TG-FAHFA 80:7", "c": 80, "db": 7, "mz": 1294.091}, {"name": "TG-FAHFA 80:8", "c": 80, "db": 8, "mz": 1292.0753}
]

#############################################################################
#  UTILITY FUNCTIONS
#############################################################################
def format_fa_string(fa_str):
    if not fa_str or not isinstance(fa_str, str): return fa_str
    fa_str = fa_str.strip()
    if ':' in fa_str:
        parts = fa_str.split(':')
        if len(parts) == 2:
            try: return f"{int(float(parts[0]))}:{int(float(parts[1]))}"
            except ValueError: pass
    return fa_str

def format_common_name_for_display(common_name: str) -> str:
    """Convert internal TG names to the compact GUI display format."""
    if not isinstance(common_name, str):
        return common_name
    text = common_name.strip()
    if text.startswith("TG(") and text.endswith(")"):
        return f"TG {text[3:-1]}"
    if text.startswith("TG-FAHFA(") and text.endswith(")"):
        return f"TG {text[len('TG-FAHFA('):-1]}"
    return text

def format_subscript(text: str) -> str:
    """Convert digits and formula x placeholders to Unicode subscripts."""
    formatted = str(text).translate(SUBSCRIPT_TRANSLATION)
    return formatted.replace("ₓ-₁", "ₓ₋₁")

def render_fragment_annotation(fragment_key: str, **values) -> str:
    meta = FRAGMENT_DEFINITIONS[fragment_key]
    annotation = meta["annotation_template"].format(**values)
    if meta.get("format_subscripts", True):
        return format_subscript(annotation)
    return annotation

def fragment_intensity(fragment_key: str, sn: Optional[str] = None) -> float:
    meta = FRAGMENT_DEFINITIONS[fragment_key]
    intensity_by_sn = meta.get("intensity_by_sn", {})
    return float(intensity_by_sn.get(sn, meta["default_intensity"]))

def fragment_color(peak_type: str, is_key: bool = False, fahfa_sn: Optional[str] = None) -> str:
    type_names = [part.strip() for part in str(peak_type).split('/') if part.strip()]
    if "ead_diagnostic_ester" in type_names:
        return FRAGMENT_TYPE_METADATA["ead_diagnostic_ester"]["color"]
    for type_name in type_names:
        meta = FRAGMENT_TYPE_METADATA.get(type_name)
        if meta:
            return meta.get("color_by_fahfa_sn", {}).get(fahfa_sn, meta["color"])
    return "#636E72"

def fragment_table_tag(peak_type: str, is_key: bool = False, fahfa_sn: Optional[str] = None) -> Tuple[str, ...]:
    type_names = [part.strip() for part in str(peak_type).split('/') if part.strip()]
    if "ead_diagnostic_ester" in type_names:
        return ("ester",)
    is_neutral = any(type_name in NEUTRAL_FRAGMENT_TYPES for type_name in type_names)
    if is_key and is_neutral:
        return ("neutral_key",)
    if is_key:
        return ("key_sn2",) if fahfa_sn == "sn2" else ("key_sn1",)
    if is_neutral:
        return ("neutral_tag",)
    if "ead_sn2" in type_names:
        return ("ead_sn2",)
    return ("normal",)

def fragment_level_display(peak_type: str) -> str:
    """
    Convert internal fragment type names to GUI evidence levels.

    Internal fragment type values must remain unchanged because they are used
    by coloring, scoring, peak matching and plotting logic.
    """
    if not peak_type:
        return ""

    converted = []
    for part in str(peak_type).split('/'):
        type_name = part.strip()
        if not type_name:
            continue
        display_name = FRAGMENT_LEVEL_DISPLAY_MAP.get(type_name, type_name)
        if display_name not in converted:
            converted.append(display_name)

    level_order = {"L1": 0, "L2": 1, "L3": 2, "L4": 3}
    converted.sort(key=lambda name: level_order.get(name, len(level_order)))
    return " / ".join(converted)


def fragment_levels_display(levels: Sequence[int], fallback_type: str = "") -> str:
    unique_levels = sorted({int(level) for level in levels if int(level) in (1, 2, 3, 4)})
    if unique_levels:
        return " / ".join(f"L{level}" for level in unique_levels)
    return fragment_level_display(fallback_type)


def spectrum_level_tag(levels: Sequence[int]) -> Tuple[str, ...]:
    """Choose one Spectrum Prediction row color from the highest HDCS level."""
    valid_levels: Set[int] = set()
    for level in levels:
        try:
            level_number = int(level)
        except (TypeError, ValueError):
            continue
        if level_number in (1, 2, 3, 4):
            valid_levels.add(level_number)
    return (f"L{max(valid_levels)}",) if valid_levels else ("no_level",)


def spectrum_level_color(levels: Sequence[int]) -> str:
    tag = spectrum_level_tag(levels)[0]
    return SPECTRUM_LEVEL_COLORS.get(tag, SPECTRUM_LEVEL_COLORS["no_level"])


def normalize_fahfa_sn_group(value: Any) -> str:
    """Normalize all known FAHFA sn spellings without silently guessing."""
    text = str(value or "").strip().lower().replace("_", "-").replace(" ", "")
    if text in {"sn-1/3", "sn1/3", "sn1", "sn3", "sn-1", "sn-3", "sn-1/3position"}:
        return "sn-1"
    if text in {"sn2", "sn-2", "sn-2position"}:
        return "sn-2"
    return "unknown"


def safe_candidate_value(row_data: Any, *names: str, default: Any = None) -> Any:
    """Read a candidate field from Series/dict while accepting common name variants."""
    if hasattr(row_data, "to_dict"):
        source = row_data.to_dict()
    elif isinstance(row_data, dict):
        source = row_data
    else:
        source = vars(row_data) if hasattr(row_data, "__dict__") else {}
    normalized = {re.sub(r"[^a-z0-9]", "", str(key).lower()): value for key, value in source.items()}
    for name in names:
        if name in source and source[name] is not None:
            return source[name]
        key = re.sub(r"[^a-z0-9]", "", name.lower())
        if key in normalized and normalized[key] is not None:
            return normalized[key]
    return default


def _parse_ester_position(row_data: Any) -> Optional[int]:
    direct = safe_candidate_value(
        row_data, "ester_pos", "Ester_Position", "OH_Pos", "FAHFA_Ester_Position"
    )
    if direct not in (None, ""):
        try:
            return int(float(direct))
        except (TypeError, ValueError):
            pass
    for field_name in ("FAHFA", "TG_FAHFA_Structure", "FAHFA_Display", "name"):
        text = str(safe_candidate_value(row_data, field_name, default="") or "")
        match = re.search(r"\((\d+)\s*-\s*O\s*-", text, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _extract_fa3_hfa_names(row_data: Any, fahfa_name: str) -> Tuple[str, str]:
    """Read explicit FA3/HFA fields, with a FAHFA-only compatibility fallback."""
    fa3_name = str(
        safe_candidate_value(row_data, "FA3", "Acyl_FA", default="") or ""
    ).strip('="')
    hfa_name = str(
        safe_candidate_value(row_data, "HFA", "Hydroxy_FA", default="") or ""
    ).strip('="')
    if fa3_name and hfa_name:
        return format_fa_string(fa3_name), format_fa_string(hfa_name)
    match = re.search(
        r"(?P<fa3>\d+\s*:\s*\d+)\s*\(\s*\d+\s*-\s*O\s*-\s*(?P<hfa>\d+\s*:\s*\d+)",
        str(fahfa_name or ""),
        re.IGNORECASE,
    )
    if match:
        fa3_name = fa3_name or match.group("fa3")
        hfa_name = hfa_name or match.group("hfa")
    return (
        format_fa_string(fa3_name) if fa3_name else "",
        format_fa_string(hfa_name) if hfa_name else "",
    )


def merge_candidate_diagnostic_ions(
    diagnostic_ions: Sequence[Dict[str, Any]], tolerance_ppm: float
) -> List[Dict[str, Any]]:
    """Merge indistinguishable theoretical ions within one candidate.

    The returned entries are observable scoring dimensions.  Their effective
    evidence level is the highest level among all interpretations at that m/z,
    while the original interpretations are retained as member metadata.
    """
    tolerance = float(tolerance_ppm)
    if tolerance <= 0:
        raise ValueError("Candidate diagnostic-ion merge tolerance must be positive")

    def ordered_unique(values: Iterable[Any]) -> List[Any]:
        output: List[Any] = []
        for value in values:
            if value in (None, "") or value in output:
                continue
            output.append(value)
        return output

    ions = [deepcopy(dict(ion)) for ion in diagnostic_ions]
    ions.sort(key=lambda ion: float(ion["mz"]))
    clusters: List[List[Dict[str, Any]]] = []
    centroids: List[float] = []
    for ion in ions:
        mz = float(ion["mz"])
        if clusters and abs(mz - centroids[-1]) / centroids[-1] * 1e6 <= tolerance:
            clusters[-1].append(ion)
            centroids[-1] = float(
                np.mean([float(member["mz"]) for member in clusters[-1]])
            )
        else:
            clusters.append([ion])
            centroids.append(mz)

    merged_ions: List[Dict[str, Any]] = []
    for members, centroid in zip(clusters, centroids):
        levels = sorted({int(member["level"]) for member in members})
        groups: List[str] = []
        for member in members:
            member_groups = member.get("groups") or []
            if isinstance(member_groups, str):
                member_groups = [member_groups]
            groups.extend(str(group) for group in member_groups if group)
            if member.get("group"):
                groups.append(str(member["group"]))
        groups = ordered_unique(groups)
        fragment_keys = ordered_unique(str(member.get("fragment_key") or "") for member in members)
        types = ordered_unique(str(member.get("type") or "") for member in members)
        annotations = ordered_unique(str(member.get("annotation") or "") for member in members)
        labels = ordered_unique(str(member.get("label") or "") for member in members)
        formulas = ordered_unique(str(member.get("formula") or "") for member in members)
        formula_dicts = ordered_unique(
            deepcopy(member.get("formula_dict", {}))
            for member in members
            if member.get("formula_dict")
        )
        source_roles = ordered_unique(str(member.get("source_role") or "") for member in members)
        hfa_names = ordered_unique(str(member.get("hfa_name") or "") for member in members)
        raw_peak_indices = ordered_unique(member.get("raw_peak_index") for member in members)
        member_mzs = [float(member["mz"]) for member in members]

        merged = deepcopy(members[0])
        merged.update(
            {
                "mz": float(centroid),
                "theoretical_mz": float(centroid),
                "level": max(levels),
                "levels": levels,
                "group": groups[0] if len(groups) == 1 else None,
                "groups": groups,
                "fragment_key": " / ".join(fragment_keys),
                "fragment_keys": fragment_keys,
                "type": " / ".join(types),
                "types": types,
                "annotation": " / ".join(annotations),
                "annotations": annotations,
                "label": " / ".join(labels or annotations),
                "is_key_ion": any(bool(member.get("is_key_ion", False)) for member in members),
                "exact_mz": float(centroid),
                "formula": " / ".join(formulas),
                "formulas": formulas,
                "formula_dict": deepcopy(formula_dicts[0]) if len(formula_dicts) == 1 else {},
                "formula_dicts": formula_dicts,
                "source_role": " / ".join(source_roles),
                "source_roles": source_roles,
                "hfa_name": " / ".join(hfa_names),
                "hfa_names": hfa_names,
                "raw_peak_index": raw_peak_indices[0] if raw_peak_indices else None,
                "raw_peak_indices": raw_peak_indices,
                "member_mzs": member_mzs,
                "member_ions": deepcopy(members),
                "merged_member_count": len(members),
                "is_merged": len(members) > 1,
            }
        )
        merged_ions.append(merged)
    return merged_ions


def remove_ester_position_from_common_name(common_name: str) -> str:
    """Remove only a FAHFA abbreviation's leading ester-position number."""
    text = str(common_name or "")
    return re.sub(r"(^|[/\s(])\d+-(?=[A-Za-z])", r"\1", text)


def remove_ester_position_from_lipid_name(lipid_name: str) -> str:
    """Change a FAHFA ``(9-O-...)`` label to ``(O-...)`` without touching FA numbers."""
    return re.sub(r"\(\s*\d+\s*-\s*O\s*-", "(O-", str(lipid_name or ""), flags=re.I)


def fa_canonical_sort_key(fa_name: str) -> Tuple[int, int, str]:
    """Return the stable structural sort key used for sn-2 outer FA ordering."""
    normalized = format_fa_string(fa_name)
    if normalized in FA_INFO:
        carbon, double_bonds = FA_INFO[normalized]
        return carbon, double_bonds, normalized
    return 10**9, 10**9, str(normalized)


def canonicalize_sn2_outer_fas(fa_a: str, fa_b: str) -> Tuple[str, str]:
    """Place the smaller outer ordinary FA at sn-1 and the other at sn-3."""
    normalized_a = format_fa_string(fa_a)
    normalized_b = format_fa_string(fa_b)
    return tuple(sorted((normalized_a, normalized_b), key=fa_canonical_sort_key))


def build_position_family_key(
    candidate: Dict[str, Any], candidate_row: Optional[Any] = None
) -> Tuple[Any, ...]:
    """Identify candidates differing only in the FAHFA ester position.

    TG sn order and every chain identity remain part of the key.  Only the
    ester-position number embedded in the FAHFA component is normalized.
    """
    row = candidate_row if candidate_row is not None else {}
    sn_chains = candidate.get("sn_chains") or tuple(
        str(safe_candidate_value(row, f"FA_Sn{index}", default="") or "").strip('="')
        for index in (1, 2, 3)
    )
    normalized_sn_chains = tuple(
        remove_ester_position_from_lipid_name(str(chain)) for chain in sn_chains
    )
    ordinary_fas = candidate.get("ordinary_fas") or (
        str(safe_candidate_value(row, "FA1", default="") or ""),
        str(safe_candidate_value(row, "FA2", default="") or ""),
    )
    formula = str(candidate.get("formula") or safe_candidate_value(row, "Formula", default="") or "")
    fahfa_name = remove_ester_position_from_lipid_name(
        str(candidate.get("FAHFA_name") or safe_candidate_value(row, "FAHFA", default="") or "")
    )
    sn_fahfa = str(candidate.get("sn_fahfa") or "")
    precursor_mz = round(float(candidate.get("precursor_mz") or 0.0), 5)
    if sn_fahfa == "sn-2" and len(normalized_sn_chains) >= 3:
        canonical_sn1, canonical_sn3 = canonicalize_sn2_outer_fas(
            normalized_sn_chains[0], normalized_sn_chains[2]
        )
        canonical_sn_chains = (
            canonical_sn1,
            normalized_sn_chains[1],
            canonical_sn3,
        )
        return (
            "sn-2",
            canonical_sn_chains,
            (canonical_sn1, canonical_sn3),
            fahfa_name,
            formula,
            precursor_mz,
        )

    structure = remove_ester_position_from_lipid_name(
        str(candidate.get("structure") or candidate.get("name") or "")
    )
    common_name = remove_ester_position_from_common_name(
        str(candidate.get("theoretical_common_name") or "")
    )
    return (
        sn_fahfa,
        normalized_sn_chains,
        tuple(str(value) for value in ordinary_fas),
        fahfa_name,
        structure,
        common_name,
        formula,
        precursor_mz,
    )


def convert_tg_candidate_to_hdcs_candidate(
    row_data: Any,
    raw_peaks: Sequence[Dict[str, Any]],
    generator: Any,
    candidate_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert one dynamically generated TG-FAHFA candidate into an HDCS object."""
    structure = str(
        safe_candidate_value(
            row_data, "TG_FAHFA_Structure", "Structure", "Candidate_Name", "name", default=""
        )
        or ""
    ).strip('="')
    key = str(
        candidate_key
        or safe_candidate_value(row_data, "candidate_key", "Candidate_Key", "key", default=structure)
        or structure
    )
    sn_raw = safe_candidate_value(
        row_data, "FAHFA_Position", "sn_fahfa", "FAHFA_sn", "sn", default=""
    )
    sn_group = normalize_fahfa_sn_group(sn_raw)
    if sn_group == "unknown":
        raise ValueError(f"Cannot normalize FAHFA sn group for candidate {key}: {sn_raw!r}")
    ester_pos = _parse_ester_position(row_data)
    if ester_pos is None:
        raise ValueError(f"Cannot parse FAHFA ester position for candidate {key}")

    diagnostic_ions: List[Dict[str, Any]] = []
    for peak_index, peak in enumerate(raw_peaks):
        fragment_key = str(peak.get("fragment_key") or "")
        mapping = HDCS_FRAGMENT_MAP.get(fragment_key)
        if not mapping:
            continue
        group = mapping.get("group")
        if mapping.get("group_from_ester_position"):
            group = f"C{ester_pos}"
        diagnostic_ions.append(
            {
                "fragment_key": fragment_key,
                "fragment_keys": list(peak.get("fragment_keys", [fragment_key])),
                "type": str(peak.get("type", "")),
                "annotation": str(peak.get("annotation", "")),
                "label": str(peak.get("annotation", "")),
                "level": int(mapping["level"]),
                "levels": [int(mapping["level"])],
                "mz": float(peak["mz"]),
                "exact_mz": float(peak.get("exact_mz", peak["mz"])),
                "group": group,
                "groups": [group] if group is not None else [],
                "is_key_ion": bool(peak.get("is_key_ion", False)),
                "formula": str(peak.get("formula", "")),
                "formula_dict": deepcopy(peak.get("formula_dict", {})),
                "source_role": str(peak.get("source_role", "")),
                "hfa_name": str(peak.get("hfa_name", "")),
                "hfa_abbr": str(peak.get("hfa_abbr", "")),
                "hfa_carbon": peak.get("hfa_carbon"),
                "hfa_double_bonds": peak.get("hfa_double_bonds"),
                "raw_peak_index": peak_index,
            }
        )
    if not diagnostic_ions:
        raise ValueError(f"Candidate {key} produced no HDCS diagnostic ions")

    precursor_mz = safe_candidate_value(
        row_data, "Parent_Theo", "Precursor_mz", "precursor_mz", "Parent_Exp", default=0.0
    )
    fahfa_name = str(
        safe_candidate_value(row_data, "FAHFA", "FAHFA_name", "FAHFA_Display", default="") or ""
    )
    fa3_name, hfa_name = _extract_fa3_hfa_names(row_data, fahfa_name)
    fa3_peak = next(
        (peak for peak in raw_peaks if peak.get("fragment_key") == "fa3_protonated"),
        None,
    )
    if fa3_peak is not None:
        fa3_protonated_theoretical_mz = float(
            fa3_peak.get("exact_mz", fa3_peak.get("mz", 0.0))
        )
    elif fa3_name in FA_INFO:
        fa3_protonated_theoretical_mz = calculate_fa3_protonated_mz(*FA_INFO[fa3_name])
    else:
        fa3_protonated_theoretical_mz = 0.0
    theoretical_common_name = format_common_name_for_display(
        str(getattr(generator, "last_common_name", "") or structure or key)
    )
    theoretical_lipid_name = structure or key
    if not theoretical_lipid_name.startswith("TG "):
        theoretical_lipid_name = f"TG {theoretical_lipid_name}"
    candidate = {
        "key": key,
        "name": structure or key,
        "structure": structure or key,
        "sn_fahfa": sn_group,
        "ester_pos": int(ester_pos),
        "FAHFA_name": fahfa_name,
        "fa3_name": fa3_name,
        "hfa_name": hfa_name,
        "fa3_protonated_theoretical_mz": fa3_protonated_theoretical_mz,
        "precursor_mz": float(precursor_mz or 0.0),
        "sn_chains": tuple(
            str(safe_candidate_value(row_data, f"FA_Sn{index}", default="") or "").strip('="')
            for index in (1, 2, 3)
        ),
        "ordinary_fas": (
            str(safe_candidate_value(row_data, "FA1", default="") or ""),
            str(safe_candidate_value(row_data, "FA2", default="") or ""),
        ),
        "formula": str(safe_candidate_value(row_data, "Formula", default="") or ""),
        "theoretical_common_name": theoretical_common_name,
        "theoretical_lipid_name": theoretical_lipid_name,
        "raw_diagnostic_ions": deepcopy(diagnostic_ions),
        "diagnostic_ions": deepcopy(diagnostic_ions),
    }
    candidate["position_family_key"] = build_position_family_key(candidate, row_data)
    return candidate


def diagnostic_signature(candidate: Dict[str, Any]) -> Tuple[Tuple[Any, ...], ...]:
    def observable_groups(ion: Dict[str, Any]) -> Tuple[str, ...]:
        groups = ion.get("groups") or ([ion["group"]] if ion.get("group") else [])
        return tuple(str(group) for group in groups if group)

    return tuple(
        sorted(
            (
                int(ion["level"]),
                round(float(ion["mz"]), 5),
                observable_groups(ion),
            )
            for ion in candidate.get("diagnostic_ions", [])
        )
    )


def assign_equivalence_groups(candidates: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    signature_to_group: Dict[Tuple[Tuple[Any, ...], ...], str] = {}
    result: Dict[str, str] = {}
    for candidate in candidates:
        signature = diagnostic_signature(candidate)
        if signature not in signature_to_group:
            signature_to_group[signature] = f"EG-{len(signature_to_group) + 1:03d}"
        group = signature_to_group[signature]
        candidate["diagnostic_signature"] = signature
        candidate["equivalent_group"] = group
        result[candidate["key"]] = group
    return result


@dataclass(frozen=True)
class HDCSIonMatch:
    found: bool
    exp_index: Optional[int] = None
    exp_mz: Optional[float] = None
    ppm_error: Optional[float] = None
    intensity: float = 0.0
    sn_value: Optional[float] = None


class HDCSSpectrumMatcher:
    """Mass-error/intensity matcher adapted from the original HDCS-MD scorer."""

    def __init__(self, spectrum: pd.DataFrame):
        if spectrum is None or spectrum.empty:
            raise ValueError("Experimental spectrum is empty")
        ordered = spectrum.sort_values("mz").reset_index(drop=True)
        self.mzs = ordered["mz"].astype(float).to_numpy()
        self.intensities = ordered["intensity"].fillna(0.0).astype(float).to_numpy()
        self.sn_values = (
            ordered["sn_value"].astype(float).to_numpy()
            if "sn_value" in ordered and ordered["sn_value"].notna().any()
            else np.full(len(ordered), np.nan)
        )
        self.max_intensity = float(np.max(self.intensities)) if len(self.intensities) else 0.0

    def options(self, target_mz: float, tol_ppm: float, min_intensity: float) -> List[HDCSIonMatch]:
        delta = target_mz * tol_ppm * 1e-6
        left = int(np.searchsorted(self.mzs, target_mz - delta, side="left"))
        right = int(np.searchsorted(self.mzs, target_mz + delta, side="right"))
        output: List[HDCSIonMatch] = []
        for index in range(left, right):
            intensity = float(self.intensities[index])
            sn_value = float(self.sn_values[index]) if math.isfinite(self.sn_values[index]) else None
            if intensity < min_intensity:
                continue
            exp_mz = float(self.mzs[index])
            ppm = abs(exp_mz - target_mz) / target_mz * 1e6
            output.append(HDCSIonMatch(True, index, exp_mz, ppm, intensity, sn_value))
        output.sort(
            key=lambda item: (
                float(item.ppm_error or 0.0) / max(tol_ppm, 1e-12)
                - 0.05 * item.intensity / max(self.max_intensity, 1e-12),
                -item.intensity,
            )
        )
        return output

    def detect(self, target_mz: float, tol_ppm: float, min_intensity: float) -> HDCSIonMatch:
        options = self.options(target_mz, tol_ppm, min_intensity)
        return options[0] if options else HDCSIonMatch(False)


def cluster_dynamic_diagnostic_ions(
    candidates: Sequence[Dict[str, Any]], tol_ppm: float
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for candidate in candidates:
        for ion_index, ion in enumerate(candidate["diagnostic_ions"]):
            items.append(
                {
                    "candidate_key": candidate["key"],
                    "ion_index": ion_index,
                    **ion,
                }
            )
    items.sort(key=lambda item: float(item["mz"]))
    clusters: List[Dict[str, Any]] = []
    for item in items:
        mz = float(item["mz"])
        candidate_already_present = bool(
            clusters
            and any(
                member["candidate_key"] == item["candidate_key"]
                for member in clusters[-1]["members"]
            )
        )
        if (
            clusters
            and not candidate_already_present
            and abs(mz - clusters[-1]["mz"]) / clusters[-1]["mz"] * 1e6 <= tol_ppm
        ):
            cluster = clusters[-1]
            cluster["members"].append(item)
            cluster["mz"] = float(np.mean([float(member["mz"]) for member in cluster["members"]]))
        else:
            clusters.append({"mz": mz, "members": [item]})
    for index, cluster in enumerate(clusters, start=1):
        keys = list(dict.fromkeys(member["candidate_key"] for member in cluster["members"]))
        groups: List[str] = []
        for member in cluster["members"]:
            member_groups = member.get("groups") or []
            if isinstance(member_groups, str):
                member_groups = [member_groups]
            groups.extend(str(group) for group in member_groups if group)
            if member.get("group"):
                groups.append(str(member["group"]))
        cluster.update(
            {
                "cluster_id": f"mz_cluster_{index:04d}",
                "candidate_keys": keys,
                "levels": sorted({int(member["level"]) for member in cluster["members"]}),
                "groups": list(dict.fromkeys(groups)),
                "is_shared": len(keys) > 1,
                "is_unique": len(keys) == 1,
            }
        )
    return clusters


def _threshold_for_level(config: HDCSConfig, level: int) -> float:
    return {
        1: config.l1_threshold,
        2: config.l2_threshold,
        3: config.l3_threshold,
        4: config.l4_threshold,
    }.get(int(level), config.l1_threshold)


def _threshold_for_diagnostic_ion(config: HDCSConfig, ion: Dict[str, Any]) -> float:
    """Return the strictest raw-intensity threshold for one observable ion."""
    member_ions = ion.get("member_ions")
    if isinstance(member_ions, Sequence) and not isinstance(member_ions, (str, bytes)):
        members = [member for member in member_ions if isinstance(member, dict)]
    else:
        members = []
    if not members:
        members = [ion]

    thresholds: List[float] = []
    for member in members:
        levels = member.get("levels") or [member.get("level", 1)]
        if isinstance(levels, (str, bytes)):
            levels = [levels]
        member_thresholds = []
        for level in levels:
            try:
                member_thresholds.append(_threshold_for_level(config, int(level)))
            except (TypeError, ValueError):
                continue
        thresholds.append(max(member_thresholds or [_threshold_for_level(config, 1)]))
    return max(thresholds or [_threshold_for_level(config, int(ion.get("level", 1)))])


def scan_dynamic_diagnostic_ions(
    matcher: HDCSSpectrumMatcher,
    candidate: Dict[str, Any],
    config: HDCSConfig,
    cluster_registry: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    cluster_lookup: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for cluster in cluster_registry:
        for member in cluster["members"]:
            cluster_lookup[(member["candidate_key"], int(member["ion_index"]))] = cluster

    proposals: List[Tuple[float, int, HDCSIonMatch]] = []
    for ion_index, ion in enumerate(candidate["diagnostic_ions"]):
        threshold = _threshold_for_diagnostic_ion(config, ion)
        for match in matcher.options(float(ion["mz"]), config.ppm_tolerance, threshold):
            quality = float(match.ppm_error or 0.0) / config.ppm_tolerance
            quality -= 0.05 * match.intensity / max(matcher.max_intensity, 1e-12)
            proposals.append((quality, ion_index, match))
    proposals.sort(key=lambda item: (item[0], -item[2].intensity))
    assigned_ions: Dict[int, HDCSIonMatch] = {}
    used_exp: Set[int] = set()
    for _quality, ion_index, match in proposals:
        if ion_index in assigned_ions or match.exp_index in used_exp:
            continue
        assigned_ions[ion_index] = match
        if match.exp_index is not None:
            used_exp.add(match.exp_index)

    detected: Dict[str, Dict[str, Any]] = {}
    for ion_index, ion in enumerate(candidate["diagnostic_ions"]):
        match = assigned_ions.get(ion_index, HDCSIonMatch(False))
        cluster = cluster_lookup.get((candidate["key"], ion_index), {})
        detected[f"merged|L{ion['level']}|{ion_index}|{float(ion['mz']):.6f}"] = {
            "fragment_key": ion["fragment_key"],
            "fragment_keys": list(ion.get("fragment_keys", [ion["fragment_key"]])),
            "type": ion.get("type", ""),
            "types": list(ion.get("types", [ion.get("type", "")])),
            "annotation": ion.get("annotation", ""),
            "annotations": list(ion.get("annotations", [ion.get("annotation", "")])),
            "level": int(ion["level"]),
            "levels": list(ion.get("levels", [int(ion["level"])])),
            "group": ion.get("group"),
            "groups": list(
                ion.get("groups")
                or ([ion["group"]] if ion.get("group") else [])
            ),
            "theo_mz": float(ion["mz"]),
            "theoretical_mz": float(ion["mz"]),
            "exact_mz": float(ion.get("exact_mz", ion["mz"])),
            "is_key_ion": bool(ion.get("is_key_ion", False)),
            "formula": str(ion.get("formula", "")),
            "formula_dict": deepcopy(ion.get("formula_dict", {})),
            "formulas": list(ion.get("formulas", [ion.get("formula", "")])),
            "formula_dicts": deepcopy(
                ion.get("formula_dicts", [ion.get("formula_dict", {})])
            ),
            "source_role": str(ion.get("source_role", "")),
            "source_roles": list(
                ion.get("source_roles", [ion.get("source_role", "")])
            ),
            "hfa_name": str(ion.get("hfa_name", "")),
            "hfa_names": list(ion.get("hfa_names", [ion.get("hfa_name", "")])),
            "hfa_abbr": str(ion.get("hfa_abbr", "")),
            "hfa_carbon": ion.get("hfa_carbon"),
            "hfa_double_bonds": ion.get("hfa_double_bonds"),
            "found": match.found,
            "exp_mz": match.exp_mz,
            "experimental_mz": match.exp_mz,
            "ppm_err": match.ppm_error,
            "ppm_error": match.ppm_error,
            "intensity": match.intensity,
            "sn_value": match.sn_value,
            "threshold": _threshold_for_diagnostic_ion(config, ion),
            "cluster_id": cluster.get("cluster_id"),
            "cluster_candidate_keys": list(cluster.get("candidate_keys", [candidate["key"]])),
            "is_shared": bool(cluster.get("is_shared", False)),
            "is_unique": bool(cluster.get("is_unique", True)),
            "raw_peak_indices": list(ion.get("raw_peak_indices", [])),
            "member_mzs": list(ion.get("member_mzs", [float(ion["mz"])])),
            "member_ions": deepcopy(ion.get("member_ions", [ion])),
            "merged_member_count": int(ion.get("merged_member_count", 1)),
            "is_merged": bool(ion.get("is_merged", False)),
        }
    return detected


def compute_dynamic_wdic(
    detected: Dict[str, Dict[str, Any]], weights: Dict[int, float]
) -> float:
    denominator = sum(float(weights.get(int(item["level"]), 0.0)) for item in detected.values())
    numerator = sum(
        float(weights.get(int(item["level"]), 0.0))
        for item in detected.values()
        if item["found"]
    )
    return numerator / denominator if denominator else 0.0


def compute_dynamic_nep(
    detected: Dict[str, Dict[str, Any]], weights: Dict[int, float]
) -> float:
    l3 = [item for item in detected.values() if int(item["level"]) == 3]
    denominator = sum(float(weights.get(3, 0.0)) for _item in l3)
    missing = sum(float(weights.get(3, 0.0)) for item in l3 if not item["found"])
    return missing / denominator if denominator else 0.0


def _level_statistics(detected: Dict[str, Dict[str, Any]]) -> Dict[int, Dict[str, int]]:
    stats: Dict[int, Dict[str, int]] = defaultdict(lambda: {"found": 0, "total": 0})
    for item in detected.values():
        level = int(item["level"])
        stats[level]["total"] += 1
        if item["found"]:
            stats[level]["found"] += 1
    return dict(stats)


def _is_fa3_protonated_ion(ion: Dict[str, Any]) -> bool:
    keys = [ion.get("fragment_key")]
    fragment_keys = ion.get("fragment_keys") or []
    if isinstance(fragment_keys, str):
        fragment_keys = [fragment_keys]
    keys.extend(fragment_keys)
    return "fa3_protonated" in {str(key) for key in keys if key}


def annotate_fa3_protonated_evidence(rows: Sequence[Dict[str, Any]]) -> None:
    """Attach FA3+H identity and strongest-ion annotations without scoring changes."""
    by_fa3: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any]]]] = defaultdict(list)
    for row in rows:
        fa3_ions = [
            ion for ion in row.get("detected_ions", {}).values()
            if _is_fa3_protonated_ion(ion)
        ]
        if len(fa3_ions) > 1:
            row["fa3_protonated_warning"] = (
                "Multiple observable FA3+H ions were merged into the strongest available interpretation."
            )
        ion = max(
            fa3_ions,
            key=lambda item: (bool(item.get("found")), float(item.get("intensity") or 0.0)),
            default=None,
        )
        found = bool(ion and ion.get("found"))
        row.update(
            {
                "fa3_protonated_found": found,
                "fa3_protonated_exp_mz": ion.get("exp_mz") if found else None,
                "fa3_protonated_ppm_error": ion.get("ppm_err") if found else None,
                "fa3_protonated_intensity": float(ion.get("intensity") or 0.0) if found else 0.0,
                "fa3_protonated_threshold": float(ion.get("threshold", 0.0)) if ion else 0.0,
                "fa3_identity_supported": found,
                "fa3_protonated_rank": None,
                "fa3_protonated_is_strongest": False,
                "fa3_protonated_ratio_to_strongest": 0.0,
                "fa3_protonated_strongest_to_second_ratio": None,
            }
        )
        if found and row.get("fa3_name"):
            by_fa3[str(row["fa3_name"])].append((row, ion))

    representatives: List[Tuple[str, float]] = []
    for fa3_name, entries in by_fa3.items():
        representatives.append(
            (fa3_name, max(float(ion.get("intensity") or 0.0) for _row, ion in entries))
        )
    representatives.sort(key=lambda item: (-item[1], item[0]))
    if not representatives:
        return

    strongest_intensity = representatives[0][1]
    second_intensity = representatives[1][1] if len(representatives) > 1 else None
    ranks: Dict[str, int] = {}
    previous_intensity: Optional[float] = None
    previous_rank = 0
    for position, (fa3_name, intensity) in enumerate(representatives, start=1):
        if previous_intensity is not None and math.isclose(
            intensity, previous_intensity, rel_tol=0.0, abs_tol=1e-12
        ):
            ranks[fa3_name] = previous_rank
        else:
            ranks[fa3_name] = position
            previous_rank = position
            previous_intensity = intensity

    ratio = (
        strongest_intensity / second_intensity
        if second_intensity is not None and second_intensity > 0
        else float("inf")
    )
    for fa3_name, entries in by_fa3.items():
        for row, _ion in entries:
            intensity = float(row["fa3_protonated_intensity"])
            row.update(
                {
                    "fa3_protonated_rank": ranks[fa3_name],
                    "fa3_protonated_is_strongest": ranks[fa3_name] == 1,
                    "fa3_protonated_ratio_to_strongest": intensity / max(strongest_intensity, 1e-12),
                    "fa3_protonated_strongest_to_second_ratio": ratio,
                }
            )


def detect_dynamic_mixture(
    matcher: HDCSSpectrumMatcher,
    candidates: Sequence[Dict[str, Any]],
    clusters: Sequence[Dict[str, Any]],
    config: HDCSConfig,
) -> Dict[str, Any]:
    if not config.mixture_detection_enabled:
        return {"is_mixed": False, "sn_mixed": False, "ester_mixed": False, "evidence": []}
    candidate_by_key = {candidate["key"]: candidate for candidate in candidates}
    sn_evidence: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    ester_evidence: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for cluster in clusters:
        levels = set(cluster["levels"])
        groups = set(cluster["groups"])
        if 3 in levels:
            sn_groups = {group for group in groups if group in {"sn-1/3", "sn-2"}}
            if len(sn_groups) == 1:
                match = matcher.detect(cluster["mz"], config.ppm_tolerance, config.l3_threshold)
                if match.found:
                    group = next(iter(sn_groups))
                    sn_evidence[group].append({"cluster_id": cluster["cluster_id"], "match": match})
        if 4 in levels:
            ester_groups = {group for group in groups if group.startswith("C")}
            if len(ester_groups) == 1:
                match = matcher.detect(cluster["mz"], config.ppm_tolerance, config.l4_threshold)
                if match.found:
                    ester_group = next(iter(ester_groups))
                    sn_groups = {
                        "sn-1/3" if candidate_by_key[key]["sn_fahfa"] == "sn-1" else "sn-2"
                        for key in cluster["candidate_keys"]
                        if key in candidate_by_key
                    }
                    for sn_group in sn_groups:
                        ester_evidence[(sn_group, ester_group)].append(
                            {"cluster_id": cluster["cluster_id"], "match": match}
                        )

    sn_strength = {
        group: max(item["match"].intensity for item in entries)
        for group, entries in sn_evidence.items()
    }
    sn_mixed = False
    if len(sn_strength) > 1:
        strongest = max(sn_strength.values())
        sn_mixed = sum(
            1
            for intensity in sn_strength.values()
            if intensity >= config.sn_mixture_min_intensity
            and intensity / max(strongest, 1e-12) >= config.sn_mixture_ratio
        ) > 1

    positions_by_sn: Dict[str, Dict[str, float]] = defaultdict(dict)
    for (sn_group, ester_group), entries in ester_evidence.items():
        positions_by_sn[sn_group][ester_group] = sum(item["match"].intensity for item in entries)
    qualified_positions: Dict[str, List[str]] = {}
    for sn_group, positions in positions_by_sn.items():
        strongest = max(positions.values()) if positions else 0.0
        qualified_positions[sn_group] = [
            position
            for position, intensity in positions.items()
            if intensity >= config.ester_mixture_min_intensity
            and intensity / max(strongest, 1e-12) >= config.ester_mixture_ratio
        ]
    ester_mixed = any(len(positions) > 1 for positions in qualified_positions.values())
    evidence: List[str] = []
    if sn_mixed:
        evidence.append("Mutually exclusive sn-1/3 and sn-2 L3 evidence was detected.")
    if ester_mixed:
        evidence.append("Multiple qualified L4 ester-position groups were detected within one sn group.")
    return {
        "is_mixed": bool(sn_mixed or ester_mixed),
        "sn_mixed": sn_mixed,
        "ester_mixed": ester_mixed,
        "sn_groups": dict(sn_evidence),
        "sn_strength": sn_strength,
        "ester_groups_by_sn": dict(qualified_positions),
        "evidence": evidence,
    }


def dynamic_nnls_decompose(
    matcher: HDCSSpectrumMatcher,
    candidates: Sequence[Dict[str, Any]],
    config: HDCSConfig,
) -> Dict[str, Any]:
    if not candidates:
        raise ValueError("NNLS requires at least one qualified candidate")
    clusters = cluster_dynamic_diagnostic_ions(candidates, config.shared_ion_tolerance_ppm)
    matrix = np.zeros((len(clusters), len(candidates)), dtype=float)
    exp_vector = np.zeros(len(clusters), dtype=float)
    key_to_column = {candidate["key"]: index for index, candidate in enumerate(candidates)}
    for row_index, cluster in enumerate(clusters):
        for key in cluster["candidate_keys"]:
            matrix[row_index, key_to_column[key]] = 1.0
        threshold = min(
            _threshold_for_diagnostic_ion(config, member)
            for member in cluster["members"]
        )
        match = matcher.detect(float(cluster["mz"]), config.ppm_tolerance, threshold)
        if match.found:
            exp_vector[row_index] = match.intensity
    if not matrix.size or not np.any(matrix):
        raise ValueError("NNLS diagnostic matrix is empty")
    if np.max(exp_vector) > 0:
        exp_vector = exp_vector / np.max(exp_vector)
    coefficients_raw, residual = scipy_nnls(matrix, exp_vector)
    total = float(np.sum(coefficients_raw))
    coefficients = coefficients_raw / total if total > 0 else coefficients_raw
    rank = int(np.linalg.matrix_rank(matrix))
    condition = float(np.linalg.cond(matrix)) if matrix.size else float("inf")
    residual_rmse = float(residual) / math.sqrt(max(len(clusters), 1))
    fit_quality = float(np.clip(1.0 - float(residual) / max(float(np.linalg.norm(exp_vector)), 1e-12), 0.0, 1.0))
    reasons: List[str] = []
    if len(candidates) == 1:
        reasons.append("only one distinguishable HDCS 1.1 fingerprint")
    if rank < len(candidates):
        reasons.append("candidate diagnostic matrix is rank deficient")
    if not math.isfinite(condition) or condition > 1e6:
        reasons.append("condition number exceeds the reliability limit")
    if residual_rmse > 0.25:
        reasons.append("normalized residual exceeds the reliability limit")
    coefficient_map = {
        candidate["key"]: float(coefficients[index]) for index, candidate in enumerate(candidates)
    }
    return {
        "candidate_keys": [candidate["key"] for candidate in candidates],
        "candidate_names": [candidate["name"] for candidate in candidates],
        "alphas": coefficients.tolist(),
        "coefficients": coefficient_map,
        "residual": float(residual),
        "residual_rmse": residual_rmse,
        "fit_quality": fit_quality,
        "matrix_rank": rank,
        "n_candidates": len(candidates),
        "n_dimensions": len(clusters),
        "condition_number": condition,
        "is_reliable": not reasons,
        "unreliable_reason": "; ".join(reasons),
        "solver": "scipy.optimize.nnls",
    }


def _matches_nnls_target(row: Dict[str, Any], target_score: float) -> bool:
    """Match the raw, unrounded HDCS score, allowing only float noise."""
    score = float(row.get("final_score", 0.0))
    return math.isfinite(score) and math.isclose(
        score, target_score, rel_tol=0.0, abs_tol=1e-6
    )


def _has_zero_nep(row: Dict[str, Any]) -> bool:
    """Apply the Tier I/II requirement using the raw NEP value."""
    nep = float(row.get("nep", float("nan")))
    return math.isfinite(nep) and math.isclose(
        nep, 0.0, rel_tol=0.0, abs_tol=NEP_ZERO_TOLERANCE
    )


def _is_nnls_eligible(row: Dict[str, Any], target_score: float) -> bool:
    """Tier I: NEP is zero and raw Final HDCS equals the NNLS target."""
    return _has_zero_nep(row) and _matches_nnls_target(row, target_score)


def _meets_display_threshold(
    row: Dict[str, Any],
    minimum_score: float,
    maximum_score: float = DISPLAY_MAX_HDCS_SCORE,
) -> bool:
    """Tier I/II main list: NEP is zero and raw HDCS is in the range."""
    score = float(row.get("final_score", 0.0))
    return (
        _has_zero_nep(row)
        and math.isfinite(score)
        and score >= minimum_score - 1e-6
        and score <= maximum_score + 1e-6
    )


def _requires_manual_review(row: Dict[str, Any]) -> bool:
    """Tier III review pool: every candidate with a finite raw NEP above zero."""
    nep = float(row.get("nep", float("nan")))
    return math.isfinite(nep) and nep > NEP_ZERO_TOLERANCE


def _select_dynamic_nnls_candidates(
    candidates: Sequence[Dict[str, Any]],
    ranked_results: Sequence[Dict[str, Any]],
    target_score: float,
) -> List[Dict[str, Any]]:
    by_key = {candidate["key"]: candidate for candidate in candidates}
    selected: List[Dict[str, Any]] = []
    seen_groups: Set[str] = set()
    # Screen before building the NNLS matrix. Keep every distinct fingerprint;
    # an arbitrary top-N limit would change the fitted coefficients.
    for row in ranked_results:
        if not _is_nnls_eligible(row, target_score):
            continue
        candidate = by_key[row["candidate_key"]]
        group = candidate.get("equivalent_group", candidate["key"])
        if group in seen_groups:
            continue
        seen_groups.add(group)
        selected.append(candidate)
    return selected


def get_l4_position_owners(
    ion: Dict[str, Any],
    current_family_key: Any,
    result_by_key: Dict[str, Dict[str, Any]],
) -> Set[int]:
    """Return ester positions represented by an L4 cluster inside one family."""
    owners: Set[int] = set()
    for candidate_key in ion.get("cluster_candidate_keys", []):
        owner = result_by_key.get(str(candidate_key))
        if owner is None or owner.get("position_family_key") != current_family_key:
            continue
        try:
            owners.add(int(owner["ester_pos"]))
        except (KeyError, TypeError, ValueError):
            continue
    return owners


def resolve_position_families(rows: Sequence[Dict[str, Any]]) -> None:
    """Attach family metadata without collapsing individual position candidates."""
    families: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        families[row["position_family_key"]].append(row)

    for family_index, family_rows in enumerate(families.values(), start=1):
        all_positions = sorted({int(row["ester_pos"]) for row in family_rows})
        supported_positions = sorted(
            {int(row["ester_pos"]) for row in family_rows if row.get("position_supported")}
        )
        unsupported_positions = [
            position for position in all_positions if position not in supported_positions
        ]
        family_label = f"PF-{family_index:04d}"
        candidate_keys = [str(row["candidate_key"]) for row in family_rows]
        for row in family_rows:
            row.update(
                {
                    "position_family": family_label,
                    "position_family_candidate_keys": candidate_keys,
                    "position_family_size": len(family_rows),
                    "family_all_ester_positions": list(all_positions),
                    "family_supported_ester_positions": list(supported_positions),
                    "family_unsupported_ester_positions": list(unsupported_positions),
                    "multiple_positions_supported": len(supported_positions) > 1,
                }
            )


DISPLAY_RANK_SCORE_TOLERANCE = 1e-6


def get_display_score(row: Dict[str, Any]) -> float:
    """Return the numeric score used by every GUI display-row sort."""
    return float(
        row.get(
            "display_final_score",
            row.get("display_score", row.get("final_score", 0.0)),
        )
    )


def hdcs_score_sort_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    """Globally sort GUI rows by score, then deterministic same-score details."""
    display_type_order = {
        "confirmed_position": 0,
        "unresolved_positions": 1,
    }
    try:
        ester_position_rank = int(row.get("reported_ester_position"))
    except (TypeError, ValueError):
        ester_position_rank = 10**9
    stable_name = str(
        row.get(
            "reported_lipid_name",
            row.get("candidate_name", row.get("candidate_key", "")),
        )
    )
    return (
        -get_display_score(row),
        display_type_order.get(str(row.get("display_type", "")), 9),
        -float(row.get("display_l4_score", row.get("l4_score", 0.0))),
        ester_position_rank,
        stable_name,
    )


def nnls_display_sort_key(row: Dict[str, Any]) -> Tuple[Any, ...]:
    """Sort by the unrounded NNLS value, with HDCS as a stable tie-breaker."""
    if not bool(row.get("nnls_eligible", False)):
        return (float("inf"), hdcs_score_sort_key(row))
    value = float(
        row.get(
            "display_nnls_contribution",
            row.get("nnls_contribution", 0.0),
        )
    )
    return (
        -value if math.isfinite(value) else float("inf"),
        hdcs_score_sort_key(row),
    )


def assign_display_ranks(
    display_rows: Sequence[Dict[str, Any]],
    tolerance: float = DISPLAY_RANK_SCORE_TOLERANCE,
) -> None:
    """Assign competition ranks after rows have received their global score order."""
    previous_score: Optional[float] = None
    previous_rank = 0
    for position, row in enumerate(display_rows, start=1):
        score = get_display_score(row)
        if (
            previous_score is not None
            and math.isclose(score, previous_score, rel_tol=0.0, abs_tol=tolerance)
        ):
            row["display_rank"] = previous_rank
        else:
            row["display_rank"] = position
            previous_rank = position
            previous_score = score


def build_position_resolved_display_rows(
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build one display row per supported position plus one unresolved group."""
    families: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        families[row["position_family_key"]].append(row)

    display_rows: List[Dict[str, Any]] = []
    for family_rows in families.values():
        family_label = str(family_rows[0]["position_family"])
        all_positions = sorted({int(row["ester_pos"]) for row in family_rows})
        supported_positions = sorted(
            {int(row["ester_pos"]) for row in family_rows if row.get("position_supported")}
        )
        unsupported_positions = [
            position for position in all_positions if position not in supported_positions
        ]
        family_candidate_keys = [str(row["candidate_key"]) for row in family_rows]
        family_max_base_score = max(float(row["base_score"]) for row in family_rows)
        family_sort_key = (-family_max_base_score, family_label)

        for position in supported_positions:
            position_rows = [
                row
                for row in family_rows
                if int(row["ester_pos"]) == position and row.get("position_supported")
            ]
            representative = max(
                position_rows,
                key=lambda row: (
                    float(row["l4_score"]),
                    float(row["final_score"]),
                    float(row["base_score"]),
                ),
            )
            display_base_score = float(representative["base_score"])
            display_l4_score = float(representative["l4_score"])
            display_nnls_contribution = sum(
                float(row.get("nnls_contribution", 0.0)) for row in position_rows
            )
            display_row = dict(representative)
            display_row.update(
                {
                    "display_id": f"display-{family_label}-C{position}",
                    "display_type": "confirmed_position",
                    "candidate_key": representative["candidate_key"],
                    "representative_candidate_key": representative["candidate_key"],
                    "display_member_candidate_keys": [row["candidate_key"] for row in position_rows],
                    "reported_common_name": representative["theoretical_common_name"],
                    "reported_lipid_name": representative["theoretical_lipid_name"],
                    "position_resolution_status": "Confirmed",
                    "ester_position_confirmed": True,
                    "reported_ester_position": position,
                    "supported_ester_positions": [position],
                    "unresolved_ester_positions": [],
                    "family_supported_ester_positions": list(supported_positions),
                    "family_unsupported_ester_positions": list(unsupported_positions),
                    "multiple_positions_supported": len(supported_positions) > 1,
                    "position_family_candidate_keys": family_candidate_keys,
                    "position_family_size": len(family_rows),
                    "family_candidate_keys": family_candidate_keys,
                    "family_size": len(family_rows),
                    "display_base_score": display_base_score,
                    "display_l4_score": display_l4_score,
                    "display_final_score": display_base_score + display_l4_score,
                    "display_score": display_base_score + display_l4_score,
                    "display_nnls_contribution": float(display_nnls_contribution),
                    "display_family_sort_key": family_sort_key,
                    "display_within_family_order": (0, -display_l4_score, position),
                    "wdic_pass": any(bool(row.get("wdic_pass")) for row in position_rows),
                }
            )
            display_rows.append(display_row)

        if unsupported_positions:
            unresolved_rows = [
                row for row in family_rows if int(row["ester_pos"]) in unsupported_positions
            ]
            representative = max(
                unresolved_rows,
                key=lambda row: (
                    float(row["base_score"]),
                    float(row["wdic"]),
                    -int(row["ester_pos"]),
                ),
            )
            display_base_score = max(float(row["base_score"]) for row in unresolved_rows)
            display_nnls_contribution = sum(
                float(row.get("nnls_contribution", 0.0)) for row in unresolved_rows
            )
            display_row = dict(representative)
            display_row.update(
                {
                    "display_id": f"display-{family_label}-unresolved",
                    "display_type": "unresolved_positions",
                    "candidate_key": representative["candidate_key"],
                    "representative_candidate_key": representative["candidate_key"],
                    "display_member_candidate_keys": [row["candidate_key"] for row in unresolved_rows],
                    "reported_common_name": remove_ester_position_from_common_name(
                        representative["theoretical_common_name"]
                    ),
                    "reported_lipid_name": remove_ester_position_from_lipid_name(
                        representative["theoretical_lipid_name"]
                    ),
                    "position_resolution_status": "Unresolved",
                    "ester_position_confirmed": False,
                    "reported_ester_position": None,
                    "supported_ester_positions": [],
                    "unresolved_ester_positions": list(unsupported_positions),
                    "family_supported_ester_positions": list(supported_positions),
                    "family_unsupported_ester_positions": list(unsupported_positions),
                    "multiple_positions_supported": len(supported_positions) > 1,
                    "position_family_candidate_keys": family_candidate_keys,
                    "position_family_size": len(family_rows),
                    "family_candidate_keys": family_candidate_keys,
                    "family_size": len(family_rows),
                    "display_base_score": display_base_score,
                    "display_l4_score": 0.0,
                    "display_final_score": display_base_score,
                    "display_score": display_base_score,
                    "display_nnls_contribution": float(display_nnls_contribution),
                    "display_family_sort_key": family_sort_key,
                    "display_within_family_order": (1, 0.0, 10**9),
                    "wdic_pass": any(bool(row.get("wdic_pass")) for row in unresolved_rows),
                }
            )
            display_rows.append(display_row)

    for original_order, row in enumerate(display_rows):
        row["display_original_order"] = original_order
    display_rows.sort(key=hdcs_score_sort_key)
    assign_display_ranks(display_rows)
    return display_rows


def score_dynamic_hdcs_spectrum(
    experimental_spectrum: pd.DataFrame,
    candidates: Sequence[Dict[str, Any]],
    config: Optional[HDCSConfig] = None,
) -> Dict[str, Any]:
    """Score every dynamic TG-FAHFA candidate globally with HDCS-MD logic."""
    config = deepcopy(config) if config is not None else HDCSConfig()
    # Keep the 1.1-only NNLS rule even when an older batch caller supplies a cutoff.
    config.nnls_min_final_score = TARGET_HDCS_SCORE
    if experimental_spectrum is None or experimental_spectrum.empty:
        raise ValueError("Experimental spectrum is empty")
    if not candidates:
        raise ValueError("Dynamic HDCS candidate list is empty")
    warnings: List[str] = []
    if "sn_value" not in experimental_spectrum or not experimental_spectrum["sn_value"].notna().any():
        warnings.append("S/N column unavailable; relative-intensity fallback used.")
    if len(candidates) > 500:
        warnings.append(f"Large candidate set ({len(candidates)}); scoring may take longer.")

    for candidate in candidates:
        if candidate.get("raw_diagnostic_ions") is None:
            candidate["raw_diagnostic_ions"] = deepcopy(candidate.get("diagnostic_ions", []))
        candidate["diagnostic_ions"] = merge_candidate_diagnostic_ions(
            candidate.get("raw_diagnostic_ions", []), config.shared_ion_tolerance_ppm
        )
        if not candidate["diagnostic_ions"]:
            raise ValueError(
                f"Candidate {candidate.get('key', '<unknown>')} produced no merged HDCS diagnostic ions"
            )
        candidate.setdefault(
            "theoretical_common_name",
            format_common_name_for_display(str(candidate.get("name") or candidate.get("key") or "")),
        )
        theoretical_lipid_name = str(candidate.get("structure") or candidate.get("name") or "")
        if theoretical_lipid_name and not theoretical_lipid_name.startswith("TG "):
            theoretical_lipid_name = f"TG {theoretical_lipid_name}"
        candidate.setdefault("theoretical_lipid_name", theoretical_lipid_name)
        candidate["position_family_key"] = build_position_family_key(candidate)

    assign_equivalence_groups(candidates)
    equivalence_sizes: Dict[str, int] = defaultdict(int)
    for candidate in candidates:
        equivalence_sizes[str(candidate["equivalent_group"])] += 1
    largest_equivalence_group = max(equivalence_sizes.values(), default=0)
    if largest_equivalence_group > 20:
        warnings.append(
            f"Large scoring-equivalence group detected ({largest_equivalence_group} candidates)."
        )
    matcher = HDCSSpectrumMatcher(experimental_spectrum)
    clusters = cluster_dynamic_diagnostic_ions(candidates, config.shared_ion_tolerance_ppm)
    rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        detected = scan_dynamic_diagnostic_ions(matcher, candidate, config, clusters)
        wdic = compute_dynamic_wdic(detected, config.level_weights)
        nep = compute_dynamic_nep(detected, config.level_weights)
        base_score = wdic * (1.0 - config.nep_lambda * nep)
        rows.append(
            {
                "candidate_key": candidate["key"],
                "candidate_name": candidate["name"],
                "sn_fahfa": candidate["sn_fahfa"],
                "ester_pos": candidate["ester_pos"],
                "FAHFA_name": candidate["FAHFA_name"],
                "fa3_name": candidate.get("fa3_name", ""),
                "hfa_name": candidate.get("hfa_name", ""),
                "fa3_protonated_theoretical_mz": float(
                    candidate.get("fa3_protonated_theoretical_mz", 0.0)
                ),
                "precursor_mz": candidate["precursor_mz"],
                "equivalent_group": candidate["equivalent_group"],
                "position_family_key": candidate["position_family_key"],
                "theoretical_common_name": candidate["theoretical_common_name"],
                "theoretical_lipid_name": candidate["theoretical_lipid_name"],
                "wdic": float(wdic),
                "wdic_pass": bool(wdic > config.wdic_pass_threshold),
                "nep": float(nep),
                "base_score": float(base_score),
                "base_net_score": float(base_score),
                "detected_ions": detected,
                "level_stats": _level_statistics(detected),
            }
        )

    annotate_fa3_protonated_evidence(rows)
    result_by_key = {str(row["candidate_key"]): row for row in rows}
    for row in rows:
        current_position = int(row["ester_pos"])
        family_key = row["position_family_key"]
        candidate_l4_ions = [
            ion
            for ion in row["detected_ions"].values()
            if int(ion.get("level", 0)) == 4 or 4 in ion.get("levels", [])
        ]
        found_l4_ions = [ion for ion in candidate_l4_ions if ion.get("found", False)]
        position_specific_l4: List[Dict[str, Any]] = []
        for ion in candidate_l4_ions:
            position_owners = get_l4_position_owners(ion, family_key, result_by_key)
            ion["position_owners"] = sorted(position_owners)
            ion["is_position_specific"] = (
                len(position_owners) == 1 and current_position in position_owners
            )
            if ion["is_position_specific"]:
                position_specific_l4.append(ion)
        found_position_specific_l4 = [
            ion for ion in position_specific_l4 if ion.get("found", False)
        ]
        eligible_for_l4 = bool(row["wdic_pass"])
        l4_score = (
            config.l4_bonus_max
            * len(found_position_specific_l4)
            / len(position_specific_l4)
            if eligible_for_l4 and position_specific_l4
            else 0.0
        )
        position_supported = bool(
            eligible_for_l4 and len(found_position_specific_l4) >= 1
        )
        row.update(
            {
                "l4_score": float(l4_score),
                "l4_bonus": float(l4_score),
                "final_score": float(row["base_score"] + l4_score),
                "display_score": float(row["base_score"] + l4_score),
                "matched_ions": [ion for ion in row["detected_ions"].values() if ion["found"]],
                "missing_ions": [ion for ion in row["detected_ions"].values() if not ion["found"]],
                "exclusive_evidence": found_position_specific_l4,
                "contradictory_evidence": [],
                "eligible_for_l4": eligible_for_l4,
                "position_supported": position_supported,
                "l4_position_total_count": len(candidate_l4_ions),
                "l4_position_found_count": len(found_l4_ions),
                "l4_position_specific_total_count": len(position_specific_l4),
                "l4_position_specific_found_count": len(found_position_specific_l4),
                "l4_position_found_ratio": (
                    len(found_l4_ions) / len(candidate_l4_ions)
                    if candidate_l4_ions else 0.0
                ),
                "unique_l4_found_count": len(found_position_specific_l4),
                "unique_l4_total_intensity": float(
                    sum(ion["intensity"] for ion in found_position_specific_l4)
                ),
                "shared_l4_found_count": len(found_l4_ions) - len(found_position_specific_l4),
            }
        )

    resolve_position_families(rows)

    if not any(row["final_score"] > 0 for row in rows):
        warnings.append("All dynamic candidates received a zero HDCS-MD score.")

    rows.sort(key=lambda row: (-row["final_score"], -row["wdic"], row["candidate_name"]))
    previous_score: Optional[float] = None
    previous_rank = 0
    for position, row in enumerate(rows, start=1):
        if previous_score is not None and math.isclose(
            row["final_score"], previous_score, abs_tol=1e-6
        ):
            row["rank"] = previous_rank
        else:
            row["rank"] = position
            previous_rank = position
            previous_score = row["final_score"]

    mixture = detect_dynamic_mixture(matcher, candidates, clusters, config)
    for row in rows:
        expected_group = "sn-1/3" if row["sn_fahfa"] == "sn-1" else "sn-2"
        opposing = "sn-2" if expected_group == "sn-1/3" else "sn-1/3"
        if opposing in mixture.get("sn_groups", {}):
            row["contradictory_evidence"] = mixture["sn_groups"][opposing]

    qualified_rows = [
        row for row in rows
        if _is_nnls_eligible(row, config.nnls_min_final_score)
    ]
    display_qualified_rows = [
        row for row in rows
        if _meets_display_threshold(row, DISPLAY_MIN_HDCS_SCORE)
    ]
    nnls_candidates = _select_dynamic_nnls_candidates(
        candidates, qualified_rows, config.nnls_min_final_score
    )
    nnls_result: Optional[Dict[str, Any]] = None
    if config.nnls_enabled and nnls_candidates:
        try:
            nnls_result = dynamic_nnls_decompose(matcher, nnls_candidates, config)
            nnls_result["target_final_score"] = float(config.nnls_min_final_score)
            if not nnls_result.get("is_reliable", False):
                warnings.append(
                    "NNLS fit may be unreliable: "
                    + str(nnls_result.get("unreliable_reason") or "quality checks failed")
                )
        except Exception as exc:
            warnings.append(f"NNLS unavailable: {exc}")
    elif config.nnls_enabled:
        warnings.append(
            "NNLS not run: no candidate has raw NEP = 0 and Final HDCS Score 1.1."
        )
    coefficients = dict(nnls_result.get("coefficients", {})) if nnls_result else {}
    group_coefficients = {
        str(candidate["equivalent_group"]): float(coefficients.get(candidate["key"], 0.0))
        for candidate in nnls_candidates
    }
    qualified_keys = {str(row["candidate_key"]) for row in qualified_rows}
    for row in rows:
        eligible = str(row["candidate_key"]) in qualified_keys
        row["nnls_eligible"] = eligible
        row["nnls_contribution"] = (
            group_coefficients.get(str(row["equivalent_group"]), 0.0)
            if eligible else 0.0
        )

    mixture_candidates = [
        row["candidate_key"]
        for row in qualified_rows
        if row["nnls_contribution"] > 0.01
    ]
    if not mixture_candidates and mixture.get("is_mixed"):
        mixture_candidates = [row["candidate_key"] for row in qualified_rows[:2]]
    result_by_key = {row["candidate_key"]: row for row in rows}
    # GUI/export display and NNLS use deliberately different candidate spaces:
    # Tier I/II main list: NEP == 0 and 1.0 <= raw HDCS <= 1.1.
    # Tier I NNLS: NEP == 0 and raw HDCS == 1.1.
    display_candidate_rows = build_position_resolved_display_rows(display_qualified_rows)
    group_display_ids: Dict[str, Set[str]] = defaultdict(set)
    for display_row in display_candidate_rows:
        groups = {
            str(result_by_key[str(key)]["equivalent_group"])
            for key in display_row["display_member_candidate_keys"]
        }
        eligible_groups = {
            str(result_by_key[str(key)]["equivalent_group"])
            for key in display_row["display_member_candidate_keys"]
            if str(key) in qualified_keys
        }
        display_row["nnls_group_ids"] = sorted(groups)
        display_row["nnls_eligible"] = bool(eligible_groups)
        # One coefficient belongs to one fingerprint, not to each duplicate row.
        display_row["display_nnls_contribution"] = (
            sum(group_coefficients.get(group, 0.0) for group in eligible_groups)
            if eligible_groups else None
        )
        for group in eligible_groups:
            group_display_ids[group].add(str(display_row["display_id"]))
    if any(len(ids) > 1 for ids in group_display_ids.values()):
        warnings.append(
            "An indistinguishable NNLS fingerprint appears in multiple display rows; "
            "its repeated coefficient must not be summed across those rows."
        )
    return {
        "algorithm": "HDCS-MD Dynamic",
        "ranked_candidates": rows,
        "ranking": rows,
        "results_by_key": result_by_key,
        "display_candidate_rows": display_candidate_rows,
        "mixture_detected": bool(mixture.get("is_mixed")),
        "mixture_candidates": mixture_candidates,
        "mixture_detection": mixture,
        "nnls_coefficients": coefficients,
        "nnls_group_coefficients": group_coefficients,
        "nnls_fit_quality": float(nnls_result.get("fit_quality", 0.0)) if nnls_result else 0.0,
        "nnls_min_final_score": float(config.nnls_min_final_score),
        "display_min_final_score": float(DISPLAY_MIN_HDCS_SCORE),
        "display_max_final_score": float(DISPLAY_MAX_HDCS_SCORE),
        "display_requires_zero_nep": True,
        "nnls_eligible_candidate_keys": sorted(qualified_keys),
        "manual_review_candidate_keys": sorted(
            str(row["candidate_key"]) for row in rows
            if _requires_manual_review(row)
        ),
        "decomposition": nnls_result,
        "fragment_cluster_registry": clusters,
        "warnings": warnings,
        "config": config,
    }

def is_even_carbon_fa(fa_key: str) -> bool:
    """Return True only for known fatty-acid keys with an even carbon count."""
    normalized = format_fa_string(fa_key)
    return normalized in FA_INFO and FA_INFO[normalized][0] % 2 == 0

def ppm_error(exp_mz, theo_mz):
    if theo_mz == 0: return float('inf')
    return abs(exp_mz - theo_mz) / theo_mz * 1e6

def calc_mass(C, H, O, Na=0):
    return C * MASS['C'] + H * MASS['H'] + O * MASS['O'] + Na * MASS['Na']

def fa_formula(c, db): return {"C": c, "H": 2 * c - 2 * db, "O": 2}
def fa_mass(c, db): return calc_mass(c, 2 * c - 2 * db, 2)

def build_fahfa_formula(c_acyl, db_acyl, c_hfa, db_hfa):
    return {
        'C': c_acyl + c_hfa,
        'H': (2 * c_acyl - 2 * db_acyl) + (2 * c_hfa - 2 * db_hfa) - 2,
        'O': 4,
    }

def build_tg_fahfa_formula(fahfa_form, fa3_form, fa4_form):
    return {
        'C': fahfa_form['C'] + fa3_form['C'] + fa4_form['C'] + 3,
        'H': fahfa_form['H'] + fa3_form['H'] + fa4_form['H'] + 2,
        'O': 8,
    }

def fahfa_na_mass(formula): return calc_mass(formula['C'], formula['H'], formula['O'], Na=1)
def fahfa_cho2_loss_mass(formula): return calc_mass(formula['C'] - 1, formula['H'] - 1, formula['O'] - 2, Na=1)
def tg_na_mass(formula): return calc_mass(formula['C'], formula['H'], formula['O'], Na=1)

def lookup_fa_abbr(carbons, double_bonds): return FA_NAMES.get((carbons, double_bonds), f"{carbons}:{double_bonds}")
def lookup_hfa_abbr(carbons, double_bonds):
    key = (carbons, double_bonds)
    if key in HFA_ABBRS: return HFA_ABBRS[key]
    base = FA_NAMES.get(key)
    if base: return f"H{base}"
    return f"H{carbons}:{double_bonds}"

def get_accurate_abbr(fa_str):
    if '(' in fa_str:
        m = re.match(r'(\d+:\d+)\((\d+)-O-(\d+:\d+)\)', fa_str)
        acyl, pos, hfa = m.groups()
        c_a, db_a = map(int, acyl.split(':'))
        c_h, db_h = map(int, hfa.split(':'))
        abbr_a = FA_NAMES.get((c_a, db_a), acyl)
        base = FA_NAMES.get((c_h, db_h))
        abbr_h = HFA_ABBRS.get((c_h, db_h), f"H{base}" if base else f"H{hfa}")
        return f"{pos}-{abbr_a}{abbr_h}"
    else:
        c, db = map(int, fa_str.split(':'))
        return FA_NAMES.get((c, db), fa_str)

#############################################################################
#  SMILES PARSER — Pure-Python graph-topology engine
#############################################################################
class SMILESParser:
    def __init__(self, smiles: str):
        self.smiles = smiles.strip()
        self.atoms: List[Dict[str, Any]] = []
        self.bonds: List[Tuple[int, int, int]] = []
        self._parse()

    def _parse(self):
        i, stack, prev_idx, bo = 0, [], -1, 1
        s = self.smiles

        while i < len(s):
            c = s[i]
            if c == '=': bo = 2; i += 1; continue
            if c == '#': bo = 3; i += 1; continue
            if c in ':/\\': bo = 1; i += 1; continue
            if c == '(': stack.append(prev_idx); i += 1; continue
            if c == ')': prev_idx = stack.pop(); i += 1; continue
            if c == '.': prev_idx = -1; bo = 1; i += 1; continue

            if c == '[':
                j = s.index(']', i)
                atom = self._parse_bracket(s[i + 1:j])
                idx = len(self.atoms)
                self.atoms.append(atom)
                if prev_idx >= 0: self.bonds.append((prev_idx, idx, bo))
                prev_idx = idx; bo = 1; i = j + 1; continue

            if c == 'C' and i + 1 < len(s) and s[i + 1] == 'l': elem, skip = 'Cl', 2
            elif c == 'B' and i + 1 < len(s) and s[i + 1] == 'r': elem, skip = 'Br', 2
            elif c in 'BCNOPSFIC': elem, skip = c, 1
            else: elem, skip = None, 0

            if elem:
                i += skip
                h_count = 0
                while i < len(s) and s[i].isdigit():
                    h_count = h_count * 10 + int(s[i]); i += 1
                idx = len(self.atoms)
                self.atoms.append({'element': elem, 'explicit_h': h_count, 'charge': 0, 'aromatic': False})
                if prev_idx >= 0: self.bonds.append((prev_idx, idx, bo))
                prev_idx = idx; bo = 1; continue

            if c in 'cnosp':
                idx = len(self.atoms)
                self.atoms.append({'element': c.upper(), 'explicit_h': 0, 'charge': 0, 'aromatic': True})
                if prev_idx >= 0: self.bonds.append((prev_idx, idx, 1))
                prev_idx = idx; bo = 1; i += 1; continue
            i += 1

    @staticmethod
    def _parse_bracket(content: str) -> Dict[str, Any]:
        result = {'element': '', 'explicit_h': 0, 'charge': 0, 'aromatic': False}
        m = re.match(r'^(\d+)', content)
        if m: content = content[m.end():]
        m = re.match(r'^([A-Z][a-z]?)', content)
        if m:
            result['element'] = m.group(1)
            content = content[m.end():]
        elif content and content[0].islower():
            result['element'] = content[0].upper()
            result['aromatic'] = True
            content = content[1:]
        h_match = re.search(r'H(\d*)', content)
        if h_match: result['explicit_h'] = int(h_match.group(1)) if h_match.group(1) else 1
        content = re.sub(r'@+[THSP]*\d*H?\d*', '', content)
        cm = re.search(r'([+-])(\d*)', content)
        if cm:
            sign = 1 if cm.group(1) == '+' else -1
            result['charge'] = sign * (int(cm.group(2)) if cm.group(2) else 1)
        return result

    def get_exact_mass(self) -> float:
        bond_orders: Dict[int, int] = {}
        for ai, aj, b in self.bonds:
            bond_orders[ai] = bond_orders.get(ai, 0) + b
            bond_orders[aj] = bond_orders.get(aj, 0) + b
        formula: Dict[str, int] = {}
        total_charge = 0

        for idx, atom in enumerate(self.atoms):
            elem = atom['element']
            eh = atom.get('explicit_h', 0)
            chg = atom.get('charge', 0)
            total_charge += chg
            formula[elem] = formula.get(elem, 0) + 1
            formula['H'] = formula.get('H', 0) + eh
            if elem in VALENCES:
                implicit_h = max(0, VALENCES[elem] - bond_orders.get(idx, 0) - abs(chg) - eh)
                formula['H'] = formula.get('H', 0) + implicit_h

        mass = sum(ATOMIC_MASSES.get(e, 0.0) * n for e, n in formula.items())
        mass -= total_charge * ELECTRON_MASS
        return mass


# =============================================================================
# Subgraph -> SMILES Exporter (Used for drawing fragment structures)
# =============================================================================
class SMILESWriter:
    @staticmethod
    def from_subgraph(
        parser: SMILESParser,
        atom_indices: Set[int],
        neighbors: Dict[int, List[int]],
        bond_orders: Dict[Tuple[int, int], int],
        sodiated: bool = True,
    ) -> str:
        organic = {i for i in atom_indices if parser.atoms[i]['element'] != 'Na'}
        if not organic: return '[Na+]' if sodiated else ''

        components: List[Set[int]] = []
        remaining = set(organic)
        while remaining:
            start = max(remaining, key=lambda i: (ATOMIC_MASSES.get(parser.atoms[i]['element'], 0), i))
            comp: Set[int] = set()
            queue = [start]
            while queue:
                curr = queue.pop()
                if curr not in remaining: continue
                comp.add(curr)
                remaining.discard(curr)
                for nb in neighbors[curr]:
                    if nb in remaining: queue.append(nb)
            components.append(comp)

        parts = [SMILESWriter._write_component(parser, comp, neighbors, bond_orders) for comp in components]
        parts = [p for p in parts if p]
        smiles = '.'.join(parts)
        if sodiated: smiles = f"{smiles}.[Na+]" if smiles else '[Na+]'
        return smiles

    @staticmethod
    def _write_component(parser: SMILESParser, atoms: Set[int], neighbors: Dict[int, List[int]], bond_orders: Dict[Tuple[int, int], int]) -> str:
        if not atoms: return ''
        start = max(atoms, key=lambda i: (ATOMIC_MASSES.get(parser.atoms[i]['element'], 0), i))
        visited: Set[int] = set()
        return SMILESWriter._dfs(parser, start, atoms, neighbors, bond_orders, visited, is_root=True)

    @staticmethod
    def _dfs(parser: SMILESParser, idx: int, atoms: Set[int], neighbors: Dict[int, List[int]], bond_orders: Dict[Tuple[int, int], int], visited: Set[int], is_root: bool, parent: int = -1, parent_bo: int = 1) -> str:
        visited.add(idx)
        atom = parser.atoms[idx]
        elem = atom['element']
        aromatic = atom.get('aromatic', False)
        charge = atom.get('charge', 0)
        explicit_h = atom.get('explicit_h', 0)

        total_bo = sum(bond_orders.get((idx, n), 0) for n in neighbors[idx] if n in atoms)
        impl_h = 0
        if elem in VALENCES:
            impl_h = max(0, VALENCES[elem] - total_bo - abs(charge) - explicit_h)

        symbol = SMILESWriter._atom_symbol(elem, aromatic, explicit_h, impl_h, charge, is_root or parent_bo > 1)

        child_edges: List[Tuple[int, int]] = []
        for nb in neighbors[idx]:
            if nb in atoms and nb not in visited:
                bo = bond_orders.get((idx, nb), 1)
                child_edges.append((nb, bo))
        child_edges.sort(key=lambda x: (-ATOMIC_MASSES.get(parser.atoms[x[0]]['element'], 0), x[0]))

        if not child_edges: return symbol

        branches: List[str] = []
        for nb, bo in child_edges:
            bond_prefix = '=' if bo == 2 else '#' if bo == 3 else ''
            branches.append(bond_prefix + SMILESWriter._dfs(parser, nb, atoms, neighbors, bond_orders, visited, False, idx, bo))

        if len(branches) == 1: return symbol + branches[0]
        return symbol + branches[0] + ''.join(f'({b})' for b in branches[1:])

    @staticmethod
    def _atom_symbol(elem: str, aromatic: bool, explicit_h: int, impl_h: int, charge: int, in_branch: bool) -> str:
        h_total = explicit_h if explicit_h > 0 else impl_h
        needs_bracket = charge != 0 or explicit_h > 0
        if aromatic and not needs_bracket: return elem.lower()
        if not needs_bracket:
            if elem == 'C': return 'C'
            if elem == 'O' and h_total == 0: return 'O'
            if elem in ('N', 'S', 'P') and h_total == 0: return elem
        parts = elem
        if explicit_h > 0: parts += f'H{explicit_h if explicit_h > 1 else ""}'
        elif impl_h > 0: parts += f'H{impl_h if impl_h > 1 else ""}'
        if charge > 0: parts += '+' if charge == 1 else f'+{charge}'
        elif charge < 0: parts += '-' if charge == -1 else str(charge)
        return f'[{parts}]'

# =============================================================================
#  CANDIDATE GENERATOR — TG-FAHFA isomer enumeration
# =============================================================================
# LipidMAPS double bond positions (carboxyl carbon = C1)
DB_POSITIONS = {
    (14, 1): [9],      (16, 1): [9],      (18, 1): [9],
    (20, 1): [11],     (22, 1): [13],     (24, 1): [15],
    (18, 2): [9, 12],  (18, 3): [9, 12, 15],  (18, 4): [6, 9, 12, 15],
    (20, 2): [11, 14], (20, 3): [8, 11, 14],  (20, 4): [5, 8, 11, 14],
    (20, 5): [5, 8, 11, 14, 17],
    (22, 2): [13, 16], (22, 3): [13, 16, 19], (22, 4): [7, 10, 13, 16],
    (22, 5): [4, 7, 10, 13, 16],  (22, 6): [4, 7, 10, 13, 16, 19],
    (24, 2): [15, 18], (24, 4): [9, 12, 15, 18], (24, 5): [9, 12, 15, 18, 21],
    (24, 6): [6, 9, 12, 15, 18, 21],
}

def get_r_alkyl(c, db):
    """Return R-group SMILES for a fatty acid chain (C:db) with real double bond positions."""
    if c <= 1: return ""
    if db == 0: return "C" * (c - 1)
    positions = DB_POSITIONS.get((c, db))
    if not positions:
        step = max(3, (c - 3) // (db + 1))
        positions = [step * i + 2 for i in range(1, db + 1)]
    parts = []
    prev = 2
    for pos in sorted(positions):
        n_span = pos - prev
        if n_span > 0:
            parts.append("C" * n_span)
        parts.append("C=C")
        prev = pos + 2
    n_tail = c - prev + 1
    if n_tail > 0:
        parts.append("C" * n_tail)
    return "".join(parts)

def get_fa_alkyl_smiles(c, db):
    if c <= 1: return ""
    return get_r_alkyl(c, db)

def generate_hfa_alkyl_smiles(c_hfa, d_hfa, acyl_fa_key, oh_pos):
    """
    Generate the HFA alkyl chain SMILES, preserving backbone unsaturation.
    Numbering: C1 = carboxyl carbon (C=O, outside this string)
               C2 = first aliphatic carbon in the returned string
               Branch at C(oh_pos) from the carboxyl end.
    Uses get_r_alkyl for both backbone and branch to preserve all C=C bonds.
    """
    c_acyl, d_acyl = FA_INFO.get(acyl_fa_key, (0, 0))
    r_acyl = get_r_alkyl(c_acyl, d_acyl)  # branch chain with real double bond positions
    r_hfa = get_r_alkyl(c_hfa, d_hfa)    # HFA backbone with real double bond positions

    # Insert branch at the (oh_pos-1)th C of the backbone
    # (C1 is the carboxyl carbon outside this string, C2 is first alkyl C)
    target_c = oh_pos - 1
    c_count = 0
    smiles = ""
    for char in r_hfa:
        smiles += char
        if char.upper() == 'C':
            c_count += 1
            if c_count == target_c:
                branch = f"(OC(=O){r_acyl})" if r_acyl else "(OC(=O))"
                smiles += branch
    return smiles

def generate_tg_smiles_general(sn1_alkyl, sn2_alkyl, sn3_alkyl):
    return f"C(OC(=O){sn1_alkyl})C(OC(=O){sn2_alkyl})COC(=O){sn3_alkyl}.[Na+]"

def count_carbon_double_bonds(smiles_str):
    """Count C=C double bonds in a SMILES string (excluding C=O bonds)."""
    count = 0
    i = 0
    while i < len(smiles_str):
        if smiles_str[i:i+2] == "C=":
            # Check if the next char after "C=" is "C" (C=C, not C=O)
            if i+2 < len(smiles_str) and smiles_str[i+2] == 'C':
                count += 1
        i += 1
    return count

def validate_candidate_structure(name, smiles, expected_formula_dict, verbose=True):
    """
    Cross-validate candidate name, SMILES, and formula.
    Returns (is_valid, errors_list).
    """
    errors = []
    # Count expected double bonds from formula
    C = expected_formula_dict.get('C', 0)
    H = expected_formula_dict.get('H', 0)
    O = expected_formula_dict.get('O', 0)
    # Degree of unsaturation: DoU = C - H/2 - O/2 + 1 (for neutral molecules without N)
    expected_dou = max(0, C - H // 2 - O // 2 + 1)
    
    # Count C=C bonds in SMILES (C=C patterns, excluding C=O)
    smiles_cc_count = count_carbon_double_bonds(smiles)
    
    if verbose:
        print(f"  [DEBUG] Candidate: {name}")
        print(f"  [DEBUG]   Expected formula: C{C}H{H}O{O}, DoU={expected_dou}")
        print(f"  [DEBUG]   C=C bonds in SMILES: {smiles_cc_count}")
    
    if smiles_cc_count != expected_dou:
        # But DoU also accounts for rings and C=O bonds, not just C=C
        # The difference should be the number of C=O bonds (ester groups)
        # Each TG has 4 C=O bonds (3 glycerol esters + 1 FAHFA branch)
        c_o_bonds = smiles.count("C=O")
        if smiles_cc_count != expected_dou:
            err = f"DoU mismatch: expected {expected_dou}, C=C bonds={smiles_cc_count}"
            if verbose:
                print(f"  [WARNING] {err}")
            errors.append(err)
    
    if verbose:
        print()
    return len(errors) == 0, errors

def verify_smiles_ester_position(hfa_alkyl_smiles, expected_oh_pos):
    c_count = 0
    for ch in hfa_alkyl_smiles:
        if ch == '(': break
        if ch == 'C': c_count += 1
    actual_oh_pos = c_count + 1
    if actual_oh_pos != expected_oh_pos:
        raise ValueError(f"SMILES ester position mismatch: expected {expected_oh_pos}, but got {actual_oh_pos}")
    return True

# =====================================================
# Core Matching and Identification
# =====================================================
def match_parent_to_library(parent_mz_exp, library):
    matches = []
    for entry in library:
        mz_theo = entry['mz']
        if ppm_error(parent_mz_exp, mz_theo) <= PPM_PARENT:
            matches.append(entry)
    return matches

def load_spectrum(file):
    if not os.path.exists(file): raise FileNotFoundError(f"File not found: {file}")
    try: sep = '\t' if len(pd.read_csv(file, nrows=5).columns) < 2 else ','
    except Exception as exc:
        sep = ','
    df = pd.read_csv(file, sep=sep)
    df.columns = [str(c).lower().strip() for c in df.columns]

    mz_col = next((name for name in ['m/z', 'mz', 'mass', 'mass-to-charge'] if name in df.columns), None)
    if mz_col is None:
        if len(df.columns) > 0: mz_col = df.columns[0]
        else: raise ValueError("CSV file is empty.")
    if mz_col != 'mz': df.rename(columns={mz_col: 'mz'}, inplace=True)
    df['mz'] = pd.to_numeric(df['mz'], errors='coerce')

    int_col = next((name for name in ['intensity', 'int', 'intens', 'a.u.'] if name in df.columns), None)
    if int_col is None:
        for col in df.columns:
            if col != 'mz' and 'sn' not in col:
                int_col = col; break
    if int_col and int_col != 'intensity': df.rename(columns={int_col: 'intensity'}, inplace=True)
    df['intensity'] = pd.to_numeric(df.get('intensity', 0), errors='coerce')
    df = df.dropna(subset=['mz'])

    sn_col = next((name for name in ['sn', 's/n', 'signal/noise'] if name in df.columns), None)
    if sn_col: df['sn_value'] = pd.to_numeric(df[sn_col], errors='coerce')
    else: df['sn_value'] = np.nan
    return df


#############################################################################
#  INPUT PREVIEW AND COLUMN MAPPING
#############################################################################

MZ_COLUMN_ALIASES = (
    "Mass/Charge (Da)",
    "Mass/Charge(Da)",
    "Mass / Charge (Da)",
    "Mass / Charge(Da)",
    "m/z",
    "M/Z",
    "mz",
    "Mass-to-Charge",
    "Mass to Charge",
    "Mass",
)
INTENSITY_COLUMN_ALIASES = (
    "Height",
    "Hight",
    "Intensity",
    "Peak Height",
    "Abundance",
    "Area",
    "Peak Area",
)

def read_raw_spectrum_table(file_path: str) -> Tuple[pd.DataFrame, str]:
    """Read a comma- or tab-delimited spectrum without renaming source columns."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    encoding_used = ""
    sample = ""
    for encoding in ("utf-8-sig", "utf-8"):
        try:
            with open(file_path, "r", encoding=encoding, newline="") as handle:
                sample = handle.read(65536)
            encoding_used = encoding
            break
        except UnicodeDecodeError:
            continue
    if not encoding_used:
        raise ValueError("The input file is not valid UTF-8 or UTF-8-SIG text.")
    if not sample.strip():
        raise ValueError("The input file is empty.")
    try:
        delimiter = csv.Sniffer().sniff(sample, delimiters=",\t").delimiter
    except csv.Error:
        delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
    delimiter_name = "Tab" if delimiter == "\t" else "Comma"
    first_line = sample.splitlines()[0] if sample.splitlines() else ""
    raw_headers = next(csv.reader([first_line], delimiter=delimiter), [])
    try:
        frame = pd.read_csv(file_path, sep=delimiter, encoding=encoding_used)
    except pd.errors.EmptyDataError as exc:
        raise ValueError("The input file contains no tabular data.") from exc
    if len(frame.columns) == 0:
        raise ValueError("The input file contains no columns.")
    if len(frame) == 0:
        raise ValueError("The input file contains column names but no data rows.")
    warnings: List[str] = []
    duplicate_headers = sorted(
        {name for name in raw_headers if raw_headers.count(name) > 1}
    )
    if duplicate_headers:
        warnings.append(
            "Duplicate source columns were made unique by pandas: "
            + ", ".join(str(name) for name in duplicate_headers)
        )
    frame.attrs["input_warnings"] = warnings
    frame.attrs["delimiter"] = delimiter_name
    frame.attrs["encoding"] = encoding_used
    frame.attrs["raw_headers"] = list(raw_headers)
    return frame, delimiter_name


def _normalized_column_name(value: Any) -> str:
    """Normalize a source heading for conservative alias matching."""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _infer_column_from_aliases(
    columns: Sequence[str], aliases: Sequence[str]
) -> Optional[str]:
    """Return the original heading using exact-first, then normalized matching."""
    source_columns = [str(column) for column in columns]
    for alias in aliases:
        for column in source_columns:
            if column.strip() == alias:
                return column
    normalized_columns = [
        (column, _normalized_column_name(column)) for column in source_columns
    ]
    for alias in aliases:
        normalized_alias = _normalized_column_name(alias)
        for column, normalized_column in normalized_columns:
            if normalized_column == normalized_alias:
                return column
    return None


def infer_mz_column(columns: Sequence[str]) -> Optional[str]:
    """Recommend an m/z heading without falling back to a positional column."""
    return _infer_column_from_aliases(columns, MZ_COLUMN_ALIASES)


def infer_intensity_column(columns: Sequence[str]) -> Optional[str]:
    """Recommend an explicit intensity-like column without positional fallback."""
    return _infer_column_from_aliases(columns, INTENSITY_COLUMN_ALIASES)


def infer_spectrum_column_mapping(
    columns: Sequence[str],
) -> Dict[str, Optional[str]]:
    """Recommend selectable source columns while preserving their real names."""
    return {
        "mz": infer_mz_column(columns),
        "intensity": infer_intensity_column(columns),
    }


def valid_initial_column(
    requested: Optional[str],
    available_columns: Sequence[str],
    fallback: Optional[str],
) -> str:
    """Use a saved mapping only while its original source heading still exists."""
    available = [str(column) for column in available_columns]
    if requested in available:
        return str(requested)
    if fallback in available:
        return str(fallback)
    return ""


def normalize_spectrum_from_mapping(
    raw_df: pd.DataFrame,
    mapping: Dict[str, Optional[str]],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Create traceable standard spectrum columns from a confirmed mapping."""
    if raw_df is None or raw_df.empty:
        raise ValueError("The raw input table is empty.")
    mz_column = mapping.get("mz")
    if not mz_column or mz_column not in raw_df.columns:
        raise ValueError("Select a valid m/z column.")
    intensity_column = mapping.get("intensity")
    if not intensity_column or intensity_column not in raw_df.columns:
        raise ValueError("Select a valid Intensity column.")
    if mz_column == intensity_column:
        raise ValueError("m/z and Intensity must use different columns.")
    mz_values = pd.to_numeric(raw_df[mz_column], errors="coerce")
    intensity_values = pd.to_numeric(raw_df[intensity_column], errors="coerce")
    finite_mz = mz_values.map(lambda value: bool(pd.notna(value) and math.isfinite(float(value))))
    finite_intensity = intensity_values.map(
        lambda value: bool(pd.notna(value) and math.isfinite(float(value)))
    )
    positive_mz = finite_mz & (mz_values > 0)
    nonnegative_intensity = finite_intensity & (intensity_values >= 0)
    valid_mask = positive_mz & nonnegative_intensity
    source_rows = pd.Series(np.arange(len(raw_df), dtype=int) + 2, index=raw_df.index)
    normalized = pd.DataFrame(
        {
            "source_row": source_rows.loc[valid_mask].astype(int),
            "mz": mz_values.loc[valid_mask].astype(float),
            "intensity": intensity_values.loc[valid_mask].astype(float),
            "calculated_sn": np.nan,
        }
    ).reset_index(drop=True)
    stats = {
        "original_row_count": int(len(raw_df)),
        "valid_row_count": int(valid_mask.sum()),
        "invalid_row_count": int((~valid_mask).sum()),
        "numeric_conversion_failure_count": int((~(finite_mz & finite_intensity)).sum()),
        "nonpositive_mz_count": int((finite_mz & (mz_values <= 0)).sum()),
        "negative_intensity_count": int(
            (finite_intensity & (intensity_values < 0)).sum()
        ),
        "column_mapping": dict(mapping),
    }
    if normalized.empty:
        raise ValueError("No valid m/z-intensity rows remain after mapping validation.")
    return normalized, stats


def resolve_precursor_from_base_peak(
    spectrum: pd.DataFrame,
) -> Tuple[float, float, int]:
    """Return m/z, intensity, and source row for the highest valid SCIEX peak."""
    if spectrum is None or spectrum.empty:
        raise ValueError("No valid peaks are available for precursor selection.")
    if "mz" not in spectrum.columns or "intensity" not in spectrum.columns:
        raise ValueError("Precursor selection requires m/z and intensity columns.")
    working = spectrum.copy()
    working["mz"] = pd.to_numeric(working["mz"], errors="coerce")
    working["intensity"] = pd.to_numeric(working["intensity"], errors="coerce")
    valid = (
        working["mz"].notna()
        & working["intensity"].notna()
        & np.isfinite(working["mz"])
        & np.isfinite(working["intensity"])
        & (working["mz"] > 0)
        & (working["intensity"] >= 0)
    )
    working = working.loc[valid].copy()
    if working.empty:
        raise ValueError("No valid non-negative-intensity peaks are available for precursor selection.")
    maximum_intensity = float(working["intensity"].max())
    tied = working[working["intensity"] == maximum_intensity].copy()
    selected = tied.sort_values(
        by=["mz", "source_row"] if "source_row" in tied.columns else ["mz"],
        ascending=[False, True] if "source_row" in tied.columns else [False],
        kind="stable",
    ).iloc[0]
    source_row = int(selected.get("source_row", int(selected.name) + 2))
    return float(selected["mz"]), maximum_intensity, source_row


#############################################################################
#  EXPORT HELPERS
#############################################################################

def to_json_safe(value: Any) -> Any:
    """Convert analysis values into finite, UTF-8 JSON-safe primitives."""
    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, (str, int)):
        return value
    if pd.isna(value):
        return None
    return str(value)


def sanitize_filename(value: Any) -> str:
    """Return a Windows-safe file or directory name component."""
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(value)).strip(" .")
    return text or "TG-FAHFA_Result"


def _export_join(values: Any) -> str:
    """Join list-like export values without exposing Python representations."""
    if values is None:
        return ""
    if isinstance(values, (str, bytes)):
        return str(values)
    if isinstance(values, (list, tuple, set)):
        return "; ".join(str(value) for value in values if value not in (None, ""))
    return str(values)


def filter_spectrum_by_manual_sn(
    spectrum: pd.DataFrame,
    noise: float,
    sn_threshold: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply the authoritative manual Noise/S/N filter without mutating input data."""
    if spectrum is None or spectrum.empty:
        raise ValueError("Experimental spectrum is empty.")
    if not math.isfinite(noise) or noise <= 0:
        raise ValueError("Noise must be greater than 0.")
    if not math.isfinite(sn_threshold) or sn_threshold < 0:
        raise ValueError("S/N threshold cannot be negative.")
    if "mz" not in spectrum.columns or "intensity" not in spectrum.columns:
        raise ValueError("Experimental spectrum requires m/z and intensity columns.")

    working = spectrum.copy()
    working["mz"] = pd.to_numeric(working["mz"], errors="coerce")
    working["intensity"] = pd.to_numeric(working["intensity"], errors="coerce")
    working = working.dropna(subset=["mz", "intensity"]).copy()
    working = working[working["intensity"] >= 0].copy()
    if working.empty:
        raise ValueError("No valid m/z-intensity peaks were found.")

    # Manual Noise/S/N mode is authoritative for this analysis.
    working["calculated_sn"] = working["intensity"] / noise
    working["sn_value"] = working["calculated_sn"]
    original_peak_count = len(working)
    minimum_intensity = noise * sn_threshold
    retained = working[working["calculated_sn"] >= sn_threshold].copy()
    retained = retained.sort_values("mz").reset_index(drop=True)
    retained_peak_count = len(retained)
    stats = {
        "original_peak_count": int(original_peak_count),
        "retained_peak_count": int(retained_peak_count),
        "removed_peak_count": int(original_peak_count - retained_peak_count),
        "noise": float(noise),
        "sn_threshold": float(sn_threshold),
        "minimum_intensity": float(minimum_intensity),
    }
    return retained, stats

def get_parent_ion(spectrum):
    if 'intensity' not in spectrum.columns or spectrum['intensity'].isna().all():
        print("   Warning: no valid intensity column found; using max m/z as parent.")
        return spectrum['mz'].max(), 0
    max_idx = spectrum['intensity'].idxmax()
    return spectrum.loc[max_idx, 'mz'], spectrum.loc[max_idx, 'intensity']

def identify_FA_guided(parent_mz_theo, spectrum, parent_mz_exp=None):
    results = []
    spec_to_search = spectrum
    spec_mz = spec_to_search["mz"].values

    for mz in spec_mz:
        nl_theo = parent_mz_theo - mz
        nl_exp = (parent_mz_exp - mz) if parent_mz_exp is not None else nl_theo
        if 200 <= nl_theo <= 450 or (parent_mz_exp is not None and 200 <= nl_exp <= 450):
            for fa, (c, db) in FA_INFO.items():
                theo = fa_mass(c, db)
                match_theo = ppm_error(nl_theo, theo) <= PPM
                match_exp = (parent_mz_exp is not None and ppm_error(nl_exp, theo) <= PPM)
                if match_theo or match_exp:
                    best_nl = nl_theo if match_theo else nl_exp
                    results.append({"FA": fa, "Fragment": mz, "NL": best_nl, "PPM": ppm_error(best_nl, theo)})
    return pd.DataFrame(results)

def build_fahfa_library(valid_acyl_fas):
    records = []
    valid_acyl_fas = [format_fa_string(fa) for fa in valid_acyl_fas if is_even_carbon_fa(fa)]
    for fa1 in valid_acyl_fas:
        if not is_even_carbon_fa(fa1): continue
        c1, d1 = FA_INFO[fa1]
        for fa2 in FA_INFO.keys():
            if not is_even_carbon_fa(fa2): continue
            c2, d2 = FA_INFO[fa2]
            formula = build_fahfa_formula(c1, d1, c2, d2)
            max_oh = min(13, c2 - 1)
            for oh_pos in range(3, max_oh + 1):
                records.append({
                    "FAHFA_Internal": f"{fa1}/O-{fa2}",
                    "FAHFA_Display": f"{fa1}({oh_pos}-O-{fa2})",
                    "Formula": f"C{formula['C']}H{formula['H']}O4",
                    "C": formula["C"], "H": formula["H"], "O": formula["O"],
                    "Mz_Na": fahfa_na_mass(formula), "Mz_CHO2": fahfa_cho2_loss_mass(formula),
                    "Acyl_FA": fa1, "Hydroxy_FA": fa2, "OH_Pos": oh_pos
                })
    return pd.DataFrame(records)

def identify_fahfa(spectrum, valid_acyl_fas):
    if not valid_acyl_fas: return pd.DataFrame()
    fahfa_db = build_fahfa_library(valid_acyl_fas)
    hits = []
    region = spectrum[(spectrum["mz"] >= 450) & (spectrum["mz"] <= 750)]
    spec_mz_set = set(np.round(spectrum["mz"].values, 4))

    for _, row in fahfa_db.iterrows():
        target_mz = row["Mz_Na"]
        target_cho2 = row["Mz_CHO2"]
        matched_peaks = region[np.abs(region["mz"] - target_mz) / target_mz * 1e6 <= PPM]
        if matched_peaks.empty: continue

        for peak_row in matched_peaks.itertuples():
            if any(ppm_error(mz2, target_cho2) <= PPM for mz2 in spec_mz_set):
                hits.append({
                    "Peak": peak_row.mz, "FAHFA_Internal": row["FAHFA_Internal"],
                    "FAHFA_Display": row["FAHFA_Display"], "Formula": row["Formula"],
                    "Mz_Na": row["Mz_Na"], "Mz_CHO2_Theo": row["Mz_CHO2"],
                    "C": row["C"], "H": row["H"], "O": row["O"],
                    "Acyl_FA": row["Acyl_FA"], "Hydroxy_FA": row["Hydroxy_FA"], "OH_Pos": row["OH_Pos"]
                })
    return pd.DataFrame(hits)

def build_candidates_strict(parent_mz_exp, parent_mz_theo, fa_list, fahfa_hits, target_c, target_db):
    results = []
    if not fa_list or fahfa_hits.empty: return pd.DataFrame()

    fa_list = [format_fa_string(fa) for fa in fa_list if is_even_carbon_fa(fa)]
    if not fa_list: return pd.DataFrame()
    fa_formulas = {fa: fa_formula(*FA_INFO[fa]) for fa in fa_list if is_even_carbon_fa(fa)}

    for _, fh in fahfa_hits.iterrows():
        acyl_fa, hydroxy_fa, fahfa_display = format_fa_string(fh["Acyl_FA"]), format_fa_string(fh["Hydroxy_FA"]), fh["FAHFA_Display"]
        oh_pos, fahfa_form = fh["OH_Pos"], {"C": fh["C"], "H": fh["H"], "O": fh["O"]}

        if not is_even_carbon_fa(acyl_fa) or not is_even_carbon_fa(hydroxy_fa):
            continue

        fahfa_c = FA_INFO[acyl_fa][0] + FA_INFO[hydroxy_fa][0]
        fahfa_db = FA_INFO[acyl_fa][1] + FA_INFO[hydroxy_fa][1]

        for fa_tuple in itertools.combinations_with_replacement(fa_list, 2):
            fa1, fa2 = fa_tuple
            if (fa1 not in fa_formulas or fa2 not in fa_formulas
                    or not is_even_carbon_fa(fa1) or not is_even_carbon_fa(fa2)):
                continue

            current_total_c = fahfa_c + FA_INFO[fa1][0] + FA_INFO[fa2][0]
            current_total_db = fahfa_db + FA_INFO[fa1][1] + FA_INFO[fa2][1]

            if current_total_c != target_c or current_total_db != target_db: continue

            tg_form = build_tg_fahfa_formula(fahfa_form, fa_formulas[fa1], fa_formulas[fa2])
            theo_mass = tg_na_mass(tg_form)
            formula_str = f"C{tg_form['C']}H{tg_form['H']}O{tg_form['O']}"

            if ppm_error(parent_mz_theo, theo_mass) <= 5.0:
                ppm_val_exp = ppm_error(parent_mz_exp, theo_mass)

                hfa_smiles_alkyl = generate_hfa_alkyl_smiles(FA_INFO[hydroxy_fa][0], FA_INFO[hydroxy_fa][1], acyl_fa, oh_pos)
                verify_smiles_ester_position(hfa_smiles_alkyl, oh_pos)
                fa1_smiles_alkyl = get_fa_alkyl_smiles(FA_INFO[fa1][0], FA_INFO[fa1][1])
                fa2_smiles_alkyl = get_fa_alkyl_smiles(FA_INFO[fa2][0], FA_INFO[fa2][1])

                if not hfa_smiles_alkyl or not fa1_smiles_alkyl or not fa2_smiles_alkyl: continue

                canonical_sn1_fa, canonical_sn3_fa = canonicalize_sn2_outer_fas(fa1, fa2)
                canonical_sn1_smiles = get_fa_alkyl_smiles(
                    FA_INFO[canonical_sn1_fa][0], FA_INFO[canonical_sn1_fa][1]
                )
                canonical_sn3_smiles = get_fa_alkyl_smiles(
                    FA_INFO[canonical_sn3_fa][0], FA_INFO[canonical_sn3_fa][1]
                )
                if not canonical_sn1_smiles or not canonical_sn3_smiles:
                    continue

                raw_isomers = [
                    (1, f"{fahfa_display}/{fa1}/{fa2}", "Sn-1/3", fahfa_display, fa1, fa2, hfa_smiles_alkyl, fa1_smiles_alkyl, fa2_smiles_alkyl),
                    (2, f"{fahfa_display}/{fa2}/{fa1}", "Sn-1/3", fahfa_display, fa2, fa1, hfa_smiles_alkyl, fa2_smiles_alkyl, fa1_smiles_alkyl),
                    (3, f"{canonical_sn1_fa}/{fahfa_display}/{canonical_sn3_fa}", "Sn-2", canonical_sn1_fa, fahfa_display, canonical_sn3_fa, canonical_sn1_smiles, hfa_smiles_alkyl, canonical_sn3_smiles),
                ]

                seen_desc = set()
                for idx, desc, pos, sn1, sn2, sn3, sm1, sm2, sm3 in raw_isomers:
                    if desc not in seen_desc:
                        seen_desc.add(desc)
                        smiles_str = generate_tg_smiles_general(sm1, sm2, sm3)
                        tg_form = build_tg_fahfa_formula(fahfa_form, fa_formulas[fa1], fa_formulas[fa2])
                        # The GUI reports progress in its status line; avoid flooding the
                        # Windows console with one debug block for every dynamic candidate.
                        validate_candidate_structure(desc, smiles_str, tg_form, verbose=False)
                        results.append({
                            "Parent_Exp": round(parent_mz_exp, 4),
                            "Parent_Theo": round(theo_mass, 5),
                            "PPM": round(ppm_val_exp, 2),
                            "FAHFA": fahfa_display, "FA1": fa1, "FA2": fa2,
                            "FA3": acyl_fa, "HFA": hydroxy_fa,
                            "Ester_Position": int(oh_pos),
                            "FAHFA_Position": pos, "TG_FAHFA_Structure": desc,
                            "FA_Sn1": sn1, "FA_Sn2": sn2, "FA_Sn3": sn3,
                            "Formula": formula_str, "SMILES": smiles_str, "SortOrder": idx
                        })
    return pd.DataFrame(results)

#############################################################################
#  FRAGMENT GENERATOR — TG-FAHFA EAD MS/MS spectrum predictor
#############################################################################
class TGFAHFASpectrumGenerator:
    def __init__(self):
        self.ead_offsets = EAD_OFFSETS
        self.na_mass = NA_MASS
        self._last_msp_name = "TG-FAHFA"
        self._last_common_name = "TG-FAHFA"
        self._last_fahfa_sn = None  # sn position of FAHFA chain
        self.last_raw_peaks: List[Dict[str, Any]] = []
        self.last_display_peaks: List[Dict[str, Any]] = []

    @property
    def last_msp_name(self) -> str: return self._last_msp_name
    @property
    def last_common_name(self) -> str: return self._last_common_name

    @staticmethod
    def _fa_lipid_label(info: Dict[str, Any]) -> str: return f"{info['C']}:{info['db']}"

    @staticmethod
    def _hfa_branch_ester_position(parser: SMILESParser, neighbors: Dict[int, List[int]], hfa_atoms: Set[int], carbonyl_c: int, attachment_c: int) -> int:
        if attachment_c not in hfa_atoms or carbonyl_c not in hfa_atoms: return 0
        best = 0
        def dfs(curr: int, pos: int, visited: Set[int]) -> None:
            nonlocal best
            if curr == attachment_c: best = max(best, pos)
            for nb in neighbors[curr]:
                if nb in visited or nb not in hfa_atoms: continue
                elem = parser.atoms[nb]['element']
                if elem == 'C':
                    visited.add(nb)
                    dfs(nb, pos + 1, visited)
                    visited.remove(nb)
                elif elem == 'O':
                    visited.add(nb)
                    for nn in neighbors[nb]:
                        if nn in visited or nn not in hfa_atoms: continue
                        if parser.atoms[nn]['element'] == 'C':
                            visited.add(nn)
                            dfs(nn, pos + 1, visited)
                            visited.remove(nn)
                            break
                    visited.remove(nb)
        dfs(carbonyl_c, 1, {carbonyl_c})
        return best

    @staticmethod
    def _fahfa_lipid_label(chain: Dict[str, Any]) -> str:
        branch, hfa, pos = chain['branch'], chain['hfa'], chain.get('ester_position', 0)
        return f"{branch['C']}:{branch['db']}-({pos}-O-{hfa['C']}:{hfa['db']})"

    @staticmethod
    def _ordered_chains(chains_info: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sn_order = {'sn1': 0, 'sn2': 1, 'sn3': 2}
        return sorted(chains_info, key=lambda c: sn_order.get(c.get('sn', 'sn1'), 9))

    @staticmethod
    def _build_msp_name(chains_info: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for chain in TGFAHFASpectrumGenerator._ordered_chains(chains_info):
            parts.append(TGFAHFASpectrumGenerator._fahfa_lipid_label(chain) if chain['is_fahfa'] else TGFAHFASpectrumGenerator._fa_lipid_label(chain['info']))
        return f"TG-FAHFA({'/'.join(parts)})"

    @staticmethod
    def _fahfa_common_label(chain: Dict[str, Any]) -> str:
        pos = chain.get('ester_position', 0)
        return f"{pos}-{chain['name']}" if pos else chain['name']

    @staticmethod
    def _build_common_name(chains_info: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for chain in TGFAHFASpectrumGenerator._ordered_chains(chains_info):
            parts.append(TGFAHFASpectrumGenerator._fahfa_common_label(chain) if chain['is_fahfa'] else chain['info']['abbr'])
        return f"TG({'/'.join(parts)})"

    @staticmethod
    def strip_sodium_adduct(smiles: str) -> Tuple[str, bool]:
        s = smiles.strip()
        had_salt = bool(re.search(r'(?:^|\.)\[Na\+\](?:\.|$)', s))
        s = re.sub(r'\.\[Na\+\]', '', s)
        s = re.sub(r'\[Na\+\]\.', '', s)
        parts = [p.strip() for p in s.split('.') if p.strip() and p.strip() != '[Na+]']
        return '.'.join(parts).strip('.'), had_salt

    def _get_fragment_mass(self, parser: SMILESParser, atom_indices: Set[int], neighbors: Dict[int, List[int]], bond_orders: Dict[Tuple[int, int], int]) -> float:
        mass = 0.0
        for idx in atom_indices:
            atom = parser.atoms[idx]
            elem = atom['element']
            mass += ATOMIC_MASSES.get(elem, 0.0)
            eh = atom.get('explicit_h', 0)
            mass += eh * ATOMIC_MASSES['H']
            if elem in VALENCES:
                total_bo = sum(bond_orders.get((idx, n), 0) for n in neighbors[idx])
                impl_h = max(0, VALENCES[elem] - total_bo - abs(atom.get('charge', 0)) - eh)
                mass += impl_h * ATOMIC_MASSES['H']
        return mass

    def _ion_smiles(self, parser: SMILESParser, atom_indices: Set[int], neighbors: Dict[int, List[int]], bond_orders: Dict[Tuple[int, int], int]) -> str:
        return SMILESWriter.from_subgraph(parser, atom_indices, neighbors, bond_orders, sodiated=False)

    @staticmethod
    def _compute_formula(parser: SMILESParser, atom_indices: Set[int], neighbors: Dict[int, List[int]], bond_orders: Dict[Tuple[int, int], int]) -> Dict[str, int]:
        """Compute elemental formula (C,H,O,N,Na) from a set of atom indices."""
        formula: Dict[str, int] = {}
        for idx in atom_indices:
            atom = parser.atoms[idx]
            elem = atom['element']
            formula[elem] = formula.get(elem, 0) + 1
            eh = atom.get('explicit_h', 0)
            if eh:
                formula['H'] = formula.get('H', 0) + eh
            if elem in {'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I'}:
                total_bo = sum(bond_orders.get((idx, n), 0) for n in neighbors[idx])
                impl_h = max(0, VALENCES[elem] - total_bo - abs(atom.get('charge', 0)) - eh)
                if impl_h:
                    formula['H'] = formula.get('H', 0) + impl_h
        return dict(formula)

    @staticmethod
    def _formula_to_string(formula: Dict[str, int]) -> str:
        """Convert formula dict to compact string like CO2H6C51H98O6Na."""
        if not formula:
            return ''
        order = ['C', 'H', 'O', 'N', 'P', 'S', 'Na']
        parts = []
        for elem in order:
            n = formula.get(elem, 0)
            if n > 0:
                parts.append(f'{elem}{n}' if n > 1 else elem)
        return ''.join(parts)

    def _build_hfa_dehydrated_peak(
        self,
        fragment_key: str,
        hfa_info: Dict[str, Any],
        hfa_label: str,
        hfa_smiles: str,
        hfa_atom_set: Set[int],
    ) -> Dict[str, Any]:
        """Build one exact HFA-derived dehydrated sodium diagnostic ion."""
        additions = {
            "hfa_dehydrated_adduct_c2h2o": {"C": 2, "H": 2, "O": 1},
            "hfa_dehydrated_adduct_c3h4": {"C": 3, "H": 4, "O": 0},
            "hfa_dehydrated_sn2_c2h2": {"C": 2, "H": 2, "O": 0},
        }
        if fragment_key not in additions:
            raise ValueError(f"Unsupported HFA diagnostic fragment: {fragment_key}")

        base_formula = {
            "C": int(hfa_info["C"]),
            "H": int(hfa_info["H"]) - 2,
            "O": int(hfa_info["O"]) - 1,
            "Na": 1,
        }
        if any(count < 0 for count in base_formula.values()):
            raise ValueError(
                f"HFA dehydration produced a negative atom count for {hfa_label}: "
                f"{base_formula}"
            )
        addition = additions[fragment_key]
        formula_dict = {
            "C": base_formula["C"] + addition["C"],
            "H": base_formula["H"] + addition["H"],
            "O": base_formula["O"] + addition["O"],
            "Na": 1,
        }
        if any(count < 0 for count in formula_dict.values()):
            raise ValueError(
                f"HFA diagnostic fragment has a negative atom count for {hfa_label}: "
                f"{formula_dict}"
            )

        water_mass = 2 * ATOMIC_MASSES["H"] + ATOMIC_MASSES["O"]
        offset_key = str(FRAGMENT_DEFINITIONS[fragment_key]["offset"])
        exact_mz = (
            float(hfa_info["mass"])
            - water_mass
            + self.na_mass
            - ELECTRON_MASS
            + self.ead_offsets[offset_key]
        )
        peak = self._make_fragment_peak(
            fragment_key,
            exact_mz,
            label=hfa_label,
            smiles=hfa_smiles,
            atom_set=hfa_atom_set,
            structure_edit=f"hfa_dehydrated:{offset_key}",
            formula=self._formula_to_string(formula_dict),
        )
        peak.update(
            {
                "formula_dict": dict(formula_dict),
                "charge": 1,
                "adduct": "[M+Na]+",
                "source_role": "HFA",
                "hfa_name": hfa_label,
                "hfa_abbr": hfa_label,
                "hfa_carbon": int(hfa_info["C"]),
                "hfa_double_bonds": int(hfa_info["db"]),
                "exact_mz": exact_mz,
            }
        )
        return peak

    def _validate_fragment_consistency(self, peaks: List[Dict[str, Any]]) -> List[str]:
        """Validate elemental conservation and mass consistency for all fragments."""
        warnings = []
        for p in peaks:
            fd = p.get('formula_dict', {})
            if not fd:
                continue
            if any(v < 0 for v in fd.values()):
                warnings.append(f'Fragment {p.get("annotation", "?")} has negative atom counts in formula')
            # Cross-check: mass computed from formula vs peak mz (within 50 ppm tolerance)
            mass_f = sum(ATOMIC_MASSES.get(e, 0.0) * n for e, n in fd.items())
            if mass_f > 0:
                ppm = ppm_error(p['mz'], mass_f)
                if ppm > 50:
                    warnings.append(f'Fragment {p.get("annotation", "?")} mass mismatch: formula mass={mass_f:.4f}, peak mz={p["mz"]:.4f}, {ppm:.1f} ppm')
        return warnings

    def generate(self, smiles: str) -> List[Dict[str, Any]]:
        if not isinstance(smiles, str) or not smiles.strip(): raise ValueError("Input must be a non-empty SMILES string")

        organic_smiles, had_na_salt = self.strip_sodium_adduct(smiles)
        if not organic_smiles: raise ValueError("SMILES contains only sodium adduct, no organic structure")

        parser = SMILESParser(organic_smiles)
        if len(parser.atoms) < 10: raise ValueError("SMILES too short to be a valid lipid")

        neighbors: Dict[int, List[int]] = {i: [] for i in range(len(parser.atoms))}
        bond_orders: Dict[Tuple[int, int], int] = {}
        for ai, aj, bo in parser.bonds:
            neighbors[ai].append(aj); neighbors[aj].append(ai)
            bond_orders[(ai, aj)] = bo; bond_orders[(aj, ai)] = bo

        all_atoms = set(range(len(parser.atoms)))
        ion_smiles = lambda atoms: self._ion_smiles(parser, atoms, neighbors, bond_orders)

        ester_carbons = []
        for idx, atom in enumerate(parser.atoms):
            if atom['element'] == 'C':
                double_o, single_o = -1, -1
                for n in neighbors[idx]:
                    if parser.atoms[n]['element'] == 'O':
                        bo = bond_orders[(idx, n)]
                        if bo == 2: double_o = n
                        elif bo == 1: single_o = n
                if double_o != -1 and single_o != -1:
                    ester_carbons.append({'C': idx, 'O_double': double_o, 'O_single': single_o, 'C_attached': -1})

        for ec in ester_carbons:
            for n in neighbors[ec['O_single']]:
                if n != ec['C']:
                    ec['C_attached'] = n; break

        def get_component_atoms(start_idx: int, blocked_nodes: Set[int]) -> Set[int]:
            visited, queue = set([start_idx]), [start_idx]
            while queue:
                curr = queue.pop(0)
                for n in neighbors[curr]:
                    if n not in visited and n not in blocked_nodes:
                        visited.add(n)
                        queue.append(n)
            return visited

        glycerol_ester_groups, fahfa_branch_ester_group = [], None
        for ec in ester_carbons:
            c_att = ec['C_attached']
            if c_att == -1: continue
            is_glycerol = False
            for other_ec in ester_carbons:
                if other_ec['C'] != ec['C'] and other_ec['C_attached'] != -1 and other_ec['C_attached'] in neighbors[c_att]:
                    is_glycerol = True; break
            if is_glycerol: glycerol_ester_groups.append(ec)
            else: fahfa_branch_ester_group = ec

        if len(glycerol_ester_groups) != 3:
            glycerol_ester_groups = ester_carbons[:3]
            if len(ester_carbons) > 3: fahfa_branch_ester_group = ester_carbons[3]

        def calculate_fa_properties(atom_set: Set[int], is_hfa: bool = False, extra_H: int = 0) -> Dict[str, Any]:
            formula: Dict[str, int] = {}
            for idx in atom_set:
                atom = parser.atoms[idx]
                elem = atom['element']
                formula[elem] = formula.get(elem, 0) + 1
                formula['H'] = formula.get('H', 0) + atom.get('explicit_h', 0)
                if elem in {'C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I'}:
                    total_bo = sum(bond_orders[(idx, n)] for n in neighbors[idx])
                    impl_h = max(0, VALENCES[elem] - total_bo - abs(atom.get('charge', 0)) - atom.get('explicit_h', 0))
                    formula['H'] = formula.get('H', 0) + impl_h

            formula['H'] = formula.get('H', 0) + 1 + extra_H
            formula['O'] = formula.get('O', 0) + 1
            C, H, O = formula.get('C', 0), formula.get('H', 0), formula.get('O', 0)
            aliphatic_db = max(0, C + 1 - H // 2 - 1)
            mass = sum(ATOMIC_MASSES.get(e, 0.0) * n for e, n in formula.items())
            abbr = lookup_hfa_abbr(C, aliphatic_db) if is_hfa else lookup_fa_abbr(C, aliphatic_db)
            return {'C': C, 'H': H, 'O': O, 'db': aliphatic_db, 'mass': mass, 'abbr': abbr}

        chains_info = []
        for ec in glycerol_ester_groups:
            chain_atoms = get_component_atoms(ec['C'], {ec['O_single']})
            if fahfa_branch_ester_group and fahfa_branch_ester_group['C'] in chain_atoms:
                branch_atoms = get_component_atoms(fahfa_branch_ester_group['C'], {fahfa_branch_ester_group['O_single']})
                hfa_atoms = chain_atoms - branch_atoms
                branch_info = calculate_fa_properties(branch_atoms, is_hfa=False)
                hfa_info = calculate_fa_properties(hfa_atoms, is_hfa=True, extra_H=1)
                fahfa_whole_info = calculate_fa_properties(chain_atoms, is_hfa=False)
                ester_pos = self._hfa_branch_ester_position(parser, neighbors, hfa_atoms, ec['C'], fahfa_branch_ester_group['C_attached']) if fahfa_branch_ester_group else 0
                chains_info.append({
                    'is_fahfa': True, 'name': f"{branch_info['abbr']}{hfa_info['abbr']}", 'mass': fahfa_whole_info['mass'],
                    'branch': branch_info, 'hfa': hfa_info, 'ec': ec,
                    'formula': f"C{fahfa_whole_info['C']}H{fahfa_whole_info['H']}O{fahfa_whole_info['O']}",
                    'atom_set': chain_atoms, 'branch_atom_set': branch_atoms, 'ester_position': ester_pos,
                })
            else:
                fa_info = calculate_fa_properties(chain_atoms, is_hfa=False)
                chains_info.append({
                    'is_fahfa': False, 'name': fa_info['abbr'], 'mass': fa_info['mass'],
                    'info': fa_info, 'ec': ec, 'formula': f"C{fa_info['C']}H{fa_info['H']}O{fa_info['O']}", 'atom_set': chain_atoms,
                })

        gly_carbons = [ec['C_attached'] for ec in glycerol_ester_groups if ec['C_attached'] != -1]
        sn2_ec, sn1_ec, sn3_ec = None, None, None
        for ec in glycerol_ester_groups:
            c_att = ec['C_attached']
            if c_att == -1: continue
            gly_neighbors = [n for n in neighbors[c_att] if n in gly_carbons]
            if len(gly_neighbors) == 2: sn2_ec = ec
            else:
                if sn1_ec is None: sn1_ec = ec
                else: sn3_ec = ec

        for c in chains_info:
            if c['ec'] == sn1_ec: c['sn'] = 'sn1'
            elif c['ec'] == sn2_ec: c['sn'] = 'sn2'
            elif c['ec'] == sn3_ec: c['sn'] = 'sn3'
            else: c['sn'] = 'sn1'

        # Assign labels by chemical role, independent of the FAHFA sn position.
        for c in chains_info:
            if c['is_fahfa']:
                c['annot_label'] = format_subscript('FA3HFA')
                c['branch_label'] = format_subscript('FA3')
            else:
                c['annot_label'] = format_subscript('FA2') if c['sn'] == 'sn3' else format_subscript('FA1')

        neutral_mass = parser.get_exact_mass()
        precursor_mz = neutral_mass + (self.na_mass - ELECTRON_MASS)

        peaks = []
        peaks.append(self._make_fragment_peak('precursor', precursor_mz, smiles=organic_smiles, atom_set=all_atoms))

        fahfa_chain = next((c for c in chains_info if c['is_fahfa']), None)
        regular_chains = [c for c in chains_info if not c['is_fahfa']]

        for chain in chains_info:
            loss_mz = precursor_mz - chain['mass']
            sn, name = chain['sn'], chain.get('annot_label', chain['name'])
            remainder = all_atoms - chain['atom_set']
            frag_smiles = ion_smiles(remainder)
            if chain['is_fahfa']:
                peaks.append(self._make_fragment_peak('neutral_loss_fahfa', loss_mz, sn=sn, label=name, smiles=frag_smiles, atom_set=remainder))
            else:
                peaks.append(self._make_fragment_peak('neutral_loss', loss_mz, sn=sn, label=name, smiles=frag_smiles, atom_set=remainder))

        self._last_fahfa_sn = fahfa_chain['sn'] if fahfa_chain else None
        if fahfa_chain:
            br_name = fahfa_chain['branch_label']
            br_mass = fahfa_chain['branch']['mass']
            fa3_info = fahfa_chain['branch']
            fa3_c = int(fa3_info['C'])
            fa3_db = int(fa3_info['db'])
            fa3_label = f"{fa3_c}:{fa3_db}"
            fa3_protonated_mz = float(fa3_info['mass']) + PROTON_MASS
            fa3_ion_formula = {
                'C': fa3_c,
                'H': int(fa3_info['H']) + 1,
                'O': 2,
            }
            fa3_peak = self._make_fragment_peak(
                'fa3_protonated',
                fa3_protonated_mz,
                label=fa3_label,
                smiles='',
                atom_set=None,
                formula=self._formula_to_string(fa3_ion_formula),
            )
            fa3_peak.update(
                {
                    'formula_dict': dict(fa3_ion_formula),
                    'charge': 1,
                    'adduct': '[M+H]+',
                    'source_role': 'FA3',
                    'fa3_name': fa3_label,
                    'fa3_abbr': str(fa3_info.get('abbr', '')),
                    'fa3_carbon': fa3_c,
                    'fa3_double_bonds': fa3_db,
                    'exact_mz': fa3_protonated_mz,
                }
            )
            peaks.append(fa3_peak)
            branch_atoms = fahfa_chain['branch_atom_set']
            fahfa_atoms = fahfa_chain['atom_set']
            fahfa_ion_smiles = ion_smiles(fahfa_atoms)

            peaks.append(self._make_fragment_peak(
                'neutral_loss_branch', precursor_mz - br_mass, branch=br_name,
                smiles=ion_smiles(all_atoms - branch_atoms), atom_set=all_atoms - branch_atoms,
            ))
            for rc in regular_chains:
                comb_atoms = all_atoms - branch_atoms - rc['atom_set']
                peaks.append(self._make_fragment_peak(
                    'neutral_loss_combined', precursor_mz - br_mass - rc['mass'],
                    branch=br_name, label=rc.get("annot_label", rc["name"]),
                    smiles=ion_smiles(comb_atoms), atom_set=comb_atoms,
                ))

            mz_fn = fahfa_chain['mass'] + (self.na_mass - ELECTRON_MASS)
            peaks.append(self._make_fragment_peak('fahfa_precursor', mz_fn, smiles=fahfa_ion_smiles, atom_set=fahfa_atoms))

            ead_series = ['ead_diagnostic']
            if fahfa_chain['sn'] != 'sn2':
                ead_series.extend(['ead_adduct_c2h2o', 'ead_adduct_c3h4', 'ead_adduct_ch_radical'])
            else:
                ead_series.append('ead_sn2_diagnostic')

            for fragment_key in ead_series:
                offset_key = FRAGMENT_DEFINITIONS[fragment_key].get('offset', fragment_key)
                peaks.append(self._make_fragment_peak(
                    fragment_key, mz_fn + self.ead_offsets[offset_key],
                    smiles=fahfa_ion_smiles, atom_set=fahfa_atoms, structure_edit=f'ead:{offset_key}',
                ))

            hfa_info = fahfa_chain['hfa']
            hfa_label = str(hfa_info.get('abbr', '')).strip() or (
                f"{int(hfa_info['C'])}:{int(hfa_info['db'])}-OH"
            )
            hfa_atom_set = set(fahfa_chain['atom_set']) - set(
                fahfa_chain['branch_atom_set']
            )
            hfa_smiles = ion_smiles(hfa_atom_set) if hfa_atom_set else ''
            if fahfa_chain['sn'] != 'sn2':
                hfa_fragment_keys = (
                    'hfa_dehydrated_adduct_c2h2o',
                    'hfa_dehydrated_adduct_c3h4',
                )
            else:
                hfa_fragment_keys = ('hfa_dehydrated_sn2_c2h2',)
            for fragment_key in hfa_fragment_keys:
                peaks.append(
                    self._build_hfa_dehydrated_peak(
                        fragment_key,
                        hfa_info,
                        hfa_label,
                        hfa_smiles,
                        hfa_atom_set,
                    )
                )

            if fahfa_chain['sn'] == 'sn1':
                sn2_chain = next((c for c in regular_chains if c.get('sn') == 'sn2'), None)
                if sn2_chain:
                    sn2_formula = {
                        'C': sn2_chain['info']['C'] + 2,
                        'H': sn2_chain['info']['H'] + 2,
                        'O': sn2_chain['info']['O'],
                        'Na': 1,
                    }
                    peaks.append(self._make_fragment_peak(
                        'ead_sn2',
                        sn2_chain['mass'] + self.ead_offsets['c2h2'] + (self.na_mass - ELECTRON_MASS),
                        smiles=ion_smiles(sn2_chain['atom_set']), atom_set=sn2_chain['atom_set'],
                        structure_edit='ead:c2h2', formula=self._formula_to_string(sn2_formula),
                    ))

            if fahfa_branch_ester_group:
                c_att = fahfa_branch_ester_group['C_attached']
                ec_hfa = fahfa_chain['ec']
                c_hfa_carbonyl = ec_hfa['C']

                proximal_piece = get_component_atoms(c_hfa_carbonyl, {c_att, ec_hfa['O_single']})
                whole_hfa_and_branch = get_component_atoms(c_hfa_carbonyl, {ec_hfa['O_single']})
                branch_atoms_ester = get_component_atoms(fahfa_branch_ester_group['C'], {fahfa_branch_ester_group['O_single']})
                hfa_atoms = whole_hfa_and_branch - branch_atoms_ester
                distal_piece = hfa_atoms - proximal_piece - {c_att, fahfa_branch_ester_group['O_single']}

                distal_mass = self._get_fragment_mass(parser, distal_piece, neighbors, bond_orders)
                branch_loss_mass = self._get_fragment_mass(parser, branch_atoms_ester, neighbors, bond_orders)

                ester_x = max(0, fahfa_chain['hfa']['C'] - fahfa_chain.get('ester_position', 0))
                ester_high_h = max(0, 2 * ester_x)
                ester_low_c = ester_x + 1
                ester_low_h = max(0, 2 * ester_x + 1)

                frag1_mz = precursor_mz - distal_mass - branch_loss_mass
                cho_mass = ATOMIC_MASSES['C'] + ATOMIC_MASSES['H'] + ATOMIC_MASSES['O']
                frag2_mz = frag1_mz - cho_mass

                frag1_atoms = all_atoms - distal_piece - branch_atoms_ester
                peaks.append(self._make_fragment_peak(
                    'ead_diagnostic_ester_high', frag1_mz, x=ester_x, h=ester_high_h,
                    smiles=ion_smiles(frag1_atoms), atom_set=frag1_atoms,
                ))
                peaks.append(self._make_fragment_peak(
                    'ead_diagnostic_ester_low', frag2_mz, z=ester_low_c, w=ester_low_h,
                    smiles=ion_smiles(frag1_atoms), atom_set=frag1_atoms, structure_edit='ester:minus_cho',
                ))

        self._last_msp_name = self._build_msp_name(chains_info)
        self._last_common_name = self._build_common_name(chains_info)
        peaks.sort(key=lambda p: p['mz'])
        # Compute elemental formula for every peak from its atom_set
        for p in peaks:
            if 'atom_set' in p and p['atom_set'] and 'formula' not in p:
                raw = self._compute_formula(parser, set(p['atom_set']), neighbors, bond_orders)
                raw['Na'] = raw.get('Na', 0) + 1
                p['formula_dict'] = dict(raw)
                p['formula'] = self._formula_to_string(raw)
        # Validate fragment consistency
        val_warnings = self._validate_fragment_consistency(peaks)
        if val_warnings:
            for w in val_warnings:
                import warnings as _warn_mod
                _warn_mod.warn(f'[TG-FAHFA Fragment Validation] {w}')
        ester_position = int(fahfa_chain.get('ester_position', 0)) if fahfa_chain else 0
        for peak in peaks:
            mapping = HDCS_FRAGMENT_MAP.get(str(peak.get('fragment_key', '')))
            if mapping:
                group = mapping.get('group')
                if mapping.get('group_from_ester_position') and ester_position:
                    group = f"C{ester_position}"
                peak['level'] = int(mapping['level'])
                peak['levels'] = [int(mapping['level'])]
                peak['group'] = group
                peak['groups'] = [group] if group is not None else []
            else:
                peak['levels'] = []
                peak['groups'] = []
        self.last_raw_peaks = [dict(peak) for peak in peaks]
        self.last_display_peaks = self._deduplicate(self.last_raw_peaks)
        return [dict(peak) for peak in self.last_display_peaks]

    @staticmethod
    def _make_fragment_peak(fragment_key: str, mz: float, sn: Optional[str] = None, smiles: str = '', atom_set: Optional[Set[int]] = None, structure_edit: Optional[str] = None, formula: Optional[str] = None, **annotation_values) -> Dict[str, Any]:
        meta = FRAGMENT_DEFINITIONS[fragment_key]
        return TGFAHFASpectrumGenerator._make_peak(
            mz,
            round(fragment_intensity(fragment_key, sn=sn), 1),
            render_fragment_annotation(fragment_key, **annotation_values),
            meta["type"],
            bool(meta.get("is_key", False)),
            fragment_key=fragment_key,
            smiles=smiles,
            atom_set=atom_set,
            structure_edit=structure_edit,
            formula=formula,
        )

    @staticmethod
    def _make_peak(mz: float, intensity: float, annotation: str, peak_type: str, is_key: bool = False, fragment_key: Optional[str] = None, smiles: str = '', atom_set: Optional[Set[int]] = None, structure_edit: Optional[str] = None, formula: Optional[str] = None) -> Dict[str, Any]:
        peak: Dict[str, Any] = {
            'mz': round(mz, 4), 'intensity': intensity, 'annotation': annotation,
            'type': peak_type, 'is_key_ion': is_key, 'smiles': smiles,
            'fragment_key': fragment_key,
            'fragment_keys': [fragment_key] if fragment_key else [],
        }
        if atom_set is not None: peak['atom_set'] = sorted(atom_set)
        if structure_edit: peak['structure_edit'] = structure_edit
        if formula: peak['formula'] = formula
        return peak

    @staticmethod
    def _deduplicate(peaks: List[Dict[str, Any]], tol_ppm: float = 5.0) -> List[Dict[str, Any]]:
        if not peaks: return peaks
        def merge_str(s1: str, s2: str) -> str:
            parts = []
            for part in s1.split('/') + s2.split('/'):
                part = part.strip()
                if part and part not in parts: parts.append(part)
            return ' / '.join(parts)

        def merge_list(values1: Iterable[Any], values2: Iterable[Any]) -> List[Any]:
            output: List[Any] = []
            for value in list(values1) + list(values2):
                if value is not None and value not in output:
                    output.append(value)
            return output

        merged = [dict(peaks[0])]
        for p in peaks[1:]:
            last = merged[-1]
            diff_ppm = abs(p['mz'] - last['mz']) / last['mz'] * 1e6
            if diff_ppm <= tol_ppm:
                new_annot = merge_str(last['annotation'], p['annotation'])
                new_type = merge_str(last['type'], p['type'])
                new_smiles = merge_str(last.get('smiles', ''), p.get('smiles', ''))
                new_formula = merge_str(last.get('formula', ''), p.get('formula', ''))
                new_fragment_keys = merge_list(last.get('fragment_keys', []), p.get('fragment_keys', []))
                new_levels = sorted(merge_list(last.get('levels', []), p.get('levels', [])))
                new_groups = merge_list(last.get('groups', []), p.get('groups', []))
                is_key = last.get('is_key_ion') or p.get('is_key_ion')
                if p.get('is_key_ion') and not last.get('is_key_ion'): best_p = dict(p)
                elif not p.get('is_key_ion') and last.get('is_key_ion'): best_p = dict(last)
                else: best_p = dict(p) if p['intensity'] > last['intensity'] else dict(last)
                best_p.update({
                    'annotation': new_annot,
                    'type': new_type,
                    'smiles': new_smiles,
                    'formula': new_formula,
                    'is_key_ion': is_key,
                    'fragment_keys': new_fragment_keys,
                    'levels': new_levels,
                    'groups': new_groups,
                })
                if 'formula_dict' in last and 'formula_dict' in p:
                    best_p['formula_dict'] = last['formula_dict'] if last.get('is_key_ion', False) else p.get('formula_dict', {})
                merged[-1] = best_p
            else:
                merged.append(dict(p))
        return merged

    @staticmethod
    def deduplicate_for_display(
        raw_peaks: List[Dict[str, Any]], tol_ppm: float = 5.0
    ) -> List[Dict[str, Any]]:
        return TGFAHFASpectrumGenerator._deduplicate(raw_peaks, tol_ppm)

#############################################################################
#  GUI APPLICATION — Tkinter-based spectral analysis platform
#############################################################################
def compact_member_keys(
    member_keys: Sequence[str],
    visible_limit: int = 3,
) -> str:
    """Keep the Current Candidate panel compact while Evidence retains all IDs."""
    keys = [str(key) for key in member_keys]
    if not keys:
        return "None"
    if len(keys) <= visible_limit:
        return ", ".join(keys)
    return f"{', '.join(keys[:visible_limit])} — (+{len(keys) - visible_limit} more)"


class SpectrumNavigationToolbar(NavigationToolbar2Tk):
    """Navigation toolbar whose Home button restores the exact full spectrum."""

    def __init__(self, canvas: Any, window: Any, reset_callback: Any, **kwargs: Any):
        self._reset_callback = reset_callback
        super().__init__(canvas, window, **kwargs)

    def home(self, *args: Any) -> None:
        self._reset_callback()


class IntegratedApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} {APP_VERSION_TAG}")
        self.geometry("1600x950")
        self.minsize(1024, 768)
        
        self.raw_spec_df = pd.DataFrame()
        self.current_spec_df = pd.DataFrame()
        self.spectrum_filter_stats: Dict[str, Any] = {}
        self.parent_mz_exp = 0.0
        self.candidates_df = pd.DataFrame()
        self._candidate_scores: Dict[int, float] = {}
        self._candidate_ranks: Dict[int, int] = {}
        self.predicted_peaks = []
        self.raw_predicted_peaks = []
        self.candidate_raw_peaks: Dict[str, List[Dict[str, Any]]] = {}
        self.candidate_display_peaks: Dict[str, List[Dict[str, Any]]] = {}
        self.hdcs_candidates: Dict[str, Dict[str, Any]] = {}
        self.hdcs_results: Dict[str, Dict[str, Any]] = {}
        self.display_candidate_rows: List[Dict[str, Any]] = []
        self.display_row_by_id: Dict[str, Dict[str, Any]] = {}
        self.hdcs_display_results: Dict[str, Dict[str, Any]] = {}
        self.candidate_index_by_key: Dict[str, int] = {}
        self.hdcs_global_result: Dict[str, Any] = {}
        self.candidate_warnings: List[str] = []
        self.preview_raw_df = pd.DataFrame()
        self.input_column_mapping: Dict[str, Optional[str]] = {
            "mz": None, "intensity": None,
        }
        self.input_mapping_confirmed = False
        self.input_mapping_file = ""
        self.input_metadata: Dict[str, Any] = {}
        self.run_context: Dict[str, Any] = {}
        self.analysis_completed = False
        self.analysis_start_time = ""
        self.analysis_completion_time = ""
        self._preview_window: Optional[tk.Toplevel] = None
        self.mz_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready.")
        self.effective_cutoff_var = tk.StringVar(value="Effective intensity cutoff: 75")
        self._hdcs_info_resize_after_id: Optional[str] = None
        self._hdcs_summary_resize_after_id: Optional[str] = None
        self.hdcs_param_vars: Dict[str, Any] = {
            "ppm": tk.StringVar(value="20"),
            "noise": tk.StringVar(value="15"),
            "sn_threshold": tk.StringVar(value="5"),
            "l1": tk.StringVar(value="150"),
            "l2": tk.StringVar(value="150"),
            "l3": tk.StringVar(value="150"),
            "l4": tk.StringVar(value="150"),
            "nnls": tk.BooleanVar(value=True),
            "mixture": tk.BooleanVar(value=True),
        }
        self._last_completed_status = ""
        self._spectrum_home_xlim: Tuple[float, float] = (100.0, 1500.0)
        self._spectrum_home_ylim: Tuple[float, float] = (-100.0, 100.0)
        self.generator = TGFAHFASpectrumGenerator()
        
        self._build_styles()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        
    def _build_styles(self):
        style = ttk.Style(self)
        if "clam" in style.theme_names(): style.theme_use("clam")
        style.configure("Treeview", rowheight=28)

    def _on_close(self) -> None:
        """Cancel pending responsive-layout callbacks before closing Tk."""
        if self._preview_window is not None:
            self._close_preview_window()
        for attribute in (
            "_hdcs_info_resize_after_id",
            "_hdcs_summary_resize_after_id",
        ):
            callback_id = getattr(self, attribute, None)
            if callback_id is not None:
                try:
                    self.after_cancel(callback_id)
                except tk.TclError:
                    pass
                setattr(self, attribute, None)
        if hasattr(self, "canvas"):
            idle_draw_id = getattr(self.canvas, "_idle_draw_id", None)
            if idle_draw_id is not None:
                try:
                    self.canvas._tkcanvas.after_cancel(idle_draw_id)
                except tk.TclError:
                    pass
                self.canvas._idle_draw_id = None
        self.destroy()

    def _bind_responsive_wrap(
        self,
        label: tk.Widget,
        container: tk.Widget,
        minimum: int = 120,
        horizontal_padding: int = 20,
    ) -> None:
        """Keep an explanatory label's wraplength synchronized with its container."""
        def update_wrap(event: Any) -> None:
            try:
                available = max(int(minimum), int(event.width) - int(horizontal_padding))
                label.configure(wraplength=available)
            except tk.TclError:
                return

        container.bind("<Configure>", update_wrap, add="+")

    def _build_ui(self):
        self.paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ================= Left Panel =================
        left_frame = ttk.Frame(self.paned)
        self.paned.add(left_frame, weight=2)

        # Fill the complete left panel with a 40:60 split between the
        # input/parameter section and the candidate section.
        left_layout = ttk.Frame(left_frame)
        left_layout.pack(fill=tk.BOTH, expand=True)
        left_layout.rowconfigure(0, weight=4, uniform="left_panel_split")
        left_layout.rowconfigure(1, weight=6, uniform="left_panel_split")
        left_layout.columnconfigure(0, weight=1)
        left_top_frame = ttk.Frame(left_layout)
        left_top_frame.grid(row=0, column=0, sticky="nsew")
        left_candidate_frame = ttk.Frame(left_layout)
        left_candidate_frame.grid(row=1, column=0, sticky="nsew")
        
        lbl_step1 = tk.Label(left_top_frame, text="Data Input & Prediction List", bg="#e0e0e0", anchor="w")
        lbl_step1.pack(fill=tk.X, pady=(0, 5))
        
        file_frame = ttk.Frame(left_top_frame)
        file_frame.pack(fill=tk.X, pady=5)
        ttk.Label(file_frame, text="Data File (.csv):").pack(side=tk.LEFT)
        self.file_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_var).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        ttk.Button(file_frame, text="Browse...", command=self._on_browse).pack(side=tk.LEFT)
        ttk.Button(
            file_frame, text="Preview Data", command=self._open_data_preview
        ).pack(side=tk.LEFT, padx=(4, 0))
        self.file_var.trace_add("write", self._on_file_path_changed)
        parameters = ttk.LabelFrame(left_top_frame, text="HDCS Parameters", padding=6)
        parameters.pack(fill=tk.BOTH, expand=True, pady=(2, 5))
        parameter_fields = [
            ("ppm tolerance", "ppm"),
            ("Noise", "noise"),
            ("S/N threshold", "sn_threshold"),
            ("L1 threshold", "l1"),
            ("L2 threshold", "l2"),
            ("L3 threshold", "l3"),
            ("L4 threshold", "l4"),
        ]
        for row_index, (label, key) in enumerate(parameter_fields):
            ttk.Label(parameters, text=label + ":").grid(
                row=row_index, column=0, sticky="e", padx=(0, 6), pady=1
            )
            entry = ttk.Entry(parameters, textvariable=self.hdcs_param_vars[key], width=10)
            entry.grid(
                row=row_index, column=1, sticky="ew", pady=1
            )
            if key in {"noise", "sn_threshold"}:
                entry.bind("<KeyRelease>", self._update_effective_cutoff_preview)
                entry.bind("<FocusOut>", self._update_effective_cutoff_preview)
        parameters.columnconfigure(1, weight=1, minsize=120)
        for row_index in range(len(parameter_fields)):
            parameters.rowconfigure(row_index, weight=1, uniform="parameter_row")

        action_frame = ttk.Frame(left_top_frame)
        action_frame.pack(fill=tk.X, pady=(0, 5))
        self.run_button = ttk.Button(
            action_frame, text="▶ Run", command=self._on_run
        )
        self.run_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.export_button = ttk.Button(
            action_frame,
            text="Export Results",
            command=self._export_analysis_results,
            state="disabled",
        )
        self.export_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        main_status_label = ttk.Label(
            left_top_frame,
            textvariable=self.status_var,
            foreground="#334155",
            anchor="w",
            justify="left",
        )
        main_status_label.pack(fill=tk.X, pady=(0, 5))
        self._bind_responsive_wrap(main_status_label, left_top_frame, minimum=140)
        ttk.Label(
            left_candidate_frame,
            text="Candidate Isomer Results:",
        ).pack(anchor="w", pady=(8, 5))
        
        # UI Requirement 2: Add Candidate Count Label
        # Candidate count (left) and Sort (right) on same row
        candidate_sort_frame = ttk.Frame(left_candidate_frame)
        candidate_sort_frame.pack(fill=tk.X, pady=(0, 5))
        self.lbl_candidate_count = ttk.Label(
            candidate_sort_frame,
            text="Displayed Candidates: 0",
            font=("Arial", 10, "bold"),
        )
        self.lbl_candidate_count.pack(side=tk.LEFT)
        # Sort: right-aligned
        self.sort_var = tk.StringVar(value="Sort by Theory NNLS")
        self.sort_combo = ttk.Combobox(
            candidate_sort_frame,
            textvariable=self.sort_var,
            values=["Sort by Theory NNLS", "Original Order", "Sort by HDCS Score", "Sort by Rank"],
            state="readonly",
            width=20,
        )
        self.sort_combo.pack(side=tk.RIGHT)
        ttk.Label(candidate_sort_frame, text="Sort:").pack(side=tk.RIGHT, padx=(0, 5))
        self.sort_combo.bind("<<ComboboxSelected>>", self._on_sort_changed)

        tree_left_frame = ttk.Frame(left_candidate_frame)
        tree_left_frame.pack(fill=tk.BOTH, expand=True)
        columns_left = ("structure", "sn1", "sn2", "sn3", "rank", "score", "nnls")
        self.tree_left = ttk.Treeview(
            tree_left_frame,
            columns=columns_left,
            show="headings",
            selectmode="browse",
        )
        self.tree_left.heading("structure", text="TG-FAHFA Candidate Structure")
        self.tree_left.heading("sn1", text="sn-1")
        self.tree_left.heading("sn2", text="sn-2")
        self.tree_left.heading("sn3", text="sn-3")
        self.tree_left.heading("rank", text="Rank")
        self.tree_left.heading("score", text="HDCS Score")
        self.tree_left.heading("nnls", text="Theory NNLS")
        self._configure_displayed_candidate_columns()
        self.tree_left.bind("<<TreeviewSelect>>", self._on_candidate_select)

        tree_left_vsb = ttk.Scrollbar(
            tree_left_frame, orient="vertical", command=self.tree_left.yview
        )
        tree_left_hsb = ttk.Scrollbar(
            tree_left_frame, orient="horizontal", command=self.tree_left.xview
        )
        self.tree_left_vsb = tree_left_vsb
        self.tree_left_hsb = tree_left_hsb
        self.tree_left.configure(
            yscrollcommand=tree_left_vsb.set,
            xscrollcommand=tree_left_hsb.set,
        )
        self.tree_left.grid(row=0, column=0, sticky="nsew")
        tree_left_vsb.grid(row=0, column=1, sticky="ns")
        tree_left_hsb.grid(row=1, column=0, sticky="ew")
        tree_left_frame.rowconfigure(0, weight=1)
        tree_left_frame.columnconfigure(0, weight=1)
        
        # ================= Right Panel =================
        right_frame = ttk.Frame(self.paned)
        self.paned.add(right_frame, weight=3)

        self.right_notebook = ttk.Notebook(right_frame)
        self.right_notebook.pack(fill=tk.BOTH, expand=True)
        self.spectrum_tab = ttk.Frame(self.right_notebook)
        self.hdcs_tab = ttk.Frame(self.right_notebook)
        self.right_notebook.add(self.spectrum_tab, text="Spectrum Prediction")
        self.right_notebook.add(self.hdcs_tab, text="HDCS Analysis Result")
        
        lbl_step5 = tk.Label(self.spectrum_tab, text="Predicted EAciD-MS/MS Spectrum", bg="#e0e0e0", anchor="w")
        lbl_step5.pack(fill=tk.X)
        
        right_top_frame = ttk.Frame(self.spectrum_tab)
        right_top_frame.pack(fill=tk.BOTH, expand=True)
        
        smiles_frame = ttk.Frame(right_top_frame)
        smiles_frame.pack(fill=tk.X, pady=5)
        ttk.Label(smiles_frame, text="SMILES:").grid(
            row=0, column=0, sticky="w", padx=(0, 4), pady=1
        )
        self.smiles_var = tk.StringVar()
        ttk.Entry(smiles_frame, textvariable=self.smiles_var).grid(
            row=0, column=1, sticky="ew", pady=1
        )
        
        ttk.Label(smiles_frame, text="Common Name:").grid(
            row=1, column=0, sticky="w", padx=(0, 4), pady=1
        )
        self.common_name_var = tk.StringVar()
        ttk.Entry(
            smiles_frame, textvariable=self.common_name_var, state="readonly"
        ).grid(row=1, column=1, sticky="ew", pady=1)
        smiles_frame.columnconfigure(1, weight=1, minsize=120)
        
        tree_right_frame = ttk.Frame(right_top_frame)
        tree_right_frame.pack(fill=tk.BOTH, expand=True)
        columns_right = ("mz", "intensity", "key", "type", "annotation")
        self.tree_right = ttk.Treeview(
            tree_right_frame,
            columns=columns_right,
            show="headings",
            selectmode="none",
        )
        self.tree_right.heading("mz", text="m/z")
        self.tree_right.heading("intensity", text="Int %")
        self.tree_right.heading("key", text="Key")
        self.tree_right.heading("type", text="Level")
        self.tree_right.heading("annotation", text="Annotation")
        
        self.tree_right.column("mz", width=100, minwidth=80, anchor="e", stretch=False)
        self.tree_right.column("intensity", width=80, minwidth=65, anchor="e", stretch=False)
        self.tree_right.column("key", width=60, minwidth=50, anchor="center", stretch=False)
        self.tree_right.column("type", width=220, minwidth=120, anchor="w", stretch=True)
        self.tree_right.column("annotation", width=350, minwidth=180, anchor="w", stretch=True)
        
        self.tree_right.tag_configure("L1", foreground=SPECTRUM_LEVEL_COLORS["L1"], font=("", 10))
        self.tree_right.tag_configure("L2", foreground=SPECTRUM_LEVEL_COLORS["L2"], font=("", 10, "bold"))
        self.tree_right.tag_configure("L3", foreground=SPECTRUM_LEVEL_COLORS["L3"], font=("", 10, "bold"))
        self.tree_right.tag_configure("L4", foreground=SPECTRUM_LEVEL_COLORS["L4"], font=("", 10, "bold"))
        self.tree_right.tag_configure("no_level", foreground=SPECTRUM_LEVEL_COLORS["no_level"], font=("", 10))
        
        vsb = ttk.Scrollbar(
            tree_right_frame, orient="vertical", command=self.tree_right.yview
        )
        hsb = ttk.Scrollbar(
            tree_right_frame, orient="horizontal", command=self.tree_right.xview
        )
        self.tree_right.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self.tree_right.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_right_frame.rowconfigure(0, weight=1)
        tree_right_frame.columnconfigure(0, weight=1)
        
        # ================= Plot Panel =================
        right_bottom_frame = ttk.Frame(self.spectrum_tab)
        right_bottom_frame.pack(fill=tk.BOTH, expand=True)
        
        # Use explicit margins instead of constrained layout.  Unclipped peak
        # labels near a high-m/z precursor can otherwise make constrained
        # layout shrink the axes and leave a large blank area on the right.
        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.fig.subplots_adjust(left=0.06, right=0.99, top=0.98, bottom=0.10)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_bottom_frame)
        self.spectrum_toolbar = SpectrumNavigationToolbar(
            self.canvas,
            right_bottom_frame,
            reset_callback=self._reset_spectrum_view,
            pack_toolbar=False,
        )
        self.spectrum_toolbar.update()
        self.spectrum_toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect("scroll_event", self._on_spectrum_scroll)
        self._build_hdcs_tab()
        self._init_plot()
        
    def _build_hdcs_tab(self):
        container = ttk.Frame(self.hdcs_tab, padding=6)
        container.pack(fill=tk.BOTH, expand=True)
        container.columnconfigure(0, weight=1)
        # The notebook tab header consumes part of the right-side height, so a
        # 38:62 content split visually aligns the diagnostic table with the
        # left panel's 40:60 parameter/candidate boundary.
        container.rowconfigure(0, weight=38, uniform="hdcs_vertical_split")
        container.rowconfigure(1, weight=62, uniform="hdcs_vertical_split")

        top = ttk.Frame(container)
        top.grid(row=0, column=0, sticky="nsew")
        info_frame = ttk.LabelFrame(top, text="Current Candidate", padding=6)
        info_frame.pack(fill=tk.BOTH, expand=True)
        self.hdcs_info_vars: Dict[str, tk.StringVar] = {}
        self.hdcs_info_value_labels: Dict[str, ttk.Label] = {}
        self.hdcs_info_field_widgets: List[Dict[str, Any]] = []
        self.hdcs_position_notice_var = tk.StringVar(value="")
        self.hdcs_position_notice_label = tk.Label(
            info_frame,
            textvariable=self.hdcs_position_notice_var,
            anchor="w",
            justify="left",
            padx=8,
            pady=5,
            font=("Arial", 10, "bold"),
            bg="#E8F5E9",
            fg="#166534",
        )
        self.hdcs_position_notice_label.grid(
            row=0, column=0, columnspan=2, sticky="ew", pady=(0, 5)
        )
        info_fields = [
            ("Candidate Name", "candidate_name"),
            ("TG-FAHFA Structure", "structure"),
            ("Theoretical Precursor m/z", "theoretical_precursor"),
            ("Experimental Precursor m/z", "experimental_precursor"),
            ("Mass Error (ppm)", "mass_error_ppm"),
            ("FAHFA sn-position", "sn_position"),
            ("FAHFA ester bond position", "ester_bond_position"),
        ]
        for row_index, (label, key) in enumerate(info_fields, start=1):
            name_label = ttk.Label(
                info_frame, text=label + ":", anchor="e", justify="right"
            )
            name_label.grid(
                row=row_index, column=0, sticky="e", padx=(0, 10), pady=1
            )
            variable = tk.StringVar(value="")
            self.hdcs_info_vars[key] = variable
            value_label = ttk.Label(
                info_frame,
                textvariable=variable,
                justify="left",
                anchor="w",
            )
            value_label.grid(
                row=row_index, column=1, sticky="ew", padx=(0, 12), pady=1
            )
            self.hdcs_info_value_labels[key] = value_label
            self.hdcs_info_field_widgets.append(
                {
                    "label": name_label,
                    "value": value_label,
                    "key": key,
                }
            )
        info_frame.columnconfigure(0, weight=0, minsize=210)
        info_frame.columnconfigure(1, weight=1, minsize=160)
        for row_index in range(1, len(info_fields) + 1):
            info_frame.rowconfigure(row_index, weight=1, uniform="candidate_info_row")
        self.hdcs_info_frame = info_frame
        info_frame.bind("<Configure>", self._resize_hdcs_info_wrap)

        summary_frame = ttk.LabelFrame(top, text="HDCS Score Summary", padding=5)
        summary_frame.pack(fill=tk.X, pady=(5, 0))
        self.hdcs_summary_vars: Dict[str, tk.StringVar] = {}
        self.hdcs_summary_widgets: List[Dict[str, Any]] = []
        summary_fields = [
            ("WDIC", "wdic"), ("NEP", "nep"), ("Base Score", "base"),
            ("L4 Evidence", "l4"), ("Final HDCS Score", "final"), ("Rank", "rank"),
            ("Mixture Detected", "mixture"), ("NNLS Contribution", "nnls"),
            ("NNLS Fit Quality", "fit"),
        ]
        for column, (label, key) in enumerate(summary_fields):
            frame = ttk.Frame(summary_frame)
            frame.grid(row=0, column=column, sticky="nsew", padx=3)
            name_label = ttk.Label(
                frame,
                text=label,
                font=("Arial", 8, "bold"),
                anchor="center",
                justify="center",
            )
            name_label.pack(fill=tk.X)
            variable = tk.StringVar(value="")
            self.hdcs_summary_vars[key] = variable
            value_label = ttk.Label(
                frame, textvariable=variable, anchor="center", justify="center"
            )
            value_label.pack(fill=tk.X)
            self.hdcs_summary_widgets.append(
                {
                    "frame": frame,
                    "label": name_label,
                    "value": value_label,
                    "key": key,
                }
            )
            self._bind_responsive_wrap(name_label, frame, minimum=70, horizontal_padding=8)
            summary_frame.columnconfigure(column, weight=1)
        self.hdcs_summary_frame = summary_frame
        summary_frame.bind("<Configure>", self._on_hdcs_summary_resize)

        self.hdcs_diagnostic_frame = ttk.LabelFrame(
            container, text="HDCS Diagnostic Ions", padding=4
        )
        diagnostic_frame = self.hdcs_diagnostic_frame
        diagnostic_frame.grid(row=1, column=0, sticky="nsew", pady=(5, 0))
        columns = ("level", "fragment_key", "annotation", "theo", "exp", "ppm", "intensity", "threshold", "found", "group", "shared")
        self.hdcs_ion_tree = ttk.Treeview(diagnostic_frame, columns=columns, show="headings")
        headings = {
            "level": "Level", "fragment_key": "Fragment Key", "annotation": "Annotation",
            "theo": "Theo m/z", "exp": "Exp m/z", "ppm": "ppm", "intensity": "Intensity", "threshold": "Threshold",
            "found": "Found", "group": "Group", "shared": "Shared",
        }
        widths = {
            "level": 52, "fragment_key": 165, "annotation": 260, "theo": 90, "exp": 90,
            "ppm": 65, "intensity": 80, "threshold": 75, "found": 55, "group": 70, "shared": 60,
        }
        for column in columns:
            self.hdcs_ion_tree.heading(column, text=headings[column])
            self.hdcs_ion_tree.column(
                column,
                width=widths[column],
                minwidth=50,
                anchor="center",
                stretch=column in {"fragment_key", "annotation"},
            )
        self.hdcs_ion_tree.column(
            "annotation", minwidth=160, anchor="w", stretch=True
        )
        self.hdcs_ion_tree.column(
            "fragment_key", minwidth=110, anchor="center", stretch=True
        )
        self.hdcs_ion_tree.tag_configure("L1", foreground=SPECTRUM_LEVEL_COLORS["L1"])
        self.hdcs_ion_tree.tag_configure("L2", foreground=SPECTRUM_LEVEL_COLORS["L2"], font=("", 9, "bold"))
        self.hdcs_ion_tree.tag_configure("L3", foreground=SPECTRUM_LEVEL_COLORS["L3"], font=("", 9, "bold"))
        self.hdcs_ion_tree.tag_configure("L4", foreground=SPECTRUM_LEVEL_COLORS["L4"], font=("", 9, "bold"))
        ion_vsb = ttk.Scrollbar(diagnostic_frame, orient="vertical", command=self.hdcs_ion_tree.yview)
        ion_hsb = ttk.Scrollbar(diagnostic_frame, orient="horizontal", command=self.hdcs_ion_tree.xview)
        self.hdcs_ion_tree.configure(yscrollcommand=ion_vsb.set, xscrollcommand=ion_hsb.set)
        self.hdcs_ion_tree.grid(row=0, column=0, sticky="nsew")
        ion_vsb.grid(row=0, column=1, sticky="ns")
        ion_hsb.grid(row=1, column=0, sticky="ew")
        diagnostic_frame.rowconfigure(0, weight=1)
        diagnostic_frame.columnconfigure(0, weight=1)

    def _configure_displayed_candidate_columns(self) -> None:
        """Set resizable defaults once; refresh and sorting must not reset them."""
        column_options = {
            "structure": (210, 120),
            "sn1": (62, 45),
            "sn2": (62, 45),
            "sn3": (62, 45),
            "rank": (52, 40),
            "score": (88, 60),
            "nnls": (115, 75),
        }
        for column, (width, minwidth) in column_options.items():
            self.tree_left.column(
                column,
                width=width,
                minwidth=minwidth,
                anchor="center",
                stretch=True,
            )

    def _resize_hdcs_info_wrap(self, event=None) -> None:
        """Debounce Current Candidate wrapping and aligned-column updates."""
        if event is None:
            return
        width = max(int(getattr(event, "width", 0)), 1)
        if self._hdcs_info_resize_after_id is not None:
            try:
                self.after_cancel(self._hdcs_info_resize_after_id)
            except tk.TclError:
                return
        self._hdcs_info_resize_after_id = self.after(
            50,
            lambda current_width=width: self._apply_hdcs_info_layout(current_width),
        )

    def _apply_hdcs_info_layout(self, width: int) -> None:
        """Keep every Current Candidate field in one aligned vertical column."""
        self._hdcs_info_resize_after_id = None
        frame = self.hdcs_info_frame
        for column in range(4):
            frame.columnconfigure(column, weight=0, minsize=0)
        label_width = min(210, max(150, int(width) // 4))
        frame.columnconfigure(0, weight=0, minsize=label_width)
        frame.columnconfigure(1, weight=1, minsize=160)
        value_wrap = max(140, int(width) - label_width - 40)
        for index, record in enumerate(self.hdcs_info_field_widgets):
            record["label"].grid_forget()
            record["value"].grid_forget()
            record["label"].grid(
                row=index + 1,
                column=0,
                sticky="e",
                padx=(0, 10),
                pady=1,
            )
            record["value"].grid(
                row=index + 1,
                column=1,
                sticky="ew",
                padx=(0, 12),
                pady=1,
            )
            record["value"].configure(wraplength=value_wrap)
        self.hdcs_position_notice_label.grid_configure(columnspan=2)

    def _on_hdcs_summary_resize(self, event: Any) -> None:
        """Debounce responsive HDCS Score Summary rearrangement."""
        width = max(int(getattr(event, "width", 0)), 1)
        if self._hdcs_summary_resize_after_id is not None:
            try:
                self.after_cancel(self._hdcs_summary_resize_after_id)
            except tk.TclError:
                return
        self._hdcs_summary_resize_after_id = self.after(
            50,
            lambda current_width=width: self._apply_hdcs_summary_layout(current_width),
        )

    def _apply_hdcs_summary_layout(self, width: int) -> None:
        """Wrap HDCS score cards into stable rows as the right pane narrows."""
        self._hdcs_summary_resize_after_id = None
        frame = self.hdcs_summary_frame
        if int(width) >= 1000:
            columns = len(self.hdcs_summary_widgets)
        elif int(width) >= 700:
            columns = 5
        elif int(width) >= 450:
            columns = 3
        else:
            columns = 2
        for column in range(len(self.hdcs_summary_widgets)):
            frame.columnconfigure(column, weight=0, minsize=0)
        for column in range(columns):
            frame.columnconfigure(column, weight=1, minsize=70)
        for index, record in enumerate(self.hdcs_summary_widgets):
            record["frame"].grid_forget()
            row, column = divmod(index, columns)
            record["frame"].grid(
                row=row, column=column, sticky="nsew", padx=3, pady=2
            )

    def _update_effective_cutoff_preview(self, event=None) -> None:
        try:
            noise = float(self.hdcs_param_vars["noise"].get())
            sn_threshold = float(self.hdcs_param_vars["sn_threshold"].get())
            if (
                math.isfinite(noise) and noise > 0
                and math.isfinite(sn_threshold) and sn_threshold >= 0
            ):
                self.effective_cutoff_var.set(
                    f"Effective intensity cutoff: {noise * sn_threshold:g}"
                )
                return
        except (TypeError, ValueError):
            pass
        self.effective_cutoff_var.set("Effective intensity cutoff: invalid parameters")

    def _get_spectrum_filter_config(self) -> Tuple[float, float]:
        try:
            noise = float(self.hdcs_param_vars["noise"].get())
            sn_threshold = float(self.hdcs_param_vars["sn_threshold"].get())
        except (TypeError, ValueError) as exc:
            raise ValueError("Noise and S/N threshold must be valid numbers.") from exc
        if not math.isfinite(noise) or noise <= 0:
            raise ValueError("Noise must be greater than 0.")
        if not math.isfinite(sn_threshold) or sn_threshold < 0:
            raise ValueError("S/N threshold cannot be negative.")
        return noise, sn_threshold

    def _get_hdcs_config(self) -> HDCSConfig:
        try:
            config = HDCSConfig(
                ppm_tolerance=float(self.hdcs_param_vars["ppm"].get()),
                shared_ion_tolerance_ppm=float(self.hdcs_param_vars["ppm"].get()),
                l1_threshold=float(self.hdcs_param_vars["l1"].get()),
                l2_threshold=float(self.hdcs_param_vars["l2"].get()),
                l3_threshold=float(self.hdcs_param_vars["l3"].get()),
                l4_threshold=float(self.hdcs_param_vars["l4"].get()),
                nnls_min_final_score=TARGET_HDCS_SCORE,
                nnls_enabled=bool(self.hdcs_param_vars["nnls"].get()),
                mixture_detection_enabled=bool(self.hdcs_param_vars["mixture"].get()),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("HDCS parameters must be valid numbers") from exc
        if config.ppm_tolerance <= 0 or min(
            config.l1_threshold, config.l2_threshold, config.l3_threshold,
            config.l4_threshold,
        ) < 0:
            raise ValueError("ppm must be positive and thresholds cannot be negative")
        return config

    @staticmethod
    def _ester_bond_position_text(value: Any, empty: str = "Not confirmed") -> str:
        """Format a numeric/C-prefixed ester position as lipid-style ``9-O``."""
        if value in (None, ""):
            return empty
        text = str(value).strip()
        match = re.fullmatch(r"(?:C)?(\d+)(?:-O)?", text, flags=re.IGNORECASE)
        return f"{int(match.group(1))}-O" if match else text

    @staticmethod
    def _evidence_position_text(values: Any, empty: str = "None") -> str:
        """Format ester-position collections without exposing Python list syntax."""
        if values in (None, ""):
            return empty
        if not isinstance(values, (list, tuple, set)):
            values = [values]
        formatted: List[str] = []
        for value in values:
            if value in (None, ""):
                continue
            formatted.append(
                IntegratedApp._ester_bond_position_text(value, empty="")
            )
        return "; ".join(formatted) or empty

    @staticmethod
    def _evidence_unique_lines(values: Any) -> List[str]:
        """Return stable, non-empty, deduplicated warning/evidence strings."""
        if values in (None, ""):
            return []
        if isinstance(values, str):
            values = [values]
        elif not isinstance(values, (list, tuple, set)):
            values = [values]
        output: List[str] = []
        for value in values:
            text = str(value).strip()
            if text and text not in output:
                output.append(text)
        return output

    @staticmethod
    def _evidence_float_text(
        value: Any, digits: int = 2, suffix: str = ""
    ) -> str:
        """Format a finite evidence value, leaving missing values truly blank."""
        if value in (None, ""):
            return ""
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return ""
        if not math.isfinite(numeric):
            return ""
        return f"{numeric:.{digits}f}{suffix}"

    def _format_evidence_ion_line(
        self, ion: Dict[str, Any], mode: str
    ) -> str:
        """Render one diagnostic ion for matched, missing, or exclusive evidence."""
        levels = ion.get("levels") or [ion.get("level", 0)]
        level_text = fragment_levels_display(levels, str(ion.get("type", "")))
        fragment_keys = ion.get("fragment_keys") or [ion.get("fragment_key", "")]
        fragment_text = " / ".join(
            str(value) for value in fragment_keys if value not in (None, "")
        ) or str(ion.get("fragment_key", ""))
        annotation = str(ion.get("annotation", "")).strip() or "No annotation"
        theoretical_mz = ion.get("theo_mz", ion.get("theoretical_mz"))
        theo_text = self._evidence_float_text(theoretical_mz, 5)
        threshold_text = self._evidence_float_text(ion.get("threshold"), 2)
        parts = [level_text or "Unclassified", fragment_text, annotation]
        if theo_text:
            parts.append(f"Theo {theo_text}")
        if mode in {"matched", "exclusive"}:
            exp_text = self._evidence_float_text(ion.get("exp_mz"), 5)
            ppm_text = self._evidence_float_text(
                ion.get("ppm_err", ion.get("ppm_error")), 2
            )
            intensity_text = self._evidence_float_text(ion.get("intensity"), 2)
            if exp_text:
                parts.append(f"Exp {exp_text}")
            if ppm_text:
                parts.append(f"{ppm_text} ppm")
            if intensity_text:
                parts.append(f"Intensity {intensity_text}")
        if threshold_text:
            parts.append(f"Threshold {threshold_text}")
        if mode == "missing":
            groups = ion.get("groups") or ([ion.get("group")] if ion.get("group") else [])
            group_text = " / ".join(str(value) for value in groups if value)
            parts.append(f"Group {group_text or 'None'}")
            parts.append(f"Shared {'Yes' if ion.get('is_shared') else 'No'}")
        elif mode == "exclusive":
            group_text = str(ion.get("group") or "Position specific")
            parts.append(group_text if "specific" in group_text.lower() else f"{group_text}-specific")
        return " | ".join(parts)

    def _format_contradictory_evidence_line(self, item: Any) -> str:
        """Render contradictory evidence without leaking a raw dictionary repr."""
        if isinstance(item, str):
            return item
        if not isinstance(item, dict):
            return str(item)
        for key in ("message", "description", "evidence", "annotation"):
            if item.get(key):
                return str(item[key])
        parts = ["Opposing sn evidence"]
        if item.get("cluster_id"):
            parts.append(f"cluster {item['cluster_id']}")
        match = item.get("match")
        if match is not None:
            exp_mz = getattr(match, "exp_mz", None)
            ppm_error_value = getattr(match, "ppm_error", None)
            intensity = getattr(match, "intensity", None)
            if exp_mz is not None:
                parts.append(f"m/z {float(exp_mz):.5f}")
            if ppm_error_value is not None:
                parts.append(f"{float(ppm_error_value):.2f} ppm")
            if intensity is not None:
                parts.append(f"intensity {float(intensity):.2f}")
        for key, value in item.items():
            if key in {"cluster_id", "match", "message", "description", "evidence", "annotation"}:
                continue
            if isinstance(value, (str, int, float, bool)) and value not in (None, ""):
                parts.append(f"{key.replace('_', ' ')} {value}")
        return ": ".join([parts[0], "; ".join(parts[1:])]) if len(parts) > 1 else parts[0]

    def _build_candidate_evidence_content(
        self, result: Dict[str, Any], candidate: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build the single source of truth used by GUI and Excel Evidence text."""
        candidate_key = str(
            result.get("representative_candidate_key", result.get("candidate_key", ""))
        )
        member_keys = [
            str(key)
            for key in (
                result.get("display_member_candidate_keys") or [candidate_key]
            )
            if key not in (None, "")
        ]
        reported_position = result.get("reported_ester_position")
        is_unresolved = (
            result.get("display_type") == "unresolved_positions"
            or reported_position is None
        )
        supported_text = self._evidence_position_text(
            result.get("supported_ester_positions"), "None"
        )
        unresolved_text = self._evidence_position_text(
            result.get("unresolved_ester_positions"), "None"
        )
        family_supported_text = self._evidence_position_text(
            result.get("family_supported_ester_positions"), "None"
        )
        family_unsupported_text = self._evidence_position_text(
            result.get("family_unsupported_ester_positions"), "None"
        )
        reported_position_text = (
            "Not confirmed"
            if is_unresolved
            else self._ester_bond_position_text(reported_position)
        )
        display_type_text = (
            "Unresolved position group" if is_unresolved else "Confirmed position"
        )
        found_l4 = int(result.get("l4_position_specific_found_count", 0) or 0)
        total_l4 = int(result.get("l4_position_specific_total_count", 0) or 0)
        l4_evidence_text = (
            f"{found_l4}/{total_l4} position-specific L4 ions detected"
            + ("; ester position unresolved" if is_unresolved else "")
        )
        representation_note = (
            "Important: this display row represents an unresolved position group. "
            "Candidate identity, Theory NNLS contribution and unresolved-position "
            "metadata may summarize multiple underlying candidates, while the "
            "diagnostic-ion table and predicted spectrum correspond to the "
            "representative theoretical candidate."
            if is_unresolved
            else "The ester position is supported by position-specific L4 evidence."
        )

        detected_ions = list(result.get("detected_ions", {}).values())
        matched_ions = list(result.get("matched_ions") or [
            ion for ion in detected_ions if ion.get("found")
        ])
        missing_ions = list(result.get("missing_ions") or [
            ion for ion in detected_ions if not ion.get("found")
        ])
        exclusive_ions = list(result.get("exclusive_evidence") or [])
        contradictions = list(result.get("contradictory_evidence") or [])
        matched_lines = [
            self._format_evidence_ion_line(ion, "matched") for ion in matched_ions
        ]
        missing_lines = [
            self._format_evidence_ion_line(ion, "missing") for ion in missing_ions
        ]
        exclusive_lines = [
            self._format_evidence_ion_line(ion, "exclusive")
            for ion in exclusive_ions
        ]
        contradiction_lines = [
            self._format_contradictory_evidence_line(item)
            for item in contradictions
        ]
        mixture_detection = self.hdcs_global_result.get("mixture_detection", {})
        mixture_lines = self._evidence_unique_lines(
            mixture_detection.get("evidence", [])
        )
        if not mixture_lines:
            mixture_lines = ["No mixture evidence above thresholds."]

        global_warning_lines = self._evidence_unique_lines(
            self.hdcs_global_result.get("warnings", [])
        )
        candidate_warning_lines = self._evidence_unique_lines(
            [
                warning
                for warning in self.candidate_warnings
                if any(key and key in str(warning) for key in member_keys)
            ]
        )
        display_warning_values: List[Any] = []
        for warning_key in ("warning", "warnings", "fa3_protonated_warning"):
            warning_value = result.get(warning_key)
            if isinstance(warning_value, (list, tuple, set)):
                display_warning_values.extend(warning_value)
            elif warning_value not in (None, ""):
                display_warning_values.append(warning_value)
        display_warning_lines = self._evidence_unique_lines(display_warning_values)
        warning_lines = self._evidence_unique_lines(
            global_warning_lines + candidate_warning_lines + display_warning_lines
        )

        fa3_found = bool(result.get("fa3_protonated_found", False))
        fa3_theoretical_mz = result.get("fa3_protonated_theoretical_mz")
        fa3_experimental_mz = (
            result.get("fa3_protonated_exp_mz") if fa3_found else None
        )
        fa3_ppm_error = (
            result.get("fa3_protonated_ppm_error") if fa3_found else None
        )
        fa3_intensity = (
            result.get("fa3_protonated_intensity") if fa3_found else None
        )
        fa3_rank = result.get("fa3_protonated_rank") if fa3_found else None
        fa3_ratio = (
            result.get("fa3_protonated_ratio_to_strongest") if fa3_found else None
        )
        strongest_second_ratio = (
            result.get("fa3_protonated_strongest_to_second_ratio")
            if fa3_found else None
        )
        strongest_second_text = ""
        if strongest_second_ratio is not None:
            try:
                if math.isinf(float(strongest_second_ratio)):
                    strongest_second_text = "Only one FA3+H ion detected"
                else:
                    strongest_second_text = f"{float(strongest_second_ratio):.3f}"
            except (TypeError, ValueError):
                strongest_second_text = str(strongest_second_ratio)
        strongest_second_numeric: Optional[float] = None
        if strongest_second_ratio is not None:
            try:
                parsed_ratio = float(strongest_second_ratio)
                if math.isfinite(parsed_ratio):
                    strongest_second_numeric = parsed_ratio
            except (TypeError, ValueError):
                pass

        position_summary = (
            f"{representation_note}\n\n"
            f"Position family: {result.get('position_family', '')}\n"
            f"Position resolution status: {result.get('position_resolution_status', '')}\n"
            f"Reported ester position: {reported_position_text}\n"
            f"Supported ester positions: {supported_text}\n"
            f"Unresolved ester positions: {unresolved_text}\n"
            f"Display type: {display_type_text}\n"
            f"Member candidates: {'; '.join(member_keys) or 'None'}\n"
            f"L4 position evidence: {l4_evidence_text}\n"
            f"Reported common name: {result.get('reported_common_name', '')}\n"
            f"Reported lipid name: {result.get('reported_lipid_name', '')}\n"
            f"Theoretical candidate name: {result.get('theoretical_common_name', candidate.get('name', ''))}\n"
            f"FA3 chain: {result.get('fa3_name', candidate.get('fa3_name', ''))}\n"
            f"FA3+H theoretical m/z: {self._evidence_float_text(fa3_theoretical_mz, 5)}\n"
            f"FA3+H status: {'Found' if fa3_found else 'Missing'}\n"
            f"FA3+H experimental m/z: {self._evidence_float_text(fa3_experimental_mz, 5)}\n"
            f"FA3+H ppm error: {self._evidence_float_text(fa3_ppm_error, 2)}\n"
            f"FA3+H intensity: {self._evidence_float_text(fa3_intensity, 2)}\n"
            f"FA3+H threshold: {self._evidence_float_text(result.get('fa3_protonated_threshold'), 2)}\n"
            f"FA3+H rank among candidate FA3 ions: {fa3_rank if fa3_rank is not None else ''}\n"
            f"FA3+H strongest: {'Yes' if result.get('fa3_protonated_is_strongest') else 'No' if fa3_found else ''}\n"
            f"FA3+H ratio to strongest: {self._evidence_float_text(fa3_ratio, 3)}\n"
            f"Strongest/second FA3+H ratio: {strongest_second_text}"
        )
        full_text = (
            position_summary
            + "\n\nMatched diagnostic ions:\n  "
            + ("\n  ".join(matched_lines) or "None")
            + "\n\nMissing diagnostic ions:\n  "
            + ("\n  ".join(missing_lines) or "None")
            + "\n\nExclusive evidence:\n  "
            + ("\n  ".join(exclusive_lines) or "None")
            + "\n\nContradictory evidence:\n  "
            + ("\n  ".join(contradiction_lines) or "None")
            + "\n\nMixture interpretation:\n  "
            + "\n  ".join(mixture_lines)
            + "\n\nWarnings:\n  "
            + ("\n  ".join(warning_lines) or "None")
        )
        return {
            "is_unresolved": is_unresolved,
            "display_type_text": display_type_text,
            "reported_position_text": reported_position_text,
            "supported_positions_text": supported_text,
            "unresolved_positions_text": unresolved_text,
            "family_supported_positions_text": family_supported_text,
            "family_unsupported_positions_text": family_unsupported_text,
            "member_keys": member_keys,
            "l4_evidence_text": l4_evidence_text,
            "representation_note": representation_note,
            "matched_ions": matched_ions,
            "missing_ions": missing_ions,
            "exclusive_ions": exclusive_ions,
            "contradictions": contradictions,
            "matched_lines": matched_lines,
            "missing_lines": missing_lines,
            "exclusive_lines": exclusive_lines,
            "contradiction_lines": contradiction_lines,
            "mixture_lines": mixture_lines,
            "global_warning_lines": global_warning_lines,
            "candidate_warning_lines": candidate_warning_lines,
            "display_warning_lines": display_warning_lines,
            "warning_lines": warning_lines,
            "fa3_found": fa3_found,
            "fa3_experimental_mz": fa3_experimental_mz,
            "fa3_ppm_error": fa3_ppm_error,
            "fa3_intensity": fa3_intensity,
            "fa3_rank": fa3_rank,
            "fa3_ratio": fa3_ratio,
            "strongest_second_ratio": strongest_second_numeric,
            "position_summary": position_summary,
            "full_text": full_text,
        }

    def _update_hdcs_analysis(self, display_id: str) -> None:
        result = self.display_row_by_id.get(display_id)
        if result is None:
            result = self.hdcs_results.get(display_id)
        candidate_key = str((result or {}).get("representative_candidate_key", (result or {}).get("candidate_key", display_id)))
        candidate = self.hdcs_candidates.get(candidate_key)
        if not candidate or not result:
            return
        reported_position = result.get("reported_ester_position")
        unresolved_positions = list(result.get("unresolved_ester_positions", []))
        is_unresolved = (
            result.get("display_type") == "unresolved_positions"
            or reported_position is None
        )
        possible_positions_text = (
            " / ".join(
                self._ester_bond_position_text(position)
                for position in unresolved_positions
            )
            or "Unknown"
        )
        if is_unresolved:
            self.hdcs_position_notice_var.set(
                "FAHFA ester bond position unresolved — possible positions: "
                f"{possible_positions_text}. "
                "The spectrum and diagnostic-ion table below use one representative "
                "theoretical candidate from this unresolved position group."
            )
            self.hdcs_position_notice_label.configure(bg="#FFF4E5", fg="#9A5B00")
            self.hdcs_diagnostic_frame.configure(
                text="HDCS Diagnostic Ions — Representative Candidate"
            )
        else:
            self.hdcs_position_notice_var.set(
                "FAHFA ester bond position confirmed: "
                f"{self._ester_bond_position_text(reported_position)}"
            )
            self.hdcs_position_notice_label.configure(bg="#E8F5E9", fg="#166534")
            self.hdcs_diagnostic_frame.configure(text="HDCS Diagnostic Ions")
        theoretical_precursor = float(candidate.get("precursor_mz", 0.0) or 0.0)
        experimental_precursor = float(self.parent_mz_exp or 0.0)
        mass_error_ppm = (
            (experimental_precursor - theoretical_precursor)
            / theoretical_precursor
            * 1e6
            if theoretical_precursor > 0 and experimental_precursor > 0
            else None
        )
        info_values = {
            "candidate_name": result.get("reported_common_name", candidate["name"]),
            "structure": result.get("reported_lipid_name", candidate["structure"]),
            "theoretical_precursor": f"{theoretical_precursor:.5f}",
            "experimental_precursor": f"{experimental_precursor:.5f}",
            "mass_error_ppm": (
                f"{mass_error_ppm:+.2f}" if mass_error_ppm is not None else ""
            ),
            "sn_position": candidate["sn_fahfa"],
            "ester_bond_position": (
                f"Possible: {possible_positions_text}"
                if is_unresolved
                else self._ester_bond_position_text(reported_position)
            ),
        }
        for key, value in info_values.items():
            self.hdcs_info_vars[key].set(value)
        mixture = self.hdcs_global_result.get("mixture_detected", False)
        fit_quality = self.hdcs_global_result.get("nnls_fit_quality", 0.0)
        nnls_eligible = bool(result.get("nnls_eligible", False))
        nnls_value = result.get(
            "display_nnls_contribution", result.get("nnls_contribution")
        )
        summary_values = {
            "wdic": f"{result['wdic']:.4f}",
            "nep": f"{result['nep']:.4f}",
            "base": f"{result.get('display_base_score', result['base_score']):.4f}",
            "l4": f"{result.get('display_l4_score', result['l4_score']):.4f}",
            "final": f"{result.get('display_final_score', result['final_score']):.4f}",
            "rank": str(result.get("display_rank", result["rank"])),
            "mixture": "Yes" if mixture else "No",
            "nnls": (
                f"{float(nnls_value):.3f}"
                if nnls_eligible and nnls_value is not None else "—"
            ),
            "fit": f"{fit_quality:.3f}" if nnls_eligible else "—",
        }
        for key, value in summary_values.items():
            self.hdcs_summary_vars[key].set(value)
        for item in self.hdcs_ion_tree.get_children():
            self.hdcs_ion_tree.delete(item)
        for ion in sorted(result["detected_ions"].values(), key=lambda item: (item["level"], item["theo_mz"])):
            effective_level = f"L{ion['level']}"
            level = fragment_levels_display(ion.get("levels", [ion["level"]]))
            annotation = str(ion.get("annotation", ""))
            if int(ion.get("merged_member_count", 1)) > 1:
                annotation += f" [Merged interpretations: {ion['merged_member_count']}]"
            groups = " / ".join(str(group) for group in ion.get("groups", []) if group)
            self.hdcs_ion_tree.insert(
                "", "end",
                values=(
                    level, ion["fragment_key"], annotation, f"{ion['theo_mz']:.5f}",
                    f"{ion['exp_mz']:.5f}" if ion["exp_mz"] is not None else "",
                    f"{ion['ppm_err']:.2f}" if ion["ppm_err"] is not None else "",
                    f"{ion['intensity']:.2f}" if ion["found"] else "",
                    f"{float(ion.get('threshold', 0.0)):g}",
                    "Yes" if ion["found"] else "No", groups or ion.get("group") or "",
                    "Yes" if ion.get("is_shared") else "No",
                ),
                tags=(effective_level,),
            )
    def _remember_spectrum_home_view(self) -> None:
        """Save the freshly drawn full-spectrum limits as the Home view."""
        self._spectrum_home_xlim = tuple(float(value) for value in self.ax.get_xlim())
        self._spectrum_home_ylim = tuple(float(value) for value in self.ax.get_ylim())
        if "spectrum_toolbar" in self.__dict__:
            self.spectrum_toolbar.update()
            self.spectrum_toolbar.push_current()

    def _reset_spectrum_view(self) -> None:
        """Restore the exact unzoomed view for the currently selected spectrum."""
        self.ax.set_xlim(*self._spectrum_home_xlim)
        self.ax.set_ylim(*self._spectrum_home_ylim)
        if "spectrum_toolbar" in self.__dict__:
            self.spectrum_toolbar.update()
            self.spectrum_toolbar.push_current()
        self.canvas.draw_idle()

    def _on_spectrum_scroll(self, event: Any) -> None:
        """Zoom the spectrum horizontally around the mouse pointer."""
        if event.inaxes is not self.ax or event.xdata is None:
            return
        current_left, current_right = self.ax.get_xlim()
        current_width = current_right - current_left
        if current_width <= 0:
            return
        if event.button == "up":
            scale = 0.8
        elif event.button == "down":
            scale = 1.25
        else:
            return
        anchor = float(event.xdata)
        new_left = anchor - (anchor - current_left) * scale
        new_right = anchor + (current_right - anchor) * scale
        if new_right - new_left < 0.1:
            return
        self.ax.set_xlim(new_left, new_right)
        if "spectrum_toolbar" in self.__dict__:
            self.spectrum_toolbar.push_current()
        self.canvas.draw_idle()

    def _init_plot(self):
        self.ax.clear()
        self.ax.axhline(0, color='black', linewidth=1)
        self.ax.text(0.02, 0.95, "Experimental EAciD-MS/MS Spectrum", transform=self.ax.transAxes, fontsize=11, va='top')
        self.ax.text(0.02, 0.05, "Predicted EAciD-MS/MS Spectrum", transform=self.ax.transAxes, fontsize=11, va='bottom')
        self.ax.set_ylim(-100, 100)
        self.ax.set_xlim(100, 1500)
        self.ax.set_yticks([-100, -50, 0, 50, 100])
        self.ax.set_yticklabels([100, 50, 0, 50, 100])
        self.ax.set_axis_off()
        self._remember_spectrum_home_view()
        self.canvas.draw_idle()

    def _candidate_is_visible(self, index: int) -> bool:
        """Return whether one Tier I/II candidate is in the main GUI list."""
        if self.candidates_df.empty or index < 0 or index >= len(self.candidates_df):
            return False
        row = self.candidates_df.iloc[index]
        candidate_id = str(row.get("_candidate_id", f"candidate-{index:05d}"))
        return any(
            str(result.get("representative_candidate_key", result.get("candidate_key", "")))
            == candidate_id
            for result in self.display_candidate_rows
        )

    def _update_candidate_count(self, displayed: int, scored: int) -> None:
        self.lbl_candidate_count.config(
            text=f"Displayed Candidates: {int(displayed)}"
        )

    def _clear_candidate_display(self) -> None:
        """Clear only the selected-candidate widgets, never global scoring results."""
        self.smiles_var.set("")
        self.common_name_var.set("")
        self.predicted_peaks = []
        self.raw_predicted_peaks = []
        for item in self.tree_right.get_children():
            self.tree_right.delete(item)
        for item in self.hdcs_ion_tree.get_children():
            self.hdcs_ion_tree.delete(item)
        for variable in self.hdcs_info_vars.values():
            variable.set("")
        for variable in self.hdcs_summary_vars.values():
            variable.set("")
        if hasattr(self, "hdcs_position_notice_var"):
            self.hdcs_position_notice_var.set("")
        if hasattr(self, "hdcs_position_notice_label"):
            self.hdcs_position_notice_label.configure(bg="#E8F5E9", fg="#166534")
        if hasattr(self, "hdcs_diagnostic_frame"):
            self.hdcs_diagnostic_frame.configure(text="HDCS Diagnostic Ions")
        self._init_plot()

    def _refresh_candidate_tree(self, preserve_selection: bool = True) -> None:
        """Display Tier I/II rows; NNLS values exist only for Tier I rows."""
        if not hasattr(self, "tree_left"):
            return
        selected_display_id: Optional[str] = None
        if preserve_selection:
            selected_items = self.tree_left.selection()
            if selected_items:
                selected_display_id = str(selected_items[0])

        for item in self.tree_left.get_children():
            self.tree_left.delete(item)

        scored_count = len(self.hdcs_results)
        if self.candidates_df.empty:
            self._update_candidate_count(displayed=0, scored=scored_count)
            self._clear_candidate_display()
            return

        display_rows = self.display_candidate_rows
        nnls_result = self.hdcs_global_result.get("decomposition")
        nnls_available = isinstance(nnls_result, dict)
        entries = list(display_rows)
        mode = self.sort_var.get()
        if mode == "Sort by Theory NNLS":
            entries.sort(key=nnls_display_sort_key)
        elif mode == "Sort by HDCS Score":
            entries.sort(key=hdcs_score_sort_key)
        elif mode == "Sort by Rank":
            entries.sort(
                key=lambda result: (
                    int(result.get("display_rank", 10**9)),
                    hdcs_score_sort_key(result),
                )
            )
        else:
            entries.sort(
                key=lambda result: int(result.get("display_original_order", 10**9))
            )

        visible_display_ids = [str(result.get("display_id", result.get("candidate_key", ""))) for result in entries]
        for result in entries:
            display_id = str(result.get("display_id", result.get("candidate_key", "")))
            candidate_key = str(result.get("representative_candidate_key", result.get("candidate_key", "")))
            index = self.candidate_index_by_key.get(candidate_key)
            if index is None:
                continue
            row = self.candidates_df.iloc[index]
            structure = str(
                result.get("reported_lipid_name") or row["TG_FAHFA_Structure"]
            ).strip('="')
            sn1 = str(row["FA_Sn1"]).strip('="')
            sn2 = str(row["FA_Sn2"]).strip('="')
            sn3 = str(row["FA_Sn3"]).strip('="')
            if not result.get("ester_position_confirmed", False):
                sn1 = remove_ester_position_from_lipid_name(sn1)
                sn2 = remove_ester_position_from_lipid_name(sn2)
                sn3 = remove_ester_position_from_lipid_name(sn3)
            rank = result.get("display_rank", result.get("rank", ""))
            score = float(result.get("display_final_score", result.get("final_score", 0.0)))
            nnls_value = result.get(
                "display_nnls_contribution", result.get("nnls_contribution")
            )
            nnls_text = (
                f"{float(nnls_value):.3f}"
                if nnls_available
                and bool(result.get("nnls_eligible", False))
                and nnls_value is not None
                else "—"
            )
            self.tree_left.insert(
                "", "end", iid=display_id,
                values=(
                    structure, sn1, sn2, sn3, rank or "", f"{score:.4f}", nnls_text,
                ),
            )

        self._update_candidate_count(displayed=len(visible_display_ids), scored=scored_count)
        available_iids = set(self.tree_left.get_children())
        if selected_display_id is not None and selected_display_id in available_iids:
            target_iid: Optional[str] = selected_display_id
        elif visible_display_ids:
            target_iid = visible_display_ids[0]
        else:
            target_iid = None

        if target_iid is not None:
            self.tree_left.selection_set(target_iid)
            self.tree_left.focus(target_iid)
            self.tree_left.see(target_iid)
            self._select_candidate_hdcs()
            if self._last_completed_status:
                self.status_var.set(self._last_completed_status)
        else:
            self._clear_candidate_display()
            if scored_count:
                self.status_var.set(self._last_completed_status or "Completed")

    def _on_candidate_filter_changed(self) -> None:
        if self.candidates_df.empty or not self.hdcs_results:
            return
        self._refresh_candidate_tree(preserve_selection=True)

    #############################################################################
    #  GUI INPUT PREVIEW METHODS
    #############################################################################

    def _reset_analysis_results(self) -> None:
        """Clear results that must never survive an input or mapping change."""
        self.analysis_completed = False
        self.run_context = {}
        self.analysis_start_time = ""
        self.analysis_completion_time = ""
        if hasattr(self, "export_button"):
            self.export_button.configure(state="disabled")
        self.raw_spec_df = pd.DataFrame()
        self.current_spec_df = pd.DataFrame()
        self.spectrum_filter_stats = {}
        self.parent_mz_exp = 0.0
        self.candidates_df = pd.DataFrame()
        self.predicted_peaks = []
        self.raw_predicted_peaks = []
        self.candidate_raw_peaks.clear()
        self.candidate_display_peaks.clear()
        self.hdcs_candidates.clear()
        self.hdcs_results.clear()
        self.display_candidate_rows.clear()
        self.display_row_by_id.clear()
        self.hdcs_display_results.clear()
        self.candidate_index_by_key.clear()
        self.hdcs_global_result = {}
        self.candidate_warnings.clear()
        self._candidate_scores.clear()
        self._candidate_ranks.clear()
        self._last_completed_status = ""
        if hasattr(self, "tree_left"):
            for item in self.tree_left.get_children():
                self.tree_left.delete(item)
            self._update_candidate_count(0, 0)
        if hasattr(self, "tree_right"):
            self._clear_candidate_display()

    def _invalidate_input_mapping(self) -> None:
        """Invalidate confirmed mapping and all results for a changed input file."""
        self.preview_raw_df = pd.DataFrame()
        self.input_column_mapping = {
            "mz": None, "intensity": None,
        }
        self.input_mapping_confirmed = False
        self.input_mapping_file = ""
        self.input_metadata = {}
        self._reset_analysis_results()

    def _on_file_path_changed(self, *_args: Any) -> None:
        """Invalidate a mapping whenever the file entry no longer matches it."""
        current = self.file_var.get().strip()
        if self.input_mapping_confirmed and current:
            current_key = os.path.normcase(os.path.abspath(current))
            mapped_key = os.path.normcase(os.path.abspath(self.input_mapping_file))
            if current_key == mapped_key:
                return
        if self.input_mapping_confirmed or self.analysis_completed or not self.preview_raw_df.empty:
            self._invalidate_input_mapping()

    def _on_browse(self):
        filepath = filedialog.askopenfilename(
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if filepath:
            self.file_var.set(filepath)

    def _close_preview_window(self) -> None:
        """Release the modal grab and close the current preview safely."""
        window = self._preview_window
        self._preview_window = None
        if window is not None and window.winfo_exists():
            try:
                window.grab_release()
            except tk.TclError:
                pass
            window.destroy()
        self._preview_dialog_state = {}

    def _open_data_preview(self, run_after_confirm: bool = False) -> None:
        """Open a modal raw-data preview with explicit column mapping controls."""
        file_path = self.file_var.get().strip()
        if not file_path or not os.path.isfile(file_path):
            messagebox.showwarning(
                "Warning", "Please select a valid CSV data file first."
            )
            return
        try:
            raw_df, delimiter = read_raw_spectrum_table(file_path)
        except Exception as exc:
            messagebox.showerror("CSV Preview Error", str(exc))
            return
        if self._preview_window is not None and self._preview_window.winfo_exists():
            self._close_preview_window()
        same_confirmed_file = (
            self.input_mapping_confirmed
            and os.path.normcase(os.path.abspath(file_path))
            == os.path.normcase(os.path.abspath(self.input_mapping_file))
        )
        available_columns = tuple(str(column) for column in raw_df.columns)
        recommended = infer_spectrum_column_mapping(available_columns)
        requested_mapping = (
            dict(self.input_column_mapping) if same_confirmed_file else {}
        )
        initial_mapping = {
            role: valid_initial_column(
                requested_mapping.get(role),
                available_columns,
                recommended.get(role),
            )
            for role in ("mz", "intensity")
        }
        window = tk.Toplevel(self)
        self._preview_window = window
        window.title("CSV Data Preview and Column Mapping")
        window.geometry("1100x650")
        window.minsize(900, 600)
        window.transient(self)
        window.grab_set()
        window.protocol("WM_DELETE_WINDOW", self._close_preview_window)
        window.rowconfigure(0, weight=3)
        window.rowconfigure(2, weight=2)
        window.columnconfigure(0, weight=1)

        preview_frame = ttk.LabelFrame(window, text="CSV Data Preview", padding=5)
        preview_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 4))
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        columns = available_columns
        preview_tree = ttk.Treeview(
            preview_frame, columns=columns, show="headings", selectmode="none"
        )
        for column in columns:
            preview_tree.heading(column, text=column)
            sample_values = [column] + [
                "" if pd.isna(value) else str(value)
                for value in raw_df[column].head(15)
            ]
            width = min(180, max(80, max(len(value) for value in sample_values) * 8))
            preview_tree.column(column, width=width, minwidth=70, stretch=False)
        for row_index, row in raw_df.head(15).iterrows():
            preview_tree.insert(
                "", "end", iid=f"preview-{row_index}",
                values=tuple("" if pd.isna(value) else value for value in row.tolist()),
            )
        preview_vsb = ttk.Scrollbar(
            preview_frame, orient="vertical", command=preview_tree.yview
        )
        preview_hsb = ttk.Scrollbar(
            preview_frame, orient="horizontal", command=preview_tree.xview
        )
        preview_tree.configure(
            yscrollcommand=preview_vsb.set, xscrollcommand=preview_hsb.set
        )
        preview_tree.grid(row=0, column=0, sticky="nsew")
        preview_vsb.grid(row=0, column=1, sticky="ns")
        preview_hsb.grid(row=1, column=0, sticky="ew")
        ttk.Label(
            preview_frame,
            text=f"Previewing first {min(15, len(raw_df))} rows of {len(raw_df)} rows",
        ).grid(row=2, column=0, sticky="w", pady=(4, 0))

        mapping_frame = ttk.LabelFrame(window, text="Column Mapping", padding=6)
        mapping_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        mapping_frame.columnconfigure(1, weight=1)
        ttk.Label(mapping_frame, text="m/z column:").grid(
            row=0, column=0, sticky="e", padx=(0, 6), pady=2
        )
        mz_var = tk.StringVar(value=initial_mapping.get("mz") or "")
        mz_combo = ttk.Combobox(
            mapping_frame,
            textvariable=mz_var,
            values=columns,
            state="readonly",
        )
        mz_combo.grid(row=0, column=1, sticky="ew", pady=2)
        mz_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._update_preview_validation(),
        )
        ttk.Label(mapping_frame, text="Intensity column:").grid(
            row=1, column=0, sticky="e", padx=(0, 6), pady=2
        )
        intensity_var = tk.StringVar(
            value=initial_mapping.get("intensity") or ""
        )
        intensity_combo = ttk.Combobox(
            mapping_frame,
            textvariable=intensity_var,
            values=columns,
            state="readonly",
        )
        intensity_combo.grid(row=1, column=1, sticky="ew", pady=2)
        intensity_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._update_preview_validation(),
        )

        summary_frame = ttk.LabelFrame(
            window, text="Mapping Validation Summary", padding=5
        )
        summary_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=4)
        summary_frame.rowconfigure(0, weight=1)
        summary_frame.columnconfigure(0, weight=1)
        summary_text = tk.Text(summary_frame, height=11, wrap="word")
        summary_scroll = ttk.Scrollbar(
            summary_frame, orient="vertical", command=summary_text.yview
        )
        summary_text.configure(yscrollcommand=summary_scroll.set, state="disabled")
        summary_text.grid(row=0, column=0, sticky="nsew")
        summary_scroll.grid(row=0, column=1, sticky="ns")

        button_frame = ttk.Frame(window)
        button_frame.grid(row=3, column=0, sticky="e", padx=8, pady=(4, 8))
        ttk.Button(
            button_frame, text="Cancel", command=self._close_preview_window
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(
            button_frame,
            text="Confirm Mapping",
            command=lambda: self._confirm_preview_mapping(False),
        ).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(
            button_frame,
            text="Confirm & Run",
            command=lambda: self._confirm_preview_mapping(True),
        ).pack(side=tk.LEFT)

        self._preview_dialog_state = {
            "window": window,
            "file_path": os.path.abspath(file_path),
            "raw_df": raw_df,
            "delimiter": delimiter,
            "mapping_vars": {
                "mz": mz_var,
                "intensity": intensity_var,
            },
            "mz_combo": mz_combo,
            "intensity_combo": intensity_combo,
            "summary_text": summary_text,
            "run_after_confirm": bool(run_after_confirm),
            "last_validation": {},
        }
        self._update_preview_validation()

    def _update_preview_validation(self) -> Dict[str, Any]:
        """Refresh mapping statistics and resolved precursor information."""
        state = getattr(self, "_preview_dialog_state", None)
        if not state or not state["window"].winfo_exists():
            return {}
        mapping = {
            role: variable.get() or None
            for role, variable in state["mapping_vars"].items()
        }
        result: Dict[str, Any] = {"valid": False, "mapping": mapping}
        lines = [
            f"Detected delimiter: {state['delimiter']}",
            f"Total rows: {len(state['raw_df'])}",
            f"m/z column: {mapping.get('mz') or 'Not found'}",
            f"Intensity column: {mapping.get('intensity') or 'Not selected'}",
        ]
        try:
            normalized, stats = normalize_spectrum_from_mapping(
                state["raw_df"], mapping
            )
            precursor, precursor_intensity, precursor_source_row = (
                resolve_precursor_from_base_peak(normalized)
            )
            tie_count = int(
                (normalized["intensity"] == precursor_intensity).sum()
            )
            lines.extend(
                [
                    f"Valid m/z-intensity rows: {stats['valid_row_count']}",
                    f"Invalid rows: {stats['invalid_row_count']}",
                    f"Negative intensity rows: {stats['negative_intensity_count']}",
                    f"Highest-intensity peak m/z: {precursor:.5f}",
                    f"Highest intensity: {precursor_intensity:g}",
                    f"Precursor m/z used for analysis: {precursor:.5f}",
                ]
            )
            if tie_count > 1:
                lines.append(
                    "Warning: Multiple peaks share the maximum intensity; "
                    "the highest-m/z peak will be used."
                )
            result.update(
                {
                    "valid": True,
                    "normalized": normalized,
                    "normalization_stats": stats,
                    "precursor_mz": precursor,
                    "precursor_intensity": precursor_intensity,
                    "precursor_source_row": precursor_source_row,
                    "precursor_tie_count": tie_count,
                }
            )
        except Exception as exc:
            lines.append(f"Validation error: {exc}")
            result["error"] = str(exc)
        summary_text = state["summary_text"]
        summary_text.configure(state="normal")
        summary_text.delete("1.0", tk.END)
        summary_text.insert("1.0", "\n".join(lines))
        summary_text.configure(state="disabled")
        state["last_validation"] = result
        return result

    def _confirm_preview_mapping(self, run_analysis: bool) -> None:
        """Persist a validated mapping, optionally starting HDCS immediately."""
        state = getattr(self, "_preview_dialog_state", None)
        validation = self._update_preview_validation()
        if not state or not validation.get("valid"):
            messagebox.showerror(
                "Column Mapping Error",
                validation.get("error", "Select valid m/z and Intensity columns."),
            )
            return
        self._reset_analysis_results()
        self.preview_raw_df = state["raw_df"].copy()
        self.input_column_mapping = dict(validation["mapping"])
        self.input_mapping_confirmed = True
        self.input_mapping_file = state["file_path"]
        raw_warnings = list(state["raw_df"].attrs.get("input_warnings", []))
        if validation["precursor_tie_count"] > 1:
            raw_warnings.append(
                "Multiple peaks shared the maximum intensity. "
                "The highest-m/z peak was selected."
            )
        self.input_metadata = {
            "delimiter": state["delimiter"],
            "encoding": state["raw_df"].attrs.get("encoding", ""),
            "normalization_stats": dict(validation["normalization_stats"]),
            "precursor_mz": float(validation["precursor_mz"]),
            "precursor_intensity": float(validation["precursor_intensity"]),
            "precursor_source_row": int(validation["precursor_source_row"]),
            "precursor_tie_count": int(validation["precursor_tie_count"]),
            "warnings": list(dict.fromkeys(raw_warnings)),
        }
        self.status_var.set("Column mapping confirmed. Ready to run.")
        self._close_preview_window()
        if run_analysis:
            self.after_idle(self._run_hdcs_analysis)
            
    def _on_run(self):
        self._run_hdcs_analysis()
            
    def _run_hdcs_analysis(self):
        csv_file = self.file_var.get().strip()
        if not csv_file or not os.path.exists(csv_file):
            return messagebox.showwarning("Warning", "Please select a valid CSV data file first.")
        current_file = os.path.normcase(os.path.abspath(csv_file))
        confirmed_file = (
            os.path.normcase(os.path.abspath(self.input_mapping_file))
            if self.input_mapping_file else ""
        )
        if not self.input_mapping_confirmed or current_file != confirmed_file:
            self._open_data_preview(run_after_confirm=True)
            return
        self._reset_analysis_results()
        self.analysis_start_time = datetime.now().isoformat(timespec="seconds")
        try:
            self.status_var.set("Applying confirmed column mapping...")
            self.update_idletasks()
            if self.preview_raw_df.empty:
                self.preview_raw_df, delimiter = read_raw_spectrum_table(csv_file)
                self.input_metadata["delimiter"] = delimiter
            raw_spec, normalization_stats = normalize_spectrum_from_mapping(
                self.preview_raw_df, self.input_column_mapping
            )
            self.raw_spec_df = raw_spec.copy()
            parent_mz_exp, parent_intensity, parent_source_row = (
                resolve_precursor_from_base_peak(raw_spec)
            )
            precursor_source = "Highest-intensity peak"
            precursor_tie_count = int(
                (raw_spec["intensity"] == parent_intensity).sum()
            )
            self.input_metadata["normalization_stats"] = dict(normalization_stats)
            self.input_metadata["precursor_mz"] = float(parent_mz_exp)
            self.input_metadata["precursor_intensity"] = float(parent_intensity)
            self.input_metadata["precursor_source_row"] = int(parent_source_row)
            self.input_metadata["precursor_tie_count"] = int(precursor_tie_count)
            self.input_metadata["precursor_source"] = precursor_source
            existing_input_warnings = list(self.input_metadata.get("warnings", []))
            if precursor_tie_count > 1:
                existing_input_warnings.append(
                    "Multiple peaks shared the maximum intensity. "
                    "The highest-m/z peak was selected."
                )
            self.input_metadata["warnings"] = list(dict.fromkeys(
                existing_input_warnings
            ))
            self.parent_mz_exp = parent_mz_exp
            self.mz_var.set(f"{parent_mz_exp:.4f}")

            noise, sn_threshold = self._get_spectrum_filter_config()
            hdcs_config = self._get_hdcs_config()
            self._update_effective_cutoff_preview()
            self.status_var.set("Applying Noise/S/N filter...")
            self.update_idletasks()
            spec, filter_stats = filter_spectrum_by_manual_sn(
                raw_spec, noise=noise, sn_threshold=sn_threshold
            )
            self.spectrum_filter_stats = dict(filter_stats)
            retained_source_rows = set(
                pd.to_numeric(spec.get("source_row"), errors="coerce")
                .dropna().astype(int).tolist()
                if "source_row" in spec.columns else []
            )
            if parent_source_row not in retained_source_rows:
                raise ValueError(
                    "The highest-intensity precursor peak did not pass the current "
                    "Noise/S/N filter.\n\nPlease check the Noise and S/N threshold settings."
                )
            self.current_spec_df = spec.copy()

            matches = match_parent_to_library(parent_mz_exp, TARGET_LIBRARY)
            if not matches:
                raise ValueError(f"Precursor {parent_mz_exp:.5f} did not match any TG-FAHFA composition")
            best_match = min(matches, key=lambda item: ppm_error(parent_mz_exp, item['mz']))
            self.status_var.set("Identifying FA neutral losses...")
            self.update_idletasks()
            fa_hits = identify_FA_guided(best_match['mz'], spec, parent_mz_exp)
            if fa_hits.empty:
                raise ValueError("No ordinary fatty-acid neutral losses were identified")
            unique_fas = [format_fa_string(fa) for fa in fa_hits["FA"].drop_duplicates().tolist()]
            fahfa_hits = identify_fahfa(spec, unique_fas)
            if fahfa_hits.empty:
                raise ValueError("No FAHFA diagnostic pair was identified")
            result_df = build_candidates_strict(
                parent_mz_exp, best_match['mz'], unique_fas, fahfa_hits,
                best_match['c'], best_match['db']
            )
            if result_df.empty:
                raise ValueError("No complete TG-FAHFA candidate satisfies the identified composition")
            self.candidates_df = result_df.sort_values(
                by=["FAHFA", "FA1", "FA2", "SortOrder"]
            ).reset_index(drop=True)
            self.candidates_df["_candidate_id"] = [
                f"candidate-{index:05d}" for index in range(len(self.candidates_df))
            ]
            self.candidate_index_by_key = {
                str(row["_candidate_id"]): index
                for index, row in self.candidates_df.iterrows()
            }

            import warnings as warnings_module
            total = len(self.candidates_df)
            for index, row in self.candidates_df.iterrows():
                if index % 5 == 0 or index + 1 == total:
                    self.status_var.set(f"Generating candidate spectra: {index + 1}/{total}")
                    self.update_idletasks()
                candidate_id = str(row["_candidate_id"])
                try:
                    with warnings_module.catch_warnings(record=True) as caught:
                        warnings_module.simplefilter("always")
                        display_peaks = self.generator.generate(str(row["SMILES"]))
                    raw_peaks = [dict(peak) for peak in self.generator.last_raw_peaks]
                    self.candidate_raw_peaks[candidate_id] = raw_peaks
                    self.candidate_display_peaks[candidate_id] = [dict(peak) for peak in display_peaks]
                    hdcs_candidate = convert_tg_candidate_to_hdcs_candidate(
                        row, raw_peaks, self.generator, candidate_key=candidate_id
                    )
                    self.hdcs_candidates[candidate_id] = hdcs_candidate
                    if caught and len(self.candidate_warnings) < 20:
                        self.candidate_warnings.append(
                            f"{candidate_id}: {len(caught)} fragment consistency warning(s)."
                        )
                except Exception as exc:
                    self.candidate_warnings.append(f"{candidate_id} failed: {exc}")

            if not self.hdcs_candidates:
                raise ValueError("All candidate theoretical-spectrum conversions failed")
            self.status_var.set("Scoring candidates and running mixture analysis...")
            self.update_idletasks()
            self.hdcs_global_result = score_dynamic_hdcs_spectrum(
                spec, list(self.hdcs_candidates.values()), hdcs_config
            )
            self.hdcs_results = dict(self.hdcs_global_result["results_by_key"])
            self.display_candidate_rows = list(
                self.hdcs_global_result.get("display_candidate_rows", [])
            )
            self.display_row_by_id = {
                str(result["display_id"]): result
                for result in self.display_candidate_rows
            }
            self.hdcs_display_results = {
                str(result["display_id"]): result
                for result in self.display_candidate_rows
            }
            self._candidate_scores.clear()
            self._candidate_ranks.clear()
            for index, row in self.candidates_df.iterrows():
                candidate_id = str(row["_candidate_id"])
                result = self.hdcs_results.get(candidate_id)
                if result is None:
                    continue
                self._candidate_scores[index] = float(result["final_score"])
                self._candidate_ranks[index] = int(result["rank"])
            warning_count = len(self._collect_export_warnings())
            displayed_count = len(self.display_candidate_rows)
            self._last_completed_status = "Completed"
            self.analysis_completion_time = datetime.now().isoformat(timespec="seconds")
            self.run_context = {
                "input_file": os.path.abspath(csv_file),
                "analysis_start_time": self.analysis_start_time,
                "analysis_time": self.analysis_completion_time,
                "software_file": os.path.basename(__file__),
                "software_version": APP_VERSION_TAG,
                "column_mapping": dict(self.input_column_mapping),
                "delimiter": self.input_metadata.get("delimiter", ""),
                "precursor_mz": float(parent_mz_exp),
                "precursor_intensity": float(parent_intensity),
                "precursor_source_row": int(parent_source_row),
                "precursor_source": precursor_source,
                "precursor_selection_method": "highest_intensity_peak",
                "precursor_tie_count": int(precursor_tie_count),
                "matched_composition": f"TG-FAHFA {best_match['c']}:{best_match['db']}",
                "matched_composition_mz": float(best_match["mz"]),
                "precursor_mass_error_ppm": float(
                    ppm_error(parent_mz_exp, best_match["mz"])
                ),
                "original_row_count": int(normalization_stats["original_row_count"]),
                "valid_input_peak_count": int(normalization_stats["valid_row_count"]),
                "invalid_input_row_count": int(normalization_stats["invalid_row_count"]),
                "retained_peak_count": int(filter_stats["retained_peak_count"]),
                "removed_peak_count": int(
                    normalization_stats["valid_row_count"]
                    - filter_stats["retained_peak_count"]
                ),
                "candidate_count": int(len(self.candidates_df)),
                "display_candidate_count": int(len(self.display_candidate_rows)),
                "visible_candidate_count": int(displayed_count),
                "warning_count": int(warning_count),
                "noise": float(noise),
                "sn_threshold": float(sn_threshold),
                "effective_intensity_cutoff": float(noise * sn_threshold),
                "hdcs_config": to_json_safe(vars(hdcs_config)),
            }
            self.analysis_completed = True
            self._refresh_candidate_tree(preserve_selection=False)
            self.status_var.set(self._last_completed_status)
            self.export_button.configure(state="normal")
        except Exception as exc:
            self.analysis_completed = False
            self.run_context = {}
            self.export_button.configure(state="disabled")
            self.status_var.set(f"Run failed: {exc}")
            messagebox.showerror("HDCS-MD Run Error", str(exc))

    #############################################################################
    #  GUI EXPORT METHODS
    #############################################################################

    def _build_run_summary_dataframe(self) -> pd.DataFrame:
        """Build the two-column audit summary for the completed run."""
        context = self.run_context
        decomposition = self.hdcs_global_result.get("decomposition")
        rows = [
            ("Input file", context.get("input_file", "")),
            ("Input file basename", os.path.basename(context.get("input_file", ""))),
            ("Analysis time", context.get("analysis_time", "")),
            ("Software file", context.get("software_file", "")),
            ("Software version", context.get("software_version", "")),
            ("m/z source column", self.input_column_mapping.get("mz") or ""),
            ("Intensity source column", self.input_column_mapping.get("intensity") or ""),
            ("Delimiter", context.get("delimiter", "")),
            ("Precursor m/z", context.get("precursor_mz")),
            ("Precursor intensity", context.get("precursor_intensity")),
            ("Precursor source row", context.get("precursor_source_row")),
            ("Precursor source", context.get("precursor_source", "")),
            ("Matched TG-FAHFA composition", context.get("matched_composition", "")),
            ("Matched theoretical precursor m/z", context.get("matched_composition_mz")),
            ("Precursor mass error ppm", context.get("precursor_mass_error_ppm")),
            ("Original CSV row count", context.get("original_row_count", 0)),
            ("Valid input peak count", context.get("valid_input_peak_count", 0)),
            ("Retained peak count", context.get("retained_peak_count", 0)),
            ("Removed peak count", context.get("removed_peak_count", 0)),
            ("Noise", context.get("noise")),
            ("S/N threshold", context.get("sn_threshold")),
            ("Effective intensity cutoff", context.get("effective_intensity_cutoff")),
            ("Raw candidate count", context.get("candidate_count", 0)),
            ("Display candidate count", context.get("display_candidate_count", 0)),
            ("Manual review candidate count", len(self.hdcs_global_result.get("manual_review_candidate_keys", []))),
            ("Display minimum Final HDCS", self.hdcs_global_result.get("display_min_final_score", 1.0)),
            ("Display maximum Final HDCS", self.hdcs_global_result.get("display_max_final_score", 1.1)),
            ("Display requires NEP = 0", self.hdcs_global_result.get("display_requires_zero_nep", True)),
            ("Mixture detected", bool(self.hdcs_global_result.get("mixture_detected", False))),
            ("NNLS target Final HDCS", self.hdcs_global_result.get("nnls_min_final_score", 1.1)),
            ("NNLS eligible candidate count", len(self.hdcs_global_result.get("nnls_eligible_candidate_keys", []))),
            ("NNLS executed", isinstance(decomposition, dict)),
            ("NNLS fit quality", decomposition.get("fit_quality") if isinstance(decomposition, dict) else "Not run"),
            ("Warning count", context.get("warning_count", 0)),
        ]
        return pd.DataFrame(rows, columns=["Field", "Value"])

    def _build_parameters_dataframe(self) -> pd.DataFrame:
        """Export actual GUI and HDCSConfig values used by this run."""
        config = dict(self.run_context.get("hdcs_config", {}))
        level_weights = config.get("level_weights", {})
        parameters = [
            ("ppm tolerance", config.get("ppm_tolerance"), "Experimental/theoretical ion matching tolerance"),
            ("Noise", self.run_context.get("noise"), "Manual baseline noise"),
            ("S/N threshold", self.run_context.get("sn_threshold"), "Manual calculated S/N cutoff"),
            ("Effective intensity cutoff", self.run_context.get("effective_intensity_cutoff"), "Noise × S/N threshold"),
            ("L1 threshold", config.get("l1_threshold"), "Raw intensity threshold for L1"),
            ("L2 threshold", config.get("l2_threshold"), "Raw intensity threshold for L2"),
            ("L3 threshold", config.get("l3_threshold"), "Raw intensity threshold for L3"),
            ("L4 threshold", config.get("l4_threshold"), "Raw intensity threshold for L4"),
            ("NNLS target Final HDCS", config.get("nnls_min_final_score"), "Only raw NEP = 0 and HDCS = 1.1 candidates enter NNLS"),
            ("NNLS enabled", config.get("nnls_enabled"), "Fit all distinguishable Tier I fingerprints"),
            ("Mixture detection enabled", config.get("mixture_detection_enabled"), "Enable dynamic mixture evidence detection"),
            ("Candidate display rule", "Raw NEP = 0 and 1.0 <= HDCS <= 1.1", "Main GUI/export list contains Tier I and Tier II candidates"),
            ("Manual review rule", "Raw NEP > 0", "Tier III candidates are excluded from the main list and NNLS"),
            ("L1 weight", level_weights.get(1, level_weights.get("1")), "WDIC/NEP L1 weight"),
            ("L2 weight", level_weights.get(2, level_weights.get("2")), "WDIC/NEP L2 weight"),
            ("L3 weight", level_weights.get(3, level_weights.get("3")), "WDIC/NEP L3 weight"),
            ("L4 weight", level_weights.get(4, level_weights.get("4")), "WDIC/NEP L4 weight"),
            ("NEP lambda", config.get("nep_lambda"), "Penalty coefficient in Base Score"),
            ("L4 bonus maximum", config.get("l4_bonus_max"), "Maximum position-specific L4 bonus"),
            ("WDIC pass threshold", config.get("wdic_pass_threshold"), "Strict L4 eligibility threshold"),
            ("sn mixture ratio", config.get("sn_mixture_ratio"), "Relative sn-mixture evidence cutoff"),
            ("sn mixture minimum intensity", config.get("sn_mixture_min_intensity"), "Absolute sn-mixture intensity cutoff"),
            ("ester mixture ratio", config.get("ester_mixture_ratio"), "Relative ester-position mixture cutoff"),
            ("ester mixture minimum intensity", config.get("ester_mixture_min_intensity"), "Absolute ester-position mixture cutoff"),
            ("shared ion tolerance ppm", config.get("shared_ion_tolerance_ppm"), "Observable diagnostic-ion merge tolerance"),
            ("precursor matching tolerance", PPM_PARENT, "TG-FAHFA precursor library tolerance"),
        ]
        return pd.DataFrame(parameters, columns=["Parameter", "Value", "Meaning"])

    def _candidate_display_id_map(self) -> Dict[str, List[str]]:
        """Map every bottom-level candidate key to traceable display rows."""
        mapping: Dict[str, List[str]] = defaultdict(list)
        for row in self.display_candidate_rows:
            display_id = str(row.get("display_id", ""))
            members = row.get("display_member_candidate_keys") or [
                row.get("representative_candidate_key", row.get("candidate_key"))
            ]
            for member in members:
                key = str(member or "")
                if key and display_id and display_id not in mapping[key]:
                    mapping[key].append(display_id)
        return dict(mapping)

    def _build_candidate_ranking_dataframe(self) -> pd.DataFrame:
        """Export every final display row, including unresolved position groups."""
        nnls_ran = isinstance(self.hdcs_global_result.get("decomposition"), dict)
        records: List[Dict[str, Any]] = []
        for result in self.display_candidate_rows:
            candidate_key = str(
                result.get("representative_candidate_key", result.get("candidate_key", ""))
            )
            candidate = self.hdcs_candidates.get(candidate_key, {})
            sn_chains = list(candidate.get("sn_chains", ("", "", "")))
            sn_chains += [""] * (3 - len(sn_chains))
            records.append(
                {
                    "Display_ID": result.get("display_id"),
                    "Display_Type": result.get("display_type"),
                    "Display_Rank": result.get("display_rank", result.get("rank")),
                    "Reported_Lipid_Name": result.get("reported_lipid_name", result.get("candidate_name")),
                    "Reported_Common_Name": result.get("reported_common_name", result.get("theoretical_common_name")),
                    "Representative_Candidate": candidate_key,
                    "Member_Candidates": _export_join(result.get("display_member_candidate_keys", [])),
                    "sn-1": sn_chains[0], "sn-2": sn_chains[1], "sn-3": sn_chains[2],
                    "FA3": result.get("fa3_name", candidate.get("fa3_name", "")),
                    "HFA": result.get("hfa_name", candidate.get("hfa_name", "")),
                    "FAHFA_Position_Group": result.get("sn_fahfa", candidate.get("sn_fahfa", "")),
                    "Reported_Ester_Position": result.get("reported_ester_position"),
                    "Supported_Ester_Positions": _export_join(result.get("supported_ester_positions", [])),
                    "Unresolved_Ester_Positions": _export_join(result.get("unresolved_ester_positions", [])),
                    "Ester_Position_Confirmed": bool(result.get("ester_position_confirmed", False)),
                    "WDIC": result.get("wdic"), "WDIC_Pass": bool(result.get("wdic_pass", False)),
                    "NEP": result.get("nep"),
                    "Base_Score": result.get("display_base_score", result.get("base_score")),
                    "L4_Score": result.get("display_l4_score", result.get("l4_score")),
                    "Final_HDCS_Score": result.get("display_final_score", result.get("final_score")),
                    "Theory_NNLS": (
                        result.get("display_nnls_contribution", result.get("nnls_contribution", 0.0))
                        if nnls_ran and bool(result.get("nnls_eligible", False))
                        else (None if nnls_ran else "Not run")
                    ),
                    "Equivalent_Group": result.get("equivalent_group"),
                    "Mixture_Candidate": bool(result.get("mixture_candidate", False)),
                    "Position_Resolution_Status": result.get("position_resolution_status"),
                    "Warning": result.get("fa3_protonated_warning", ""),
                }
            )
        return pd.DataFrame(records)

    def _build_manual_review_dataframe(self) -> pd.DataFrame:
        """Export Tier III candidates (raw NEP > 0) outside the main analysis."""
        columns = [
            "Candidate_Key", "Candidate_Name", "TG-FAHFA_Candidate_Structure",
            "sn-1", "sn-2", "sn-3", "FAHFA_Position_Group",
            "Ester_Position", "WDIC", "NEP", "Base_Score", "L4_Score",
            "Final_HDCS_Score", "HDCS_Rank", "NNLS_Eligible",
            "Review_Reason",
        ]
        records: List[Dict[str, Any]] = []
        for candidate_key, result in self.hdcs_results.items():
            if not _requires_manual_review(result):
                continue
            candidate = self.hdcs_candidates.get(str(candidate_key), {})
            sn_chains = list(candidate.get("sn_chains", ("", "", "")))
            sn_chains += [""] * (3 - len(sn_chains))
            records.append(
                {
                    "Candidate_Key": candidate_key,
                    "Candidate_Name": result.get(
                        "theoretical_common_name",
                        result.get("candidate_name", candidate.get("name", "")),
                    ),
                    "TG-FAHFA_Candidate_Structure": result.get(
                        "theoretical_lipid_name",
                        candidate.get("structure", candidate.get("name", "")),
                    ),
                    "sn-1": sn_chains[0],
                    "sn-2": sn_chains[1],
                    "sn-3": sn_chains[2],
                    "FAHFA_Position_Group": result.get(
                        "sn_fahfa", candidate.get("sn_fahfa", "")
                    ),
                    "Ester_Position": result.get(
                        "ester_pos", candidate.get("ester_pos")
                    ),
                    "WDIC": result.get("wdic"),
                    "NEP": result.get("nep"),
                    "Base_Score": result.get("base_score"),
                    "L4_Score": result.get("l4_score"),
                    "Final_HDCS_Score": result.get("final_score"),
                    "HDCS_Rank": result.get("rank"),
                    "NNLS_Eligible": False,
                    "Review_Reason": (
                        "Raw NEP > 0: Tier III; excluded from the main Candidate "
                        "list and NNLS, and retained for manual mirror-plot review."
                    ),
                }
            )
        frame = pd.DataFrame(records, columns=columns)
        if not frame.empty:
            frame = frame.sort_values(
                by=["Final_HDCS_Score", "NEP", "Candidate_Name"],
                ascending=[False, True, True],
                kind="stable",
                na_position="last",
            ).reset_index(drop=True)
        return frame

    def _build_diagnostic_ions_dataframe(self) -> pd.DataFrame:
        """Export every bottom-level candidate/observable diagnostic-ion match."""
        display_map = self._candidate_display_id_map()
        display_type_map = {
            str(row.get("display_id")): str(row.get("display_type", ""))
            for row in self.display_candidate_rows
        }
        representative_map = {
            str(row.get("display_id")): str(row.get("representative_candidate_key", ""))
            for row in self.display_candidate_rows
        }
        records: List[Dict[str, Any]] = []
        for candidate_key, result in self.hdcs_results.items():
            display_ids = display_map.get(str(candidate_key), [])
            display_id_text = _export_join(display_ids)
            display_types = _export_join(
                [display_type_map.get(display_id, "") for display_id in display_ids]
            )
            representatives = _export_join(
                [representative_map.get(display_id, "") for display_id in display_ids]
            )
            for ion in result.get("detected_ions", {}).values():
                found = bool(ion.get("found", False))
                records.append(
                    {
                        "Candidate_Key": candidate_key,
                        "Candidate_Name": result.get("candidate_name", result.get("theoretical_lipid_name")),
                        "Display_ID": display_id_text,
                        "Display_Type": display_types,
                        "Representative_Candidate": representatives,
                        "Level": ion.get("level"),
                        "Levels": _export_join(ion.get("levels", [])),
                        "Fragment_Key": ion.get("fragment_key"),
                        "Fragment_Keys": _export_join(ion.get("fragment_keys", [])),
                        "Annotation": ion.get("annotation"),
                        "Theoretical_mz": ion.get("theo_mz", ion.get("theoretical_mz")),
                        "Experimental_mz": ion.get("exp_mz") if found else None,
                        "ppm_Error": ion.get("ppm_error", ion.get("ppm_err")) if found else None,
                        "Experimental_Intensity": ion.get("intensity") if found else None,
                        "Threshold": ion.get("threshold"),
                        "Found": found,
                        "Group": ion.get("group"),
                        "Groups": _export_join(ion.get("groups", [])),
                        "Shared": bool(ion.get("is_shared", False)),
                        "Position_Specific": bool(ion.get("is_position_specific", False)),
                        "Ester_Position": result.get("ester_pos"),
                        "Formula": ion.get("formula", ""),
                        "Source_Role": _export_join(ion.get("source_roles", ion.get("source_role", ""))),
                    }
                )
        return pd.DataFrame(records)

    def _build_evidence_summary_dataframe(self) -> pd.DataFrame:
        """Export one shared GUI-equivalent Evidence record per display row."""
        columns = [
            "Display_ID", "Display_Type", "Display_Rank",
            "Reported_Lipid_Name", "Reported_Common_Name",
            "Theoretical_Candidate_Name", "Representative_Candidate_Key",
            "Representative_Candidate_Name", "Member_Candidate_Keys",
            "Member_Candidate_Count", "Equivalent_Group", "Position_Family",
            "Position_Resolution_Status", "Ester_Position_Confirmed",
            "Reported_Ester_Position", "Supported_Ester_Positions",
            "Unresolved_Ester_Positions", "Family_Supported_Ester_Positions",
            "Family_Unsupported_Ester_Positions",
            "L4_Position_Specific_Found_Count",
            "L4_Position_Specific_Total_Count", "L4_Evidence_Text",
            "WDIC", "WDIC_Pass", "NEP", "Base_Score", "L4_Score",
            "Final_HDCS_Score", "Display_Final_Score", "HDCS_Rank",
            "Theory_NNLS", "NNLS_Status", "NNLS_Fit_Quality",
            "FA3_Chain", "HFA_Chain", "FA3_Protonated_Theoretical_mz",
            "FA3_Protonated_Status", "FA3_Protonated_Experimental_mz",
            "FA3_Protonated_ppm_Error", "FA3_Protonated_Intensity",
            "FA3_Protonated_Threshold", "FA3_Protonated_Rank",
            "FA3_Protonated_Is_Strongest",
            "FA3_Protonated_Ratio_To_Strongest", "Strongest_Second_FA3_Ratio",
            "Matched_Diagnostic_Ion_Count", "Matched_Diagnostic_Ions",
            "Missing_Diagnostic_Ion_Count", "Missing_Diagnostic_Ions",
            "Exclusive_Evidence_Count", "Exclusive_Evidence",
            "Contradictory_Evidence_Count", "Contradictory_Evidence",
            "Mixture_Detected", "sn_Mixture_Detected",
            "Ester_Position_Mixture_Detected", "Mixture_Interpretation",
            "Mixture_Candidate", "Global_Warnings", "Candidate_Warnings",
            "Display_Row_Warnings", "All_Warnings", "Representation_Note",
            "Full_Evidence_Text",
        ]
        decomposition = self.hdcs_global_result.get("decomposition")
        nnls_ran = isinstance(decomposition, dict)
        if nnls_ran:
            nnls_status = (
                "Reliable" if decomposition.get("is_reliable", False)
                else "Unreliable"
            )
            nnls_fit_quality = decomposition.get("fit_quality")
        else:
            nnls_status = "Not run"
            nnls_fit_quality = None
        mixture_detection = self.hdcs_global_result.get("mixture_detection", {})
        mixture_candidates = {
            str(key) for key in self.hdcs_global_result.get("mixture_candidates", [])
        }
        records: List[Dict[str, Any]] = []
        for result in self.display_candidate_rows:
            candidate_key = str(
                result.get(
                    "representative_candidate_key",
                    result.get("candidate_key", ""),
                )
            )
            candidate = self.hdcs_candidates.get(candidate_key, {})
            content = self._build_candidate_evidence_content(result, candidate)
            member_keys = list(content["member_keys"])
            fa3_found = bool(content["fa3_found"])
            row_nnls_eligible = bool(result.get("nnls_eligible", False))
            theory_nnls = (
                result.get(
                    "display_nnls_contribution",
                    result.get("nnls_contribution"),
                )
                if nnls_ran and row_nnls_eligible else None
            )
            row_nnls_status = (
                nnls_status
                if row_nnls_eligible
                else "Not eligible (raw HDCS != 1.1)"
            )
            row_nnls_fit_quality = nnls_fit_quality if row_nnls_eligible else None
            records.append(
                {
                    "Display_ID": result.get("display_id"),
                    "Display_Type": content["display_type_text"],
                    "Display_Rank": result.get("display_rank", result.get("rank")),
                    "Reported_Lipid_Name": result.get("reported_lipid_name", ""),
                    "Reported_Common_Name": result.get("reported_common_name", ""),
                    "Theoretical_Candidate_Name": result.get(
                        "theoretical_common_name", candidate.get("name", "")
                    ),
                    "Representative_Candidate_Key": candidate_key,
                    "Representative_Candidate_Name": candidate.get(
                        "name", result.get("candidate_name", "")
                    ),
                    "Member_Candidate_Keys": "; ".join(member_keys),
                    "Member_Candidate_Count": int(len(member_keys)),
                    "Equivalent_Group": result.get(
                        "equivalent_group", candidate.get("equivalent_group", "")
                    ),
                    "Position_Family": result.get("position_family", ""),
                    "Position_Resolution_Status": result.get(
                        "position_resolution_status", ""
                    ),
                    "Ester_Position_Confirmed": not content["is_unresolved"],
                    "Reported_Ester_Position": content["reported_position_text"],
                    "Supported_Ester_Positions": content[
                        "supported_positions_text"
                    ],
                    "Unresolved_Ester_Positions": content[
                        "unresolved_positions_text"
                    ],
                    "Family_Supported_Ester_Positions": content[
                        "family_supported_positions_text"
                    ],
                    "Family_Unsupported_Ester_Positions": content[
                        "family_unsupported_positions_text"
                    ],
                    "L4_Position_Specific_Found_Count": int(
                        result.get("l4_position_specific_found_count", 0) or 0
                    ),
                    "L4_Position_Specific_Total_Count": int(
                        result.get("l4_position_specific_total_count", 0) or 0
                    ),
                    "L4_Evidence_Text": content["l4_evidence_text"],
                    "WDIC": result.get("wdic"),
                    "WDIC_Pass": bool(result.get("wdic_pass", False)),
                    "NEP": result.get("nep"),
                    "Base_Score": result.get(
                        "display_base_score", result.get("base_score")
                    ),
                    "L4_Score": result.get(
                        "display_l4_score", result.get("l4_score")
                    ),
                    "Final_HDCS_Score": result.get("final_score"),
                    "Display_Final_Score": result.get(
                        "display_final_score", result.get("final_score")
                    ),
                    "HDCS_Rank": result.get("display_rank", result.get("rank")),
                    "Theory_NNLS": theory_nnls,
                    "NNLS_Status": row_nnls_status,
                    "NNLS_Fit_Quality": row_nnls_fit_quality,
                    "FA3_Chain": result.get(
                        "fa3_name", candidate.get("fa3_name", "")
                    ),
                    "HFA_Chain": result.get(
                        "hfa_name", candidate.get("hfa_name", "")
                    ),
                    "FA3_Protonated_Theoretical_mz": result.get(
                        "fa3_protonated_theoretical_mz"
                    ),
                    "FA3_Protonated_Status": "Found" if fa3_found else "Missing",
                    "FA3_Protonated_Experimental_mz": content[
                        "fa3_experimental_mz"
                    ],
                    "FA3_Protonated_ppm_Error": content["fa3_ppm_error"],
                    "FA3_Protonated_Intensity": content["fa3_intensity"],
                    "FA3_Protonated_Threshold": result.get(
                        "fa3_protonated_threshold"
                    ),
                    "FA3_Protonated_Rank": content["fa3_rank"],
                    "FA3_Protonated_Is_Strongest": (
                        bool(result.get("fa3_protonated_is_strongest"))
                        if fa3_found else None
                    ),
                    "FA3_Protonated_Ratio_To_Strongest": content["fa3_ratio"],
                    "Strongest_Second_FA3_Ratio": content[
                        "strongest_second_ratio"
                    ],
                    "Matched_Diagnostic_Ion_Count": int(
                        len(content["matched_ions"])
                    ),
                    "Matched_Diagnostic_Ions": (
                        "\n".join(content["matched_lines"]) or "None"
                    ),
                    "Missing_Diagnostic_Ion_Count": int(
                        len(content["missing_ions"])
                    ),
                    "Missing_Diagnostic_Ions": (
                        "\n".join(content["missing_lines"]) or "None"
                    ),
                    "Exclusive_Evidence_Count": int(
                        len(content["exclusive_ions"])
                    ),
                    "Exclusive_Evidence": (
                        "\n".join(content["exclusive_lines"]) or "None"
                    ),
                    "Contradictory_Evidence_Count": int(
                        len(content["contradictions"])
                    ),
                    "Contradictory_Evidence": (
                        "\n".join(content["contradiction_lines"]) or "None"
                    ),
                    "Mixture_Detected": bool(
                        self.hdcs_global_result.get("mixture_detected", False)
                    ),
                    "sn_Mixture_Detected": bool(
                        mixture_detection.get("sn_mixed", False)
                    ),
                    "Ester_Position_Mixture_Detected": bool(
                        mixture_detection.get("ester_mixed", False)
                    ),
                    "Mixture_Interpretation": "\n".join(
                        content["mixture_lines"]
                    ),
                    "Mixture_Candidate": any(
                        key in mixture_candidates for key in member_keys
                    ),
                    "Global_Warnings": (
                        "\n".join(content["global_warning_lines"]) or "None"
                    ),
                    "Candidate_Warnings": (
                        "\n".join(content["candidate_warning_lines"]) or "None"
                    ),
                    "Display_Row_Warnings": (
                        "\n".join(content["display_warning_lines"]) or "None"
                    ),
                    "All_Warnings": (
                        "\n".join(content["warning_lines"]) or "None"
                    ),
                    "Representation_Note": content["representation_note"],
                    "Full_Evidence_Text": content["full_text"],
                }
            )
        return pd.DataFrame(records, columns=columns)

    def _build_mixture_nnls_dataframe(self) -> pd.DataFrame:
        """Export Theory NNLS components, or an explicit Not run record."""
        decomposition = self.hdcs_global_result.get("decomposition")
        if not isinstance(decomposition, dict):
            return pd.DataFrame(
                [{
                    "Status": "Not run",
                    "Run_Reason": "No mixture evidence and no highest Base Score tie",
                    "Candidate_Key": "", "Candidate_Name": "", "Contribution": None,
                    "Matrix_Rank": None, "Candidate_Count": None,
                    "Dimension_Count": None, "Condition_Number": None,
                    "Residual": None, "Residual_RMSE": None, "Fit_Quality": None,
                    "Reliable": None, "Unreliable_Reason": "", "Solver": "",
                }]
            )
        mixture_detected = bool(self.hdcs_global_result.get("mixture_detected", False))
        run_reason = "Mixture evidence" if mixture_detected else "Highest Base Score tie"
        candidate_keys = list(decomposition.get("candidate_keys", []))
        names = list(decomposition.get("candidate_names", []))
        coefficients = dict(decomposition.get("coefficients", {}))
        records = []
        for index, candidate_key in enumerate(candidate_keys):
            records.append(
                {
                    "Status": "Run", "Run_Reason": run_reason,
                    "Candidate_Key": candidate_key,
                    "Candidate_Name": names[index] if index < len(names) else candidate_key,
                    "Contribution": coefficients.get(candidate_key),
                    "Matrix_Rank": decomposition.get("matrix_rank"),
                    "Candidate_Count": decomposition.get("n_candidates"),
                    "Dimension_Count": decomposition.get("n_dimensions"),
                    "Condition_Number": decomposition.get("condition_number"),
                    "Residual": decomposition.get("residual"),
                    "Residual_RMSE": decomposition.get("residual_rmse"),
                    "Fit_Quality": decomposition.get("fit_quality"),
                    "Reliable": decomposition.get("is_reliable"),
                    "Unreliable_Reason": decomposition.get("unreliable_reason", ""),
                    "Solver": decomposition.get("solver", "scipy.optimize.nnls"),
                }
            )
        return pd.DataFrame(records)

    def _build_experimental_spectrum_dataframe(self) -> pd.DataFrame:
        """Export all valid mapped peaks with filtering and match traceability."""
        noise = float(self.run_context.get("noise", 1.0))
        valid = self.raw_spec_df.copy()
        retained_rows = set(
            pd.to_numeric(self.current_spec_df.get("source_row"), errors="coerce")
            .dropna().astype(int).tolist()
            if "source_row" in self.current_spec_df else []
        )
        diagnostic_mzs: List[Tuple[float, str]] = []
        for candidate_key, result in self.hdcs_results.items():
            for ion in result.get("detected_ions", {}).values():
                if ion.get("found") and ion.get("exp_mz") is not None:
                    diagnostic_mzs.append((float(ion["exp_mz"]), str(candidate_key)))
        records = []
        for _index, peak in valid.iterrows():
            mz = float(peak["mz"])
            source_row = int(peak["source_row"])
            matched_candidates = {
                key for matched_mz, key in diagnostic_mzs
                if abs(mz - matched_mz) / max(matched_mz, 1e-12) * 1e6
                <= float(self.run_context.get("hdcs_config", {}).get("ppm_tolerance", 20.0))
            }
            records.append(
                {
                    "Source_Row": source_row, "m/z": mz,
                    "Raw_Intensity": float(peak["intensity"]),
                    "Calculated_S/N": float(peak["intensity"]) / noise,
                    "Retained_After_Filter": source_row in retained_rows,
                    "Precursor_Peak": (
                        source_row
                        == int(self.run_context.get("precursor_source_row", -1))
                        and abs(mz - self.parent_mz_exp)
                        / self.parent_mz_exp * 1e6 <= PPM_PARENT
                    ),
                    "Matched_Any_Diagnostic_Ion": bool(matched_candidates),
                    "Matched_Candidate_Count": len(matched_candidates),
                }
            )
        return pd.DataFrame(records)

    def _collect_export_warnings(self) -> List[Dict[str, str]]:
        """Collect input, generation, HDCS, NNLS, and unresolved-position warnings."""
        records: List[Dict[str, str]] = []
        for warning in self.input_metadata.get("warnings", []):
            records.append({"Warning_Type": "Input", "Candidate": "", "Message": str(warning)})
        for warning in self.candidate_warnings:
            candidate = str(warning).split(" ", 1)[0] if warning else ""
            records.append({"Warning_Type": "Candidate generation", "Candidate": candidate, "Message": str(warning)})
        for warning in self.hdcs_global_result.get("warnings", []):
            records.append({"Warning_Type": "HDCS", "Candidate": "", "Message": str(warning)})
        decomposition = self.hdcs_global_result.get("decomposition")
        if isinstance(decomposition, dict) and not decomposition.get("is_reliable", True):
            records.append({
                "Warning_Type": "NNLS", "Candidate": "",
                "Message": str(decomposition.get("unreliable_reason", "Unreliable NNLS fit")),
            })
        for row in self.display_candidate_rows:
            if row.get("display_type") == "unresolved_positions":
                records.append({
                    "Warning_Type": "Unresolved position",
                    "Candidate": str(row.get("display_id", "")),
                    "Message": "Unresolved ester positions: " + _export_join(row.get("unresolved_ester_positions", [])),
                })
        return records

    def _build_warnings_dataframe(self) -> pd.DataFrame:
        records = self._collect_export_warnings()
        if not records:
            records = [{"Warning_Type": "None", "Candidate": "", "Message": "No warnings"}]
        return pd.DataFrame(records, columns=["Warning_Type", "Candidate", "Message"])

    def _build_run_parameters_payload(self) -> Dict[str, Any]:
        """Build the compact JSON run record from actual completed-run values."""
        config = dict(self.run_context.get("hdcs_config", {}))
        weights = config.get("level_weights", {})
        return to_json_safe(
            {
                "software": {
                    "file": self.run_context.get("software_file"),
                    "version": self.run_context.get("software_version"),
                    "analysis_time": self.run_context.get("analysis_time"),
                },
                "input": {
                    "file": self.run_context.get("input_file"),
                    "delimiter": self.run_context.get("delimiter"),
                    "column_mapping": dict(self.input_column_mapping),
                    "precursor_selection": {
                        "method": self.run_context.get(
                            "precursor_selection_method",
                            "highest_intensity_peak",
                        ),
                        "precursor_mz": self.run_context.get("precursor_mz"),
                        "precursor_intensity": self.run_context.get(
                            "precursor_intensity"
                        ),
                        "source_row": self.run_context.get(
                            "precursor_source_row"
                        ),
                    },
                },
                "spectrum_filter": {
                    "noise": self.run_context.get("noise"),
                    "sn_threshold": self.run_context.get("sn_threshold"),
                    "effective_intensity_cutoff": self.run_context.get("effective_intensity_cutoff"),
                },
                "hdcs": {
                    "ppm_tolerance": config.get("ppm_tolerance"),
                    "l1_threshold": config.get("l1_threshold"),
                    "l2_threshold": config.get("l2_threshold"),
                    "l3_threshold": config.get("l3_threshold"),
                    "l4_threshold": config.get("l4_threshold"),
                    "level_weights": {
                        "L1": weights.get(1, weights.get("1")),
                        "L2": weights.get(2, weights.get("2")),
                        "L3": weights.get(3, weights.get("3")),
                        "L4": weights.get(4, weights.get("4")),
                    },
                    "nep_lambda": config.get("nep_lambda"),
                    "wdic_pass_threshold": config.get("wdic_pass_threshold"),
                    "l4_bonus_max": config.get("l4_bonus_max"),
                    "nnls_min_final_score": config.get("nnls_min_final_score"),
                    "nnls_enabled": config.get("nnls_enabled"),
                    "mixture_detection_enabled": config.get("mixture_detection_enabled"),
                },
            }
        )

    def _format_export_workbook(self, writer: Any) -> None:
        """Apply simple Excel/WPS-compatible headings, filters, widths, and formats."""
        from openpyxl.styles import Alignment, Font
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            if worksheet.max_row >= 1 and worksheet.max_column >= 1:
                worksheet.auto_filter.ref = worksheet.dimensions
            for cell in worksheet[1]:
                cell.font = Font(bold=True)
            for column_cells in worksheet.columns:
                header = str(column_cells[0].value or "")
                max_length = max(
                    len(str(cell.value)) if cell.value is not None else 0
                    for cell in column_cells[: min(len(column_cells), 300)]
                )
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(
                    45, max(10, max_length + 2)
                )
                header_lower = header.lower()
                if "m/z" in header_lower or header_lower.endswith("_mz"):
                    number_format = "0.00000"
                elif "ppm" in header_lower:
                    number_format = "0.00"
                elif any(token in header_lower for token in ("score", "contribution", "fit_quality", "wdic", "nep")):
                    number_format = "0.0000"
                else:
                    number_format = None
                if number_format:
                    for cell in column_cells[1:]:
                        if isinstance(cell.value, (int, float)) and not isinstance(cell.value, bool):
                            cell.number_format = number_format
            if worksheet.title == "Evidence_Summary":
                long_text_widths = {
                    "Matched_Diagnostic_Ions": 55,
                    "Missing_Diagnostic_Ions": 55,
                    "Exclusive_Evidence": 48,
                    "Contradictory_Evidence": 48,
                    "Mixture_Interpretation": 48,
                    "Global_Warnings": 45,
                    "Candidate_Warnings": 45,
                    "Display_Row_Warnings": 45,
                    "All_Warnings": 55,
                    "Representation_Note": 65,
                    "Full_Evidence_Text": 80,
                }
                header_columns = {
                    str(cell.value or ""): cell.column_letter
                    for cell in worksheet[1]
                }
                for header, width in long_text_widths.items():
                    column_letter = header_columns.get(header)
                    if column_letter:
                        worksheet.column_dimensions[column_letter].width = width
                for row in worksheet.iter_rows(min_row=2):
                    for cell in row:
                        cell.alignment = Alignment(
                            wrap_text=True,
                            vertical="top",
                        )

    def _ensure_candidate_spectrum_for_export(self) -> bool:
        """Ensure the figure represents a current or highest-ranked candidate."""
        if self.predicted_peaks and not self.current_spec_df.empty:
            self._draw_butterfly_chart_fast()
            return True
        visible = self.tree_left.get_children() if hasattr(self, "tree_left") else ()
        if visible:
            self.tree_left.selection_set(visible[0])
            self._select_candidate_hdcs()
            return bool(self.predicted_peaks)
        if not self.display_candidate_rows or self.current_spec_df.empty:
            return False
        top_result = min(
            self.display_candidate_rows,
            key=lambda row: int(row.get("display_rank", 10**9)),
        )
        candidate_key = str(top_result.get("representative_candidate_key", ""))
        peaks = self.candidate_display_peaks.get(candidate_key)
        if not peaks:
            return False
        self.predicted_peaks = [dict(peak) for peak in peaks]
        self.raw_predicted_peaks = [
            dict(peak) for peak in self.candidate_raw_peaks.get(candidate_key, [])
        ]
        candidate = self.hdcs_candidates.get(candidate_key, {})
        self.generator._last_fahfa_sn = "sn2" if candidate.get("sn_fahfa") == "sn-2" else "sn1"
        self._draw_butterfly_chart_fast()
        return True

    def _build_analysis_log(self) -> str:
        warnings = self._collect_export_warnings()
        decomposition = self.hdcs_global_result.get("decomposition")
        evidence_summary = self._build_evidence_summary_dataframe()
        if not evidence_summary.empty:
            ordered_evidence = evidence_summary.sort_values(
                by=["Display_Rank", "Display_ID"],
                kind="stable",
                na_position="last",
            )
            top_evidence = ordered_evidence.iloc[0]
            top_candidate = (
                top_evidence.get("Reported_Lipid_Name")
                or top_evidence.get("Display_ID")
                or ""
            )
            top_position_status = top_evidence.get(
                "Position_Resolution_Status", ""
            )
            top_matched_count = int(
                top_evidence.get("Matched_Diagnostic_Ion_Count", 0) or 0
            )
            top_missing_count = int(
                top_evidence.get("Missing_Diagnostic_Ion_Count", 0) or 0
            )
            top_exclusive_count = int(
                top_evidence.get("Exclusive_Evidence_Count", 0) or 0
            )
            top_contradictory_count = int(
                top_evidence.get("Contradictory_Evidence_Count", 0) or 0
            )
        else:
            top_candidate = ""
            top_position_status = ""
            top_matched_count = 0
            top_missing_count = 0
            top_exclusive_count = 0
            top_contradictory_count = 0
        lines = [
            f"Analysis start time: {self.run_context.get('analysis_start_time', '')}",
            f"Analysis completion time: {self.run_context.get('analysis_time', '')}",
            f"Input file: {self.run_context.get('input_file', '')}",
            f"Column mapping: {json.dumps(self.input_column_mapping, ensure_ascii=False)}",
            f"m/z source column: {self.input_column_mapping.get('mz', '')}",
            f"Intensity source column: {self.input_column_mapping.get('intensity', '')}",
            "Precursor selection method: Highest-intensity peak",
            f"Precursor m/z: {self.run_context.get('precursor_mz', '')}",
            f"Precursor intensity: {self.run_context.get('precursor_intensity', '')}",
            f"Precursor source row: {self.run_context.get('precursor_source_row', '')}",
            f"Matched composition: {self.run_context.get('matched_composition', '')}",
            f"Original peak count: {self.run_context.get('original_row_count', 0)}",
            f"Retained peak count: {self.run_context.get('retained_peak_count', 0)}",
            f"Candidate count: {self.run_context.get('candidate_count', 0)}",
            f"Displayed candidate count: {self.run_context.get('display_candidate_count', 0)}",
            f"Mixture detected: {bool(self.hdcs_global_result.get('mixture_detected', False))}",
            f"NNLS status: {'Run' if isinstance(decomposition, dict) else 'Not run'}",
            f"Warning count: {len(warnings)}",
            f"Top displayed candidate: {top_candidate}",
            f"Top candidate position resolution: {top_position_status}",
            f"Top candidate matched diagnostic ion count: {top_matched_count}",
            f"Top candidate missing diagnostic ion count: {top_missing_count}",
            f"Top candidate exclusive evidence count: {top_exclusive_count}",
            f"Top candidate contradictory evidence count: {top_contradictory_count}",
            f"Evidence summary rows exported: {len(evidence_summary)}",
            *(
                [
                    "Multiple peaks shared the maximum intensity. "
                    "The highest-m/z peak was selected."
                ]
                if int(self.run_context.get("precursor_tie_count", 0) or 0) > 1
                else []
            ),
            "",
            "Warnings:",
        ]
        if warnings:
            lines.extend(
                f"- [{item['Warning_Type']}] {item['Candidate']}: {item['Message']}"
                for item in warnings
            )
        else:
            lines.append("- No warnings")
        return "\n".join(lines) + "\n"

    def _export_analysis_results(self) -> None:
        """Export main results plus the separate Tier III manual-review list."""
        if not self.analysis_completed or not self.run_context or not self.hdcs_results:
            messagebox.showwarning(
                "Export Results", "Run a successful analysis before exporting results."
            )
            return
        input_stem = sanitize_filename(
            os.path.splitext(os.path.basename(self.run_context["input_file"]))[0]
        )
        precursor_text = f"{float(self.run_context['precursor_mz']):.4f}"
        output_base = sanitize_filename(
            f"{input_stem}_TG-FAHFA_HDCS_{precursor_text}"
        )
        workbook_path = filedialog.asksaveasfilename(
            title="Save Excel Results",
            defaultextension=".xlsx",
            initialfile=output_base + ".xlsx",
            filetypes=[("Excel workbook", "*.xlsx")],
        )
        if not workbook_path:
            return

        try:
            import openpyxl  # noqa: F401 - verifies the required export engine
            sheets = {
                "Parameters": self._build_parameters_dataframe(),
                "Candidate_Ranking": self._build_candidate_ranking_dataframe(),
                "Manual_Review": self._build_manual_review_dataframe(),
                "Diagnostic_Ions": self._build_diagnostic_ions_dataframe(),
            }
            with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
                for sheet_name, frame in sheets.items():
                    frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                self._format_export_workbook(writer)
        except ImportError:
            messagebox.showerror(
                "Export Results",
                "Excel export requires openpyxl. Install it with: pip install openpyxl",
            )
            return
        except Exception as exc:
            messagebox.showerror("Export Results", f"Excel export failed:\n{exc}")
            return

        messagebox.showinfo(
            "Export Results",
            f"Excel file exported successfully:\n{workbook_path}",
        )

    def _on_sort_changed(self, event=None):
        self._on_sort_changed_hdcs(event)
        
    def _on_sort_changed_hdcs(self, event=None):
        if self.candidates_df.empty or not self.hdcs_results:
            return
        self._refresh_candidate_tree(preserve_selection=True)

    def _on_candidate_select(self, event=None):
        self._select_candidate_hdcs(event)

    def _select_candidate_hdcs(self, event=None):
        if not self.tree_left.selection() or self.candidates_df.empty:
            return
        display_id = str(self.tree_left.selection()[0])
        display_result = self.display_row_by_id.get(display_id)
        if display_result is None:
            return
        candidate_id = str(
            display_result.get("representative_candidate_key", display_result.get("candidate_key", ""))
        )
        index = self.candidate_index_by_key.get(candidate_id)
        if index is None:
            return
        row_data = self.candidates_df.iloc[index]
        if candidate_id not in self.candidate_display_peaks:
            return
        sn1 = str(row_data["FA_Sn1"]).strip('="')
        sn2 = str(row_data["FA_Sn2"]).strip('="')
        sn3 = str(row_data["FA_Sn3"]).strip('="')
        self.smiles_var.set(str(row_data["SMILES"]))
        common_name = display_result.get("reported_common_name")
        if not common_name:
            common_name = format_common_name_for_display(
                f"TG({get_accurate_abbr(sn1)}/{get_accurate_abbr(sn2)}/{get_accurate_abbr(sn3)})"
            )
        self.common_name_var.set(str(common_name))
        self.predicted_peaks = [dict(peak) for peak in self.candidate_display_peaks[candidate_id]]
        self.raw_predicted_peaks = [dict(peak) for peak in self.candidate_raw_peaks[candidate_id]]
        candidate = self.hdcs_candidates.get(candidate_id, {})
        self.generator._last_fahfa_sn = "sn2" if candidate.get("sn_fahfa") == "sn-2" else "sn1"
        for item in self.tree_right.get_children():
            self.tree_right.delete(item)
        for peak_index, peak in enumerate(self.predicted_peaks):
            is_key = bool(peak.get("is_key_ion", False))
            peak_type = str(peak.get("type", ""))
            display_level = fragment_levels_display(peak.get("levels", []), peak_type)
            tags = spectrum_level_tag(peak.get("levels", []))
            self.tree_right.insert(
                "", "end", iid=str(peak_index),
                values=(
                    f"{peak['mz']:.4f}", f"{peak['intensity']:.1f}",
                    "★" if is_key else "", display_level, peak.get("annotation", ""),
                ),
                tags=tags,
            )
        self._draw_butterfly_chart_fast()
        self._update_hdcs_analysis(display_id)

    def _draw_butterfly_chart_fast(self):
        self.ax.clear()
        self.ax.axhline(0, color='gray', linewidth=1)
        self.ax.set_axis_on()
        
        orig_mzs = self.current_spec_df['mz'].values
        orig_ints_raw = self.current_spec_df['intensity'].values
        max_orig_int = orig_ints_raw.max() if len(orig_ints_raw) > 0 else 100
        norm_orig_ints = (orig_ints_raw / max_orig_int) * 100
        
        y_vals_real = norm_orig_ints * 5
        ppm_errors_parent = np.abs(orig_mzs - self.parent_mz_exp) / self.parent_mz_exp * 1e6
        parent_mask = ppm_errors_parent <= 20
        y_vals_real[parent_mask] = norm_orig_ints[parent_mask] 
        y_vals_real = np.clip(y_vals_real, 0, 100)
        
        colors_upper, lws_upper = np.full(len(orig_mzs), "#636E72", dtype=object), np.full(len(orig_mzs), 1.0)
        matched_peaks_info = []
        for p in self.predicted_peaks:
            theo_mz = p['mz']
            ppm_errors = np.abs(orig_mzs - theo_mz) / theo_mz * 1e6
            valid_matches = np.where(ppm_errors <= 20)[0] 
            if len(valid_matches) > 0:
                best_idx = valid_matches[np.argmax(orig_ints_raw[valid_matches])]
                colors_upper[best_idx], lws_upper[best_idx] = "black", 2.0
                matched_peaks_info.append({'mz': orig_mzs[best_idx], 'y': y_vals_real[best_idx]})
                
        self.ax.vlines(orig_mzs, 0, y_vals_real, colors=colors_upper.tolist(), linewidths=lws_upper.tolist())
        matched_peaks_info.sort(key=lambda x: x['mz'])
        for i, info in enumerate(matched_peaks_info):
            self.ax.text(
                info['mz'], info['y'] + 8, f"{info['mz']:.4f}",
                rotation=90, va='bottom', ha='center', fontsize=8,
                color="black", fontweight='bold', clip_on=True,
            )

        pred_mzs = np.array([p["mz"] for p in self.predicted_peaks])
        pred_ints_raw = np.array([p["intensity"] for p in self.predicted_peaks])
        max_pred_int = pred_ints_raw.max() if len(pred_ints_raw) > 0 else 100
        pred_ints = - (pred_ints_raw / max_pred_int) * 100 
        
        colors_lower, lws_lower = [], []
        for p in self.predicted_peaks:
            ptype = p.get("type", "")
            is_key = p.get("is_key_ion", False)
            col = spectrum_level_color(p.get("levels", []))
            type_names = [part.strip() for part in str(ptype).split('/') if part.strip()]
            if is_key or "ead_diagnostic_ester" in type_names or "ead_sn2" in type_names:
                lw = 2.0
            elif any(type_name in NEUTRAL_FRAGMENT_TYPES for type_name in type_names):
                lw = 1.5
            else:
                lw = 1.2
            colors_lower.append(col); lws_lower.append(lw)
            
        self.ax.vlines(pred_mzs, 0, pred_ints, colors=colors_lower, linewidths=lws_lower)
        for mz, int_val, col in zip(pred_mzs, pred_ints, colors_lower):
            self.ax.text(
                mz, int_val - 3, f"{mz:.4f}", rotation=90,
                va='top', ha='center', fontsize=7, color=col, clip_on=True,
            )

        self.ax.text(0.02, 0.95, "Experimental EAciD-MS/MS Spectrum", transform=self.ax.transAxes, fontsize=11, va='top')
        self.ax.text(0.02, 0.05, "Predicted EAciD-MS/MS Spectrum", transform=self.ax.transAxes, fontsize=11, va='bottom')
        
        x_min, x_max = min(orig_mzs.min(), pred_mzs.min()) - 50, max(orig_mzs.max(), pred_mzs.max()) + 50

        self.ax.set_ylim(-125, 125); self.ax.set_xlim(x_min, x_max)
        self.ax.set_yticks([-100, -50, 0, 50, 100])
        self.ax.set_yticklabels([100, 50, 0, 50, 100])
        self.ax.spines['top'].set_visible(False); self.ax.spines['right'].set_visible(False); self.ax.spines['bottom'].set_visible(False)
        self.ax.tick_params(axis='x', direction='inout', pad=15)
        self._remember_spectrum_home_view()
        self.canvas.draw_idle()

#############################################################################
#  MAIN ENTRY POINT
#############################################################################
if __name__ == "__main__":
    app = IntegratedApp()
    app.mainloop()
