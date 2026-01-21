import pandas as pd
from dataclasses import dataclass, fields
import const 
import json 
import numpy as np 
from typing import List 

@dataclass 
class Params:
    r_e: float
    c_bulk: float
    E_i: float
    E_v: float
    T: float
    dA: float
    Dref: float
    ERef: float

@dataclass 
class DerivedParams:
    theta_i: float 
    theta_v: float
    dA: float 

@dataclass
class Experiment:
    scan_rate: float
    sigma: float 
    pot_flux: pd.DataFrame

def read_experiments(path, name, scan_rates):
    params = Params(**json.load(open(f"./{path}/{name}/params.json", "r")))
    coef = (const.F/(const.R*params.T))
    sigma_coef = (params.r_e**2/params.Dref)*coef
    derived_params = DerivedParams(
        theta_i = (params.E_i - params.ERef)*coef,
        theta_v = (params.E_v - params.ERef)*coef,
        dA = params.dA/params.Dref
    )
    experiments = []
    for scan_rate in scan_rates:
        pot_flux = pd.read_excel(f"./{path}/{name}/cv.xlsx", sheet_name=scan_rate)
        pot_flux.iloc[:,0] = (pot_flux.iloc[:,0] - params.ERef) * coef
        pot_flux.iloc[:,1] = (pot_flux.iloc[:,1])/(const.F*np.pi* params.r_e**2 * params.c_bulk*params.Dref/params.r_e)
        experiments.append(Experiment(
            scan_rate = scan_rate,
            sigma = sigma_coef*float(scan_rate)*(10**-2),
            pot_flux = pot_flux.rename({"Potential(V)":"Potential","Current(I)":"Flux"},axis=1)
        ))
    return params, derived_params, experiments 
