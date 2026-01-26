from typing import Union
from navier_stokes import navier_stokes_simulation
from navier_stokes_GPU import NavierStokesGPU
from navier_stokes_GPU2 import NavierStokesGPU2
import pickle
import os

base_path = "/home/gandalf/navier-stokes-project/"


# funcs
default_settings = {
    # constants
    "tau": 1,
    "omega": 1.7,
    "epsilon": 0.01,
    "x_vel": 2,
    "N_max_P": 100,
    # variables:
    "nx": 60,
    "ny": 60,
    "len_x": 1,
    "len_y": 1,
    "Re": 4000,
    "T_max": 2,
    "type_": "lid",
    "boxes": [],  # list of boxes defined by [start_x, end_x, start_y, end_y] in grid indices
    "circle": [],  # list of circles defined by [x_mid, y_mid, radius] in grid indices    
    "addon": "",  # for filename uniqueness
    "solver": "GPU2", # Literal["CPU", "GPU", "GPU2"]
    "x_vel_type": "constant", # Literal["constant", "sinus"]
}

def settings() -> dict:
    """Return a copy of the default simulation settings."""
    return default_settings.copy()

def filename_from_settings(settings: dict, subfolder: str = "") -> str:
    """Generate a filename based on simulation settings."""
    filename = (f"{settings['type_']}{settings['addon']}_Re{settings['Re']}_"
                f"{settings['nx']}x{settings['ny']}_T{int(settings['T_max']*1000)}.pkl")
    if not os.path.exists(os.path.join(base_path, "data", subfolder)):
        os.makedirs(os.path.join(base_path, "data", subfolder))
    return os.path.join(base_path, "data", subfolder, filename)

def NV_from_settings(settings: dict) -> Union[navier_stokes_simulation, NavierStokesGPU, NavierStokesGPU2]:
    if settings["solver"] == "CPU":
        NV_solver = navier_stokes_simulation
    elif settings["solver"] == "GPU":
        NV_solver = NavierStokesGPU
    elif settings["solver"] == "GPU2":
        NV_solver = NavierStokesGPU2
    else:
        raise ValueError(f"Unknown solver type: {settings['solver']}")
    NV_simulation = NV_solver(
        xn=settings["nx"],
        yn=settings["ny"],
        len_x=settings["len_x"],
        len_y=settings["len_y"],
        Re=settings["Re"],
        tau=settings["tau"],
        omega=settings["omega"],
        epsilon=settings["epsilon"],
        x_vel=settings["x_vel"],
        x_vel_type=settings["x_vel_type"],
    )
    NV_simulation.set_boundary_type_and_boxes(settings["type_"], settings["boxes"], settings["circle"])
    NV_simulation.iterate(t_end=settings["T_max"], N_max_P=settings["N_max_P"], max_histories=1000)  # type: ignore
    return NV_simulation

def load_data(filename: str) -> Union[navier_stokes_simulation, NavierStokesGPU, NavierStokesGPU2]:
    """Load a simulation from a file."""
    with open(filename, "rb") as f:
        simulation = pickle.load(f)
        print(f"Simulation loaded from {filename}, type: {type(simulation)}")
    return simulation