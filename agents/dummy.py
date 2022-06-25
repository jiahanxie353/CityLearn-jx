import os
import sys
sys.path.insert(0,'..')
from citylearn import  CityLearn
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from agents.sac import SAC as Agent


climate_zone = 5
data_path = Path("/Users/xiejiahan/PycharmProjects/RLinDR/CityLearn/data/Climate_Zone_"+str(climate_zone))
sim_period = (0, 8760*4-1)
building_ids = ["Building_"+str(i) for i in [1,2,3,4,5,6,7,8,9]]
params = {'data_path':data_path,
        'building_attributes':'building_attributes.json',
        'weather_file':'weather_data.csv',
        'solar_profile':'solar_generation_1kW.csv',
        'carbon_intensity':'carbon_intensity.csv',
        'building_ids':building_ids,
        'buildings_states_actions':'/Users/xiejiahan/PycharmProjects/RLinDR/CityLearn/buildings_state_action_space.json',
        'simulation_period': sim_period,
        'cost_function': ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption','carbon_emissions'],
        'central_agent': False,
        'save_memory': False }

env = CityLearn(**params)
observations_spaces, actions_spaces = env.get_state_action_spaces()

building_info = env.get_building_information()

params_agent = {'building_ids':building_ids,
                 'buildings_states_actions':os.path.join('/Users/xiejiahan/PycharmProjects/RLinDR/CityLearn/buildings_state_action_space.json'),
                 'building_info':building_info,
                 'observation_spaces':observations_spaces,
                 'action_spaces':actions_spaces}

state = env.reset()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ddpg')
    args = parser.parse_args()
    agents = Agent(**params_agent)
    print(agents.action_spaces)