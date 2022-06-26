import os
import sys
sys.path.insert(0,'..')
from citylearn import CityLearn
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from agents.DDPG.ddpg import DDPG as Agent


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

    agents = Agent(**params_agent)
    done = False

    action = agents.select_action(state,0.2)
    for i in range(1,25):
        next_state, reward, done, _ = env.step(action)
        action_next = agents.select_action(next_state,0.2)
        agents.add_to_buffer(state, action, reward, next_state, done)
        state = next_state
        action = action_next

    print(env.cost())
    # print(agents.critic_net)
    print(agents.action_spaces)