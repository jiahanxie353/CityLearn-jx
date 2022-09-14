import numpy as np
from maac.utils.misc import PeriodicNorm, OnehotEncode, RemoveFeature, Normalize, NoNorm
from collections import namedtuple


Constants = namedtuple('Constants', ['diffuse_solar_rad', 'direct_solar_rad_pred_24h', 'solar_gen', 'dhw_storage_soc',
                                     'cooling_storage_soc', 'non_shiftable_load'])
constants = Constants(12, 20, 24, 26, 25, 23)


def encode(env):
    """
    Encode the states variables
    :param env: CityLearn environment
    :return:
    """
    encoder, encoder_reg, state_dim = {}, {}, {}
    building_info = env.get_building_information()
    building_ids = list(building_info.keys())
    observation_spaces = {uid: o_space for uid, o_space in zip(building_ids, env.observation_spaces)}

    for uid in building_ids:
        encoder[uid] = []
        state_n = 0
        for s_name, s in env.buildings_states_actions[uid]['states'].items():
            if not s:
                encoder[uid].append(0)
            elif s_name in ["month", "hour"]:
                encoder[uid].append(PeriodicNorm(observation_spaces[uid].high[state_n]))
                state_n += 1
            elif s_name == "day":
                encoder[uid].append(OnehotEncode([1, 2, 3, 4, 5, 6, 7, 8]))
                state_n += 1
            elif s_name == "daylight_savings_status":
                encoder[uid].append(OnehotEncode([0, 1]))
                state_n += 1
            elif s_name == "net_electricity_consumption":
                encoder[uid].append(RemoveFeature())
                state_n += 1
            else:
                encoder[uid].append(Normalize(observation_spaces[uid].low[state_n],
                                              observation_spaces[uid].high[state_n]))
                state_n += 1

        encoder[uid] = np.array(encoder[uid])

        # If there is no solar PV installed, remove solar radiation variables
        if building_info[uid]['solar_power_capacity (kW)'] == 0:
            for k in range(constants.diffuse_solar_rad, constants.direct_solar_rad_pred_24h):
                if encoder[uid][k] != 0:
                    encoder[uid][k] = -1
            if encoder[uid][constants.solar_gen] != 0:
                encoder[uid][constants.solar_gen] = -1
        if building_info[uid]['Annual_DHW_demand (kWh)'] == 0 and encoder[uid][constants.dhw_storage_soc] != 0:
            encoder[uid][constants.dhw_storage_soc] = -1
        if building_info[uid]['Annual_cooling_demand (kWh)'] == 0 and encoder[uid][constants.cooling_storage_soc] != 0:
            encoder[uid][constants.cooling_storage_soc] = -1
        if building_info[uid]['Annual_nonshiftable_electrical_demand (kWh)'] == 0 and \
                encoder[uid][constants.non_shiftable_load] != 0:
            encoder[uid][constants.non_shiftable_load] = -1

        encoder[uid] = encoder[uid][encoder[uid] != 0]
        encoder[uid][encoder[uid] == -1] = RemoveFeature()

        # Defining the encoder that will transform the states used by the regression model to predict the
        # net-electricity consumption
        encoder_reg[uid] = []
        state_n = 0
        for s_name, s in env.buildings_states_actions[uid]['states'].items():
            if not s:
                encoder_reg[uid].append(0)
            elif s_name in ["month", "hour"]:
                encoder_reg[uid].append(PeriodicNorm(observation_spaces[uid].high[state_n]))
                state_n += 1
            elif s_name in ["t_out_pred_6h", "t_out_pred_12h", "t_out_pred_24h", "rh_out_pred_6h",
                            "rh_out_pred_12h", "rh_out_pred_24h", "diffuse_solar_rad_pred_6h",
                            "diffuse_solar_rad_pred_12h", "diffuse_solar_rad_pred_24h", "direct_solar_rad_pred_6h",
                            "direct_solar_rad_pred_12h", "direct_solar_rad_pred_24h"]:
                encoder_reg[uid].append(RemoveFeature())
                state_n += 1
            else:
                encoder_reg[uid].append(NoNorm())
                state_n += 1

        encoder_reg[uid] = np.array(encoder_reg[uid])

        # If there is no solar PV installed, remove solar radiation variables
        if building_info[uid]['solar_power_capacity (kW)'] == 0:
            for k in range(12, 20):
                if encoder_reg[uid][k] != 0:
                    encoder_reg[uid][k] = -1
            if encoder_reg[uid][24] != 0:
                encoder_reg[uid][24] = -1
        if building_info[uid]['Annual_DHW_demand (kWh)'] == 0 and encoder_reg[uid][26] != 0:
            encoder_reg[uid][26] = -1
        if building_info[uid]['Annual_cooling_demand (kWh)'] == 0 and encoder_reg[uid][25] != 0:
            encoder_reg[uid][25] = -1
        if building_info[uid]['Annual_nonshiftable_electrical_demand (kWh)'] == 0 and encoder_reg[uid][23] != 0:
            encoder_reg[uid][23] = -1

        encoder_reg[uid] = encoder_reg[uid][encoder_reg[uid] != 0]
        encoder_reg[uid][encoder_reg[uid] == -1] = RemoveFeature()

        state_dim[uid] = 1 + len(
            [j for j in np.hstack(encoder[uid] * np.ones(len(observation_spaces[uid].low))) if
             j is not None])

    return encoder, encoder_reg, state_dim


def normalize(normed, normalizer):
    norm_mean = normalizer[0]
    norm_std = normalizer[1]
    normed = (normed - norm_mean) / norm_std
    return normed
