import torch

z_what_dim = 64
z_where_scale_dim = 2  # sx sy
z_where_shift_dim = 2  # tx ty
z_pres_dim = 1
glimpse_size = 64
img_h = 128
img_w = img_h
img_encode_dim = 64
z_depth_dim = 1
bg_what_dim = 10

temporal_rnn_hid_dim = 128
temporal_rnn_out_dim = temporal_rnn_hid_dim
propagate_encode_dim = 32
z_where_transit_bias_net_hid_dim = 128
z_depth_transit_net_hid_dim = 128

z_pres_hid_dim = 64
z_what_from_temporal_hid_dim = 64
z_what_enc_dim = 128

prior_rnn_hid_dim = 64
prior_rnn_out_dim = prior_rnn_hid_dim

DEBUG = False

object_act_size = 21
seq_len = 10
phase_obj_num_contrain = True
phase_rejection = True

temporal_img_enc_hid_dim = 64
temporal_img_enc_dim = 128
z_where_bias_dim = 4
temporal_rnn_inp_dim = 128
prior_rnn_inp_dim = 128
bg_prior_rnn_hid_dim = 32
where_update_scale = 0.2

pres_logit_factor = 8.8

conv_lstm_hid_dim = 64

cfg = {
    "num_img_summary": 3,
    "num_cell_h": 8,
    "num_cell_w": 8,
    "phase_conv_lstm": True,
    "phase_no_background": False,
    "phase_eval": True,
    "phase_boundary_loss": False,
    "phase_generate": False,
    "phase_nll": False,
    "phase_gen_disc": True,
    "gen_disc_pres_probs": 0.1,
    "observe_frames": 5,
    "size_anc": 0.1,
    "var_s": 0.03,
    "ratio_anc": 2.5,
    "var_anc": 0.5,
    "train_station_cropping_origin": 240,
    "color_num": 700,
    "explained_ratio_threshold": 0.3,
    "tau_imp": 0.25,
    "z_pres_anneal_end_value": 1e-3,
    "phase_do_remove_detach": True,
    "remove_detach_step": 30000,
    "max_num_obj": 45,  # Remove this constrain in discovery.py if you have enough GPU memory.
}


suitable_video_list_only_from_top = [
    2,
    10,
    19,
    28,
    33,
    34,
    42,
    43,
    50,
    52,
    59,
    60,
    62,
    68,
    69,
    73,
    77,
    78,
    84,
    86,
    87,
    90,
    94,
    96,
    103,
    104,
    112,
    113,
    121,
    122,
    130,
    138,
    147,
    156,
    161,
    162,
    170,
    171,
    178,
    180,
    187,
    188,
    196,
    197,
]

suitable_video_list = [
    1,
    2,
    7,
    8,
    9,
    10,
    12,
    16,
    17,
    18,
    19,
    23,
    25,
    26,
    27,
    28,
    33,
    34,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    55,
    56,
    57,
    58,
    59,
    60,
    62,
    64,
    65,
    66,
    67,
    68,
    69,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    81,
    82,
    83,
    85,
    86,
    87,
    90,
    91,
    92,
    93,
    96,
    99,
    100,
    101,
    102,
    103,
    104,
    108,
    109,
    110,
    111,
    112,
    113,
    116,
    117,
    118,
    119,
    120,
    121,
    122,
    125,
    126,
    127,
    128,
    129,
    130,
    135,
    136,
    137,
    138,
    140,
    144,
    145,
    146,
    147,
    151,
    153,
    154,
    155,
    156,
    161,
    162,
    165,
    166,
    167,
    168,
    169,
    170,
    171,
    174,
    175,
    176,
    177,
    178,
    179,
    180,
    183,
    184,
    185,
    186,
    187,
    188,
    190,
    192,
    193,
    194,
    195,
    196,
    197,
    200,
]

top_down_list = [
    19,
    28,
    33,
    42,
    59,
    60,
    68,
    73,
    86,
    94,
    103,
    112,
    121,
    130,
    147,
    161,
    170,
    178,
    187,
]

top_down_list = [28, 33, 42, 59, 60, 68, 73, 86]

# top_down_list = [33, 42, 59, 60]

# top_down_list = [60]
video_id_embs = {
    "028": torch.tensor(
        [0.5698, 0.4523, 0.0439, 0.2748, 0.6137, 0.7246, 0.6272, 0.7679, 0.0314, 0.7192]
    ),
    "033": torch.tensor(
        [0.3194, 0.7520, 0.0468, 0.1089, 0.3940, 0.0794, 0.8959, 0.4173, 0.8779, 0.1524]
    ),
    "042": torch.tensor(
        [0.7498, 0.8630, 0.4793, 0.3743, 0.9638, 0.0117, 0.2004, 0.5346, 0.1427, 0.7409]
    ),
    "059": torch.tensor(
        [0.4577, 0.9213, 0.9728, 0.2543, 0.1908, 0.1222, 0.6117, 0.9399, 0.3682, 0.0634]
    ),
    "060": torch.tensor(
        [0.0416, 0.6504, 0.6510, 0.1798, 0.1896, 0.4336, 0.1933, 0.5292, 0.7736, 0.6729]
    ),
    "068": torch.tensor(
        [0.3736, 0.4585, 0.8045, 0.1315, 0.6613, 0.8539, 0.3696, 0.3114, 0.0410, 0.8416]
    ),
    "073": torch.tensor(
        [0.0547, 0.0443, 0.4126, 0.3588, 0.4887, 0.5848, 0.3670, 0.9690, 0.9729, 0.0097]
    ),
    "086": torch.tensor(
        [0.0956, 0.6294, 0.2731, 0.4732, 0.8368, 0.2797, 0.9791, 0.6993, 0.5390, 0.4438]
    ),
}

suitable_video_list = [str(a).zfill(3) for a in suitable_video_list]
suitable_video_list_only_from_top = [
    str(a).zfill(3) for a in suitable_video_list_only_from_top
]
top_down_list = [str(a).zfill(3) for a in top_down_list]
