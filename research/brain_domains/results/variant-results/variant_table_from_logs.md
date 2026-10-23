# 阶段 D 变体表（由 Job 日志重建）

## 行

[
  {
    "row": "cbramod",
    "n": 10,
    "r": 0.0105,
    "p": 0.0225,
    "n_sig": 7,
    "acc": 0.0055,
    "null": 0.005,
    "ratio": 1.0999999999999999
  },
  {
    "row": "v2_mirepnet_ffn",
    "n": 10,
    "r": 0.003,
    "p": 0.7435,
    "n_sig": 0,
    "acc": 0.00465,
    "null": 0.0051,
    "ratio": 0.9117647058823528
  },
  {
    "row": "v3_braintok_code",
    "n": 10,
    "r": 0.0149,
    "p": 0.01,
    "n_sig": 8,
    "acc": 0.005549999999999999,
    "null": 0.0051,
    "ratio": 1.088235294117647
  },
  {
    "row": "v1_sjepa_l0",
    "n": 10,
    "r": 0.03415,
    "p": 0.005,
    "n_sig": 9,
    "acc": 0.0056500000000000005,
    "null": 0.005050000000000001,
    "ratio": 1.1188118811881187
  },
  {
    "row": "v4_s1a_recon",
    "n": 10,
    "r": 0.0288,
    "p": 0.005,
    "n_sig": 10,
    "acc": 0.006200000000000001,
    "null": 0.00485,
    "ratio": 1.2783505154639176
  },
  {
    "row": "v2_mirepnet_ffn_shuf",
    "n": 10,
    "r": -0.0021999999999999997,
    "p": 0.672,
    "n_sig": 0,
    "acc": 0.00485,
    "null": 0.00515,
    "ratio": 0.941747572815534
  },
  {
    "row": "v3_braintok_code_shuf",
    "n": 10,
    "r": 0.0028,
    "p": 0.512,
    "n_sig": 0,
    "acc": 0.00515,
    "null": 0.0049,
    "ratio": 1.0510204081632653
  },
  {
    "row": "v1_sjepa_l0_shuf",
    "n": 10,
    "r": 0.00565,
    "p": 0.184,
    "n_sig": 2,
    "acc": 0.005350000000000001,
    "null": 0.0051,
    "ratio": 1.0490196078431373
  },
  {
    "row": "v4_s1a_recon_shuf",
    "n": 10,
    "r": 0.00255,
    "p": 0.40800000000000003,
    "n_sig": 1,
    "acc": 0.00465,
    "null": 0.005,
    "ratio": 0.9299999999999999
  }
]

## 供体贡献（配对差）

[
  {
    "variant": "v2_mirepnet_ffn",
    "control": "v2_mirepnet_ffn_shuf",
    "n": 10,
    "d_r_median": 0.0073,
    "d_r_n_pos": 6,
    "d_r_p": 0.376953125,
    "d_acc_median": -0.0002999999999999999,
    "d_acc_n_pos": 3,
    "d_acc_p": 0.9453125,
    "d_r_vs_host_median": -0.0074,
    "d_r_vs_host_n_pos": 2,
    "d_r_vs_host_p": 0.9892578125,
    "d_acc_vs_host_median": -0.0007999999999999999,
    "d_acc_vs_host_n_pos": 2,
    "d_acc_vs_host_p": 0.9892578125
  },
  {
    "variant": "v3_braintok_code",
    "control": "v3_braintok_code_shuf",
    "n": 10,
    "d_r_median": 0.01225,
    "d_r_n_pos": 9,
    "d_r_p": 0.0107421875,
    "d_acc_median": 0.0007999999999999999,
    "d_acc_n_pos": 7,
    "d_acc_p": 0.171875,
    "d_r_vs_host_median": 0.0056500000000000005,
    "d_r_vs_host_n_pos": 7,
    "d_r_vs_host_p": 0.171875,
    "d_acc_vs_host_median": 0.0002500000000000002,
    "d_acc_vs_host_n_pos": 6,
    "d_acc_vs_host_p": 0.376953125
  },
  {
    "variant": "v1_sjepa_l0",
    "control": "v1_sjepa_l0_shuf",
    "n": 10,
    "d_r_median": 0.0298,
    "d_r_n_pos": 10,
    "d_r_p": 0.0009765625,
    "d_acc_median": 0.0005500000000000001,
    "d_acc_n_pos": 6,
    "d_acc_p": 0.376953125,
    "d_r_vs_host_median": 0.017799999999999996,
    "d_r_vs_host_n_pos": 10,
    "d_r_vs_host_p": 0.0009765625,
    "d_acc_vs_host_median": 0.00014999999999999996,
    "d_acc_vs_host_n_pos": 5,
    "d_acc_vs_host_p": 0.623046875
  },
  {
    "variant": "v4_s1a_recon",
    "control": "v4_s1a_recon_shuf",
    "n": 10,
    "d_r_median": 0.0285,
    "d_r_n_pos": 10,
    "d_r_p": 0.0009765625,
    "d_acc_median": 0.0015500000000000006,
    "d_acc_n_pos": 9,
    "d_acc_p": 0.0107421875,
    "d_r_vs_host_median": 0.0177,
    "d_r_vs_host_n_pos": 10,
    "d_r_vs_host_p": 0.0009765625,
    "d_acc_vs_host_median": 0.001,
    "d_acc_vs_host_n_pos": 10,
    "d_acc_vs_host_p": 0.0009765625
  }
]
