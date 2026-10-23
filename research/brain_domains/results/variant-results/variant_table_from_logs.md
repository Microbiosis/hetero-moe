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
    "r": 0.013049999999999999,
    "p": 0.0075,
    "n_sig": 7,
    "acc": 0.00525,
    "null": 0.0048,
    "ratio": 1.0937500000000002
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
    "r": 0.008150000000000001,
    "p": 0.065,
    "n_sig": 5,
    "acc": 0.00545,
    "null": 0.005050000000000001,
    "ratio": 1.079207920792079
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
    "d_r_median": 0.00445,
    "d_r_n_pos": 8,
    "d_r_p": 0.0546875,
    "d_acc_median": -0.0002499999999999998,
    "d_acc_n_pos": 3,
    "d_acc_p": 0.9453125,
    "d_r_vs_host_median": 0.0015499999999999997,
    "d_r_vs_host_n_pos": 7,
    "d_r_vs_host_p": 0.171875,
    "d_acc_vs_host_median": -0.0002999999999999995,
    "d_acc_vs_host_n_pos": 3,
    "d_acc_vs_host_p": 0.9453125
  }
]
