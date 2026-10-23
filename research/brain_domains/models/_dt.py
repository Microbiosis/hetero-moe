import sys
sys.path.insert(0, ".")
import numpy as np
import base_specs as BS
from domain_adapters import DomainSpec, adapt, idw_interpolate

have = BS.THINGS_EEG_CHANS
eeg = np.random.randn(2, 3, 63, 250).astype(np.float32)
print("输入 dtype:", eeg.dtype)

out, names, how = idw_interpolate(eeg, have, BS.EEGDINO_CHANS)
print("idw_interpolate 输出 dtype:", out.dtype, how)

sp = DomainSpec(model="eegdino-small", chans=BS.EEGDINO_CHANS, sfreq=200.0,
                 n_times=1000, interpolate_missing=True)
a, info = adapt(eeg, sp, have)
print("adapt 输出 dtype:", a.dtype, "shape", a.shape)

sp2 = DomainSpec(model="cbramod", chans=None, sfreq=200.0, n_times=200)
a2, _ = adapt(eeg, sp2, have)
print("cbramod adapt dtype:", a2.dtype, "shape", a2.shape)
