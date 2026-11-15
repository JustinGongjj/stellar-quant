import sys; __import__('sys').path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'src'))
import numpy as np
R=str(__import__('pathlib').Path.home() / '.cache/stellarq')+'/wide2/'
for n in ['open','close','volume','amount','turnover','float_mktcap','low','valid','dates','syms']:
    try:
        a=np.load(R+f'wide_{n}.npy', mmap_mode='r'); print(n, a.shape, a.dtype)
    except Exception as e: print(n,'ERR',e)
