from src.tl_core import LineInputs, basic_params
from src.tl_matching import single_stub_shunt, quarter_wave_transform

inp = LineInputs(0.05, 3e-7, 1e-8, 8e-11, 1e9, 0.25, 25+0j)

# VSWR before matching
print("VSWR before:", basic_params(inp).VSWR)

# VSWR after matching using single stub shunt
res = single_stub_shunt(inp.R, inp.L, inp.G, inp.C, inp.f, inp.l, inp.ZL, prefer='short')
print("VSWR after (stub):", res.VSWR_src, "d_opt:", res.d_opt, "l_stub:", res.l_stub)

# VSWR after matching using quarter-wave transformer
q = quarter_wave_transform(inp.R, inp.L, inp.G, inp.C, inp.f, inp.ZL)
print("VSWR after (λ/4):", q.VSWR_src)
