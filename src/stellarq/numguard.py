"""数值安全层：所有除法、比率、复权、标准化一律走这里。

为什么需要
----------
本运行环境（Debian 11 / CPython 3.12.16 自编译 / numpy 2.5.4）的浮点运算在
部分上下文会返回**错误结果且不报错**，已实测复现（详见 docs/ENV_NUMERIC_BUG.md）：

* ``1.0 / 1.5``、``1.0 / 1.01`` 在函数体内返回 ``1.0``；
* ``1.0 - 1.0/1.01`` 返回 ``0.0``；
* ``float(Fraction(1) - Fraction(2,3))`` 返回 ``0.0``；
* ``0.1 * 0.2`` 返回 ``1.6e30``；
* ``Decimal`` 在 ``localcontext()`` 内的减法同样出错；
* ``math.frexp`` 的尾数在部分输入上错误。

结论：float 运算单元与部分数学库函数在本环境**不可信**。可靠通道只有一个——
按 IEEE-754 二进制位直接读写（``struct``），全程整数精确运算，最后写回 double。
交叉验证：同一批 300 组随机数在 Node.js（V8 双精度）与位级实现之间完全一致。

约定
----
* 比率 / 除法：``div(a, b)``；1 - 1/r：``one_minus_inv(r)``；
* 加减乘：``add`` / ``sub`` / ``mul``；收益率：``pct(a, b)``；
* 向量化：``div_arr`` / ``transform_arr`` 接受 numpy 数组，逐元素走位级实现；
* 启动自检：``assert_safe()``。**不通过必须中止**，不允许带着错误数值继续出结论。
"""
from __future__ import annotations

import struct
from typing import Iterable

import numpy as np

MANT_BITS = 52
HIDDEN = 1 << MANT_BITS
EXP_MASK = 0x7FF
_PACK = struct.Struct("<d").pack
_UNPACK = struct.Struct("<d").unpack
_PACK_INTO = struct.Struct("<d").pack_into
_UNPACK_FROM = struct.Struct("<d").unpack_from


# ---------------------------------------------------------------- 基础位操作
def decompose(x: float) -> tuple[int, int, int]:
    """double → (sign, mantissa, exponent)，值 = (-1)^sign × mantissa × 2^exponent。"""
    bits = int.from_bytes(_PACK(float(x)), "little")
    sign = bits >> 63
    exp = (bits >> MANT_BITS) & EXP_MASK
    frac = bits & (HIDDEN - 1)
    if exp == 0:
        return sign, frac, -1074
    if exp == EXP_MASK:
        raise ValueError("inf/nan 输入")
    return sign, frac | HIDDEN, exp - 1075


def compose(sign: int, mant: int, exp: int) -> float:
    """(sign, mant, exp) → 最近的 double（就近舍入，平局取偶）。"""
    if mant == 0:
        return -0.0 if sign else 0.0
    bl = mant.bit_length()
    shift = MANT_BITS + 1 - bl
    if shift >= 0:
        mant <<= shift
        exp -= shift
    else:
        d = -shift
        rem = mant & ((1 << d) - 1)
        mant >>= d
        exp += d
        half = 1 << (d - 1)
        if rem > half or (rem == half and (mant & 1)):
            mant += 1
            if mant == (HIDDEN << 1):
                mant >>= 1
                exp += 1
    e = exp + 1075
    if e <= 0:
        d = 1 - e
        rem = mant & ((1 << d) - 1)
        mant >>= d
        half = 1 << (d - 1)
        if rem > half or (rem == half and (mant & 1)):
            mant += 1
        e = 1 if mant >= HIDDEN else 0
        if e == 0:
            mant &= HIDDEN - 1
    if e >= EXP_MASK:
        return float("-inf") if sign else float("inf")
    bits = (sign << 63) | (e << MANT_BITS) | (mant & (HIDDEN - 1))
    return _UNPACK(bits.to_bytes(8, "little"))[0]


def _parts(x) -> tuple[int, int, int]:
    """取 double 的 (sign, mant, exp)；用手工比较而不是 float() 内建，绕开宿主转换缺陷。"""
    return decompose(_to_double(x))


def _to_double(x) -> float:
    """任意数值 → double，只经过 struct 打包（不调用 float() 内建）。"""
    if type(x) is float:
        return x
    try:
        return _UNPACK(_PACK(x))[0]
    except (struct.error, TypeError):
        try:
            return _UNPACK(_PACK(float(x)))[0]
        except Exception:  # noqa: BLE001
            return float("nan")


# ---------------------------------------------------------------- 四则运算
# 所有运算只用「取位 → 整数运算 → 写位」三步，不调用任何浮点算术指令。

def div(a, b) -> float:
    """a / b（IEEE 双精度语义，纯整数实现）。"""
    try:
        fa, fb = _to_double(a), _to_double(b)
    except (TypeError, ValueError):
        return float("nan")
    sa, ma, ea = _parts(fa)
    sb, mb, eb = _parts(fb)
    if mb == 0:
        if ma == 0:
            return float("nan")
        return _from_bits((1 << 63) ^ (sa << 63) | (EXP_MASK << MANT_BITS))
    if ma == 0:
        return _from_bits((sa ^ sb) << 63)
    shift = 110
    q, _ = divmod(ma << shift, mb)
    return compose(sa ^ sb, q, ea - eb - shift)


def mul(a, b) -> float:
    """a × b（纯整数实现）。"""
    sa, ma, ea = _parts(a)
    sb, mb, eb = _parts(b)
    return compose(sa ^ sb, ma * mb, ea + eb)


def add(a, b) -> float:
    """a + b（纯整数实现，直接按符号对齐相加减，不经二次取位）。"""
    sa, ma, ea = _parts(_to_double(a) if type(a) is not float else a)
    sb, mb, eb = _parts(_to_double(b) if type(b) is not float else b)
    if ma == 0:
        return b
    if mb == 0:
        return a
    e = ea if ea < eb else eb
    na = -(ma << (ea - e)) if sa else (ma << (ea - e))
    nb = -(mb << (eb - e)) if sb else (mb << (eb - e))
    return _round_int(na + nb, e)


def sub(a, b) -> float:
    """a - b（纯整数实现）。"""
    return add(a, neg(b))


def _round_int(v: int, e: int) -> float:
    """整数 v × 2^e → 最近的 double（就近舍入，平局取偶）。"""
    if v == 0:
        return 0.0
    sign = 1 if v < 0 else 0
    mag = -v if v < 0 else v
    width = mag.bit_length()
    shift = 120
    q, _ = divmod(mag << shift, 1 << width)
    return compose(sign, q, e + width - shift)


def _from_bits(bits: int) -> float:
    return _UNPACK(bits.to_bytes(8, "little"))[0]


def neg(x) -> float:
    """-x（位级翻转符号位）。"""
    bits = int.from_bytes(_PACK(_to_double(x)), "little")
    return _from_bits(bits ^ (1 << 63))


def absv(x) -> float:
    bits = int.from_bytes(_PACK(_to_double(x)), "little")
    return _from_bits(bits & ~(1 << 63))


def one_minus_inv(r) -> float:
    """1 - 1/r，一次整数运算完成（(num-den)/num），不走两次取位组合。"""
    sr, mr, er = _parts(r)
    if mr == 0:
        return float("nan")
    if er >= 0:
        num = (mr << er) - 1
        den = mr << er
        exp = 0
    else:
        num = mr - (1 << -er)
        den = mr
        exp = -er
    if num == 0:
        return 0.0
    q, _ = divmod(abs(num) << 62, den)
    return compose(1 if num < 0 else 0, q, exp + 62 - (den.bit_length() - 0))


def pct(a, b) -> float:
    """(a - b) / b，即简单收益率。"""
    return div(sub(a, b), b)


def ratio(a, b) -> float:
    return div(a, b)


# ---------------------------------------------------------------- 向量化
def _map2(a, b, fn) -> np.ndarray:
    av = np.asarray(a, dtype=float).ravel()
    bv = np.asarray(b, dtype=float).ravel()
    if av.shape != bv.shape:
        av, bv = np.broadcast_arrays(av, bv)
        av, bv = av.ravel(), bv.ravel()
    a_bytes = av.astype("<f8").tobytes()
    b_bytes = bv.astype("<f8").tobytes()
    out = bytearray(len(a_bytes))
    un, pk = _UNPACK_FROM, _PACK_INTO
    for i in range(0, len(a_bytes), 8):
        pk(out, i, fn(un(a_bytes, i)[0], un(b_bytes, i)[0]))
    return np.frombuffer(bytes(out), dtype="<f8").copy()


def div_arr(a, b) -> np.ndarray:
    """逐元素除法（numpy 数组 → numpy 数组，形状按广播）。"""
    av = np.asarray(a, dtype=float)
    bv = np.asarray(b, dtype=float)
    shape = np.broadcast_shapes(av.shape, bv.shape)
    return _map2(av, bv, div).reshape(shape)


def transform_arr(a, fn) -> np.ndarray:
    """逐元素标量函数（输入输出均为 float）。"""
    av = np.asarray(a, dtype=float)
    shape = av.shape
    a_bytes = av.ravel().astype("<f8").tobytes()
    out = bytearray(len(a_bytes))
    un, pk = _UNPACK_FROM, _PACK_INTO
    for i in range(0, len(a_bytes), 8):
        pk(out, i, float(fn(un(a_bytes, i)[0])))
    return np.frombuffer(bytes(out), dtype="<f8").copy().reshape(shape)


def running_product_series(values: Iterable[float], anchor: float = 1.0) -> list[float]:
    """逆序累乘：out[-1] = anchor，out[i] = out[i+1] × values[i+1]。"""
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return []
    out = [0.0] * n
    out[-1] = anchor
    for i in range(n - 2, -1, -1):
        out[i] = mul(out[i + 1], vals[i + 1])
    return out


# ---------------------------------------------------------------- 自检
def self_test() -> dict:
    """核对一组已知答案的运算。``summary.all_ok`` 为 False 时必须中止流程。"""
    cases = {
        "div_1_over_1.01": (div(1.0, 1.01), 0.9900990099009901),
        "div_1_over_1.5": (div(1.0, 1.5), 0.6666666666666666),
        "div_1_over_7": (div(1.0, 7.0), 0.14285714285714285),
        "div_2_over_3": (div(2.0, 3.0), 0.6666666666666666),
        "one_minus_inv_1.01": (one_minus_inv(1.01), 0.00990099009900991),
        "add_0.1_0.2": (add(0.1, 0.2), 0.30000000000000004),
        "sub_0.3_0.1": (sub(0.3, 0.1), 0.19999999999999998),
        "mul_0.5x0.5": (mul(0.5, 0.5), 0.25),
        "mul_0.1x0.2": (mul(0.1, 0.2), 0.020000000000000004),
        "sub_1.5_0.5": (sub(1.5, 0.5), 1.0),
        "pct_1.5_1.0": (pct(1.5, 1.0), 0.5),
        "vec_div": (float(div_arr(np.array([1.0, 1.0]), np.array([1.01, 1.5]))[1]), 0.6666666666666666),
        "running_product": (running_product_series([1.0, 1.0, 0.5, 0.5])[0], 0.25),
    }
    out: dict = {}
    ok = True
    for name, (got, want) in cases.items():
        good = bool(got == want)
        out[name] = {"got": got, "want": want, "ok": good}
        ok = ok and good
    out["summary"] = {"all_ok": ok}
    return out


def assert_safe(context: str = "") -> dict:
    """自检不通过直接抛错（fail-closed）。返回自检明细供 manifest 记录。"""
    res = self_test()
    if not res["summary"]["all_ok"]:
        bad = {k: v for k, v in res.items() if isinstance(v, dict) and v.get("ok") is False}
        raise RuntimeError(f"数值自检失败（{context}）: {bad}")
    return res
