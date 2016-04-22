#! /usr/bin/python
import numpy as np
"""
    n, p might represent left and right
    1, 2 might represent false and true
"""

def information_gain(n1, n2, p1, p2):
    n1p1 = float(n1) + p1
    n2p2 = float(n2) + p2
    n = float(n1) + n2
    p = float(p1) + p2
    pn = float(n) + p

    q1 = n/pn
    q2 = p/pn
    q3 = n1/n1p1
    q4 = p1/n1p1
    q5 = n2/n2p2
    q6 = p2/n2p2

    current = 0.
    rem1 = 0.
    rem2 = 0.

    eps = 0.

    if q1 > eps:
        current += q1*np.log2(q1)
    if q2 > eps:
        current += q2*np.log2(q2)
    current *= -1.

    if n1p1 > eps:
        if q3 > eps:
            rem1 += q3*np.log2(q3)
        if q4 > eps:
            rem1 += q4*np.log2(q4)
        rem1 *= -n1p1/pn

    if n2p2 > eps:
        if q5 > eps:
            rem2 += q5*np.log2(q5)
        if q6 > eps:
            rem2 += q6*np.log2(q6)
        rem2 *= -n2p2/pn

    remainder = rem1 + rem2

    return current - remainder
