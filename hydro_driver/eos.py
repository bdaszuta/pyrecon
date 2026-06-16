"""
 ,-*
(_)

@author: Boris Daszuta
@SPDX-License-Identifier: BSD-3-Clause
@function: Gamma-law ideal gas EOS for 1+1D SR hydro

Variable indices (1+1D):
  Conserved: IDN=0 (D), ISX=1 (S_x), ITAU=2 (tau), NCONS=3
  Primitive: IRHO=0 (rho), IUX=1 (u_x = W*v_x), IP=2 (P), NPRIM=3
"""

import numpy as np

# Conserved indices (1+1D)
IDN  = 0
ISX  = 1
ITAU = 2
NCONS = 3

# Primitive indices (1+1D)
IRHO = 0
IUX  = 1
IP   = 2
NPRIM = 3

# Atmosphere floor values (code units, mb=1)
RHO_ATM = 1.0e-10
P_ATM   = 1.0e-15


class IdealGasEOS:
    """Gamma-law ideal gas equation of state.

    Formulas (rest-mass density units, mb=1):
      h   = 1 + Gamma/(Gamma-1) * P/rho
      cs2 = Gamma * P / (rho * h)
    """

    def __init__(self, gamma=2.0):
        self.gamma = gamma
        self.gm1 = gamma - 1.0

    def enthalpy(self, rho, P):
        """Specific enthalpy h = 1 + Gamma/(Gamma-1) * P/rho."""
        return 1.0 + self.gamma / self.gm1 * P / rho

    def sound_speed_sq(self, rho, P):
        """Sound speed squared: cs^2 = Gamma*P/(rho*h).

        For Gamma=2: cs^2 = 2*P / (rho + 2*P).
        """
        h = self.enthalpy(rho, P)
        return self.gamma * P / (rho * h)


def prim_to_cons(prim, eos):
    """Convert primitive to undensitized conserved variables (1+1D flat bg).

    Parameters
    ----------
    prim : ndarray, shape (3,)
        prim[IRHO]=rho, prim[IUX]=u_x, prim[IP]=P
    eos : IdealGasEOS

    Returns
    -------
    cons : ndarray, shape (3,)
        D, S_x, tau

    Undensitized Valencia variables on flat Minkowski background.
    """
    rho = prim[IRHO]
    P   = prim[IP]
    u_x = prim[IUX]

    W = np.sqrt(1.0 + u_x * u_x)       # flat bg, 1+1D
    h = eos.enthalpy(rho, P)

    D   = rho * W
    S_x = rho * h * W * u_x            # = rho*h*W^2*v_x
    tau = rho * h * W * W - P - D

    return np.array([D, S_x, tau])


def cons_to_prim(cons, eos, P_guess=None,
                 max_iter=50, tol=1.0e-12):
    """Recover primitives from undensitized conserved (1+1D flat bg).

    Parameters
    ----------
    cons : ndarray, shape (3,)
        D, S_x, tau
    eos : IdealGasEOS
    P_guess : float or None
    max_iter : int
    tol : float

    Returns
    -------
    prim : ndarray, shape (3,)
        rho, u_x, P

    Raises
    ------
    ValueError on non-convergence (caller applies atmosphere).
    """
    D   = max(cons[IDN], RHO_ATM)
    S_x = cons[ISX]
    tau = max(cons[ITAU], 0.0)

    S2 = S_x * S_x
    gm1 = eos.gm1
    gfac = eos.gamma / gm1

    if S2 < 1.0e-30:
        # Static: v=0, W=1, algebraic solution
        P = tau * gm1
        if P < P_ATM:
            raise ValueError("C2P: unphysical state")
        return np.array([D, 0.0, P])

    def f_and_df(P):
        tauDP = tau + D + P
        if tauDP <= 0.0:
            return None, None
        v2 = S2 / (tauDP * tauDP)
        if v2 >= 1.0:
            return None, None
        W = 1.0 / np.sqrt(1.0 - v2)
        W2 = W * W
        f = tauDP - D * W - gfac * P * W2
        dW_dP = -W * W2 * v2 / tauDP
        dW2_dP = 2.0 * W * dW_dP
        df = 1.0 - D * dW_dP - gfac * (W2 + P * dW2_dP)
        return f, df

    if P_guess is None:
        P = max(tau * 0.1, P_ATM)
    else:
        P = max(P_guess, P_ATM)

    P_lo = P_ATM
    P_hi = 1.0e15

    for _ in range(max_iter):
        f_val, df_val = f_and_df(P)
        if f_val is None:
            P = 0.5 * (P_lo + P_hi)
            continue
        if abs(f_val) < tol:
            tauDP = tau + D + P
            v2 = S2 / (tauDP * tauDP)
            W = 1.0 / np.sqrt(1.0 - v2) if v2 < 1.0 else 1.0
            rho = D / W
            u_x = S_x * W / tauDP         # u_x = W * v_x = W * S_x/(tau+D+P)
            return np.array([rho, u_x, P])
        if df_val is None or abs(df_val) < 1.0e-30:
            if f_val > 0.0:
                P_lo = P
            else:
                P_hi = P
            P = 0.5 * (P_lo + P_hi)
            continue
        dP = -f_val / df_val
        # Check step size convergence (handles ultra-relativistic cases)
        if abs(dP) < tol * P:
            tauDP = tau + D + P
            v2 = S2 / (tauDP * tauDP)
            W = 1.0 / np.sqrt(1.0 - v2) if v2 < 1.0 else 1.0
            rho = D / W
            u_x = S_x * W / tauDP
            return np.array([rho, u_x, P])
        if dP > 0.0:
            P_new = min(P + dP, P_hi)
        else:
            P_new = max(P + dP, P_lo)
        if P_new <= P_lo or P_new >= P_hi:
            if f_val > 0.0:
                P_lo = P
            else:
                P_hi = P
            P = 0.5 * (P_lo + P_hi)
        else:
            P = P_new

    raise ValueError(f"C2P failed after {max_iter} iterations")
#
# :D
#
