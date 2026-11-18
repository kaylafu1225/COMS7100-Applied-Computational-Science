"""Hydrogen 1s Gaussian variational solver (atomic units).

The first number in each Gaussian basis data row is an exponent.
Supplied contraction coefficients are deliberately ignored, because
we optimize the expansion coefficients with the variational method.
"""
from pathlib import Path
import math
import numpy as np
import mpmath as mp

BASIS_NUMBERS = [1, 2, 3, 4, 5, 6, 12, 20, 23]
HERE = Path(__file__).resolve().parent

def read_exponents(path):
    """Read the S shell from Gaussian94-format hydrogen basis text."""
    lines = Path(path).read_text().splitlines()
    for i, line in enumerate(lines):
        fields = line.strip().split()
        if len(fields) >= 2 and fields[0].upper() == "S":
            n = int(fields[1])
            data = lines[i+1:i+1+n]
            if len(data) != n:
                raise ValueError("Incomplete S shell")
            result = [mp.mpf(row.split()[0].replace("D","E").replace("d","E"))
                      for row in data]
            if any(a <= 0 for a in result):
                raise ValueError("Exponents must be positive")
            return result
    raise ValueError("No S shell found")

def available_basis():
    available = {"STO-1G": [mp.mpf(8)/(9*mp.pi)]}
    # Reference exponents for STO-2G (can be overridden by a user-supplied file).
    available["STO-2G"] = [mp.mpf("1.309756377"), mp.mpf("0.233135974")]
    for n in BASIS_NUMBERS[1:]:
        path = HERE / f"H_STO-{n}G.txt"
        if path.exists():
            available[f"STO-{n}G"] = read_exponents(path)
    return available

def matrices(alphas):
    n = len(alphas)
    s, t, v = mp.matrix(n), mp.matrix(n), mp.matrix(n)
    for i in range(n):
        for j in range(n):
            a, b = alphas[i], alphas[j]
            p = a + b
            sij = (2*mp.sqrt(a*b)/p)**mp.mpf("1.5")
            s[i,j] = sij
            t[i,j] = (3*a*b/p)*sij
            v[i,j] = -2*mp.sqrt(p/mp.pi)*sij
    return s,t,v

def solve(alphas):
    s,t,v = matrices(alphas)
    h=t+v
    eigenvalues, u = mp.eigsy(s)
    if min(eigenvalues) <= 0:
        raise ArithmeticError("Overlap matrix not positive definite")
    x=u*mp.diag([1/mp.sqrt(z) for z in eigenvalues])*u.T
    energies, vectors=mp.eigsy(x*h*x)
    c=x*vectors[:,0]
    c=c/mp.sqrt((c.T*s*c)[0])
    if sum(c[i]*(2*alphas[i]/mp.pi)**mp.mpf(".75") for i in range(len(alphas))) < 0:
        c=-c
    T=(c.T*t*c)[0]
    V=(c.T*v*c)[0]
    return {"alpha":alphas,"coeff":[c[i] for i in range(len(alphas))],
            "T":float(T),"V":float(V),"E":float(T+V),
            "Virial":float(-V/T),
            "min_overlap":float(min(eigenvalues)),
            "overlap":np.array([[float(s[i,j]) for j in range(len(alphas))]
                                for i in range(len(alphas))])}

def compute_all(digits=80):
    mp.mp.dps=digits
    basis=available_basis()
    results={}
    errors={}
    for name, alphas in basis.items():
        try:
            results[name]=solve(alphas)
        except Exception as e:
            errors[name]=str(e)
    return results, errors

def psi(name, results, r):
    r=np.asarray(r,dtype=float)
    if name=="Exact":
        return np.exp(-r)/np.sqrt(np.pi)
    result=results[name]
    out=np.zeros_like(r)
    for a,c in zip(result["alpha"],result["coeff"]):
        aa=float(a)
        out += float(c)*(2*aa/np.pi)**.75*np.exp(-aa*r*r)
    return out

def radial(name, results, r):
    return 4*np.pi*np.asarray(r)**2*psi(name,results,r)**2
