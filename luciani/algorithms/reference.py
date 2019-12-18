import numpy as np
import scipy as sc
import pandas as pd


"""
randmio_und
"""

def randmio_und(R, itr):
    
    R = R.copy()
    n = len(R)
    i, j = np.where(np.tril(R))
    k = len(i)
    itr *= k

    # maximum number of rewiring attempts per iteration
    max_attempts = np.round(n * k / (n * (n - 1)))
    # actual number of successful rewirings
    eff = 0

    for it in range(itr):
        att = 0
        while att <= max_attempts:  # while not rewired
            while True:
                e1, e2 = np.random.randint(k, size=(2,))
                while e1 == e2:
                    e2 = np.random.randint(k)
                a = i[e1]
                b = j[e1]
                c = i[e2]
                d = j[e2]

                if a != c and a != d and b != c and b != d:
                    break  # all 4 vertices must be different

            if np.random.random() > .5:
                i.setflags(write=True)
                j.setflags(write=True)
                i[e2] = d
                j[e2] = c  # flip edge c-d with 50% probability
                c = i[e2]
                d = j[e2]  # to explore all potential rewirings

            # rewiring condition
            if not (R[a, d] or R[c, b]):
                R[a, d] = R[a, b]
                R[a, b] = 0
                R[d, a] = R[b, a]
                R[b, a] = 0
                R[c, b] = R[c, d]
                R[c, d] = 0
                R[b, c] = R[d, c]
                R[d, c] = 0

                j.setflags(write=True)
                j[e1] = d
                j[e2] = b  # reassign edge indices
                eff += 1
                break
            att += 1

    return R, eff


"""
latmio_und_connected(w,1)
"""
def latmio_und_connected(R, itr, D=None):
    n = len(R)

    ind_rp = np.random.permutation(n)  # randomly reorder matrix
    R = R.copy()
    R = R[np.ix_(ind_rp, ind_rp)]

    if D is None:
        D = np.zeros((n, n))
        un = np.mod(range(1, n), n)
        um = np.mod(range(n - 1, 0, -1), n)
        u = np.append((0,), np.where(un < um, un, um))

        for v in range(int(np.ceil(n / 2))):
            D[n - v - 1, :] = np.append(u[v + 1:], u[:v + 1])
            D[v, :] = D[n - v - 1, :][::-1]

    i, j = np.where(np.tril(R))
    k = len(i)
    itr *= k

    # maximal number of rewiring attempts per iteration
    max_attempts = np.round(n * k / (n * (n - 1) / 2))

    # actual number of successful rewirings
    eff = 0

    for it in range(itr):
        att = 0
        while att <= max_attempts:
            rewire = True
            while True:
                e1 = np.random.randint(k)
                e2 = np.random.randint(k)
                while e1 == e2:
                    e2 = np.random.randint(k)
                a = i[e1]
                b = j[e1]
                c = i[e2]
                d = j[e2]

                if a != c and a != d and b != c and b != d:
                    break

            if np.random.random() > .5:
                i.setflags(write=True)
                j.setflags(write=True)
                i[e2] = d
                j[e2] = c  # flip edge c-d with 50% probability
                c = i[e2]
                d = j[e2]  # to explore all potential rewirings

            # rewiring condition
            if not (R[a, d] or R[c, b]):
                # lattice condition
                if (D[a, b] * R[a, b] + D[c, d] * R[c, d] >= D[a, d] * R[a, b] + D[c, b] * R[c, d]):
                    # connectedness condition
                    if not (R[a, c] or R[b, d]):
                        P = R[(a, d), :].copy()
                        P[0, b] = 0
                        P[1, c] = 0
                        PN = P.copy()
                        PN[:, d] = 1
                        PN[:, a] = 1
                        while True:
                            P[0, :] = np.any(R[P[0, :] != 0, :], axis=0)
                            P[1, :] = np.any(R[P[1, :] != 0, :], axis=0)
                            P *= np.logical_not(PN)
                            if not np.all(np.any(P, axis=1)):
                                rewire = False
                                break
                            elif np.any(P[:, (b, c)]):
                                break
                            PN += P
                    # end connectedness testing

                    if rewire:  # reassign edges
                        R[a, d] = R[a, b]
                        R[a, b] = 0
                        R[d, a] = R[b, a]
                        R[b, a] = 0
                        R[c, b] = R[c, d]
                        R[c, d] = 0
                        R[b, c] = R[d, c]
                        R[d, c] = 0

                        j.setflags(write=True)
                        j[e1] = d
                        j[e2] = b
                        eff += 1
                        break
            att += 1

    Rlatt = R[np.ix_(ind_rp[::-1], ind_rp[::-1])]
    return Rlatt, R, ind_rp, eff

  