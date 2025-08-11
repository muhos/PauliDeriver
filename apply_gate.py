import numpy as np
from itertools import product

labels = {(0, 0): 'I', (1, 0): 'X', (0, 1): 'Z', (1, 1): 'Y'}

I = np.eye(2, dtype=complex)
X = np.array([[0, 1],[1, 0]], dtype=complex)
Z = np.array([[1, 0],[0,-1]], dtype=complex)
Y = 1j * X @ Z

def P(x, z):
    return {(0, 0): I, (1, 0): X, (0,1): Z, (1, 1): Y}[(x, z)]

def pauli_tensor(xc, zc, xt, zt):
    return np.kron(P(xc, zc), P(xt, zt))

def coeffstring(coeff, tol):
    sign = ""; coeff_str = ""
    is_real = abs(coeff.imag) < tol
    c = coeff.real if is_real else coeff
    if not is_real:
        coeff_str, sign = ("%.3g" % c), ''
    else:
        magnitude, sign = abs(c), ('-' if c < 0 else '')
        if np.isclose(magnitude, 1 / np.sqrt(2), atol=tol): coeff_str = "1/√2"
        elif np.isclose(magnitude, 1.0, atol=tol):          coeff_str = "1"
        elif np.isclose(magnitude, 0.5, atol=tol):          coeff_str = "1/2"
        else: coeff_str = ("%.3f" % magnitude).rstrip('0').rstrip('.')
    return sign, coeff_str

def derive_single_qubit_pauli_constraints(U, tol, print_pauli, print_table, try_simplify):
    name, U = U

    info = []
    neg_input_minterms = []      # inputs (x, z) where negative signs appear in Pauli strings
    neg_branch_minterms = []     # only branches (x, z, x', z') with negative coefficients

    print("--------[ Gate %s (single qubit) ]--------------------" % name)

    if print_pauli:
        print(" Pauli strings:")

    for x, z in product([0,1], repeat=2):
        Pin  = P(x,z)
        Pout = U @ Pin @ U.conj().T

        terms = []               # (coeff, (x',z'))
        out_xs, out_zs = set(), set()
        has_negative_input = False

        for xp, zp in product([0,1], repeat=2):
            Q = P(xp,zp)
            coeff = np.trace(Q.conj().T @ Pout) / 2  # d=2
            if abs(coeff) > tol:
                terms.append((coeff, (xp,zp)))
                out_xs.add(xp)
                out_zs.add(zp)
                if coeff.real < -tol and abs(coeff.imag) < tol:
                    has_negative_input = True
                    neg_branch_minterms.append((x, z, xp, zp))

        if has_negative_input:
            neg_input_minterms.append((x,z))

        info.append({
            "in_x": x, "in_z": z,
            "out_xs": out_xs, "out_zs": out_zs,
        })

        if print_pauli:
            input = "  %s (%s) %s† = " % (name, labels[(x,z)], name)
            parts = []
            for coeff, (xp,zp) in terms:
                sign, coeff_str = coeffstring(coeff, tol)
                parts.append("%s%s*%s" % (sign, coeff_str, labels[(xp,zp)]))
            pauli_string = " + ".join(parts).replace("+ -", "- ")
            print(" " + input + pauli_string)

    x_branches = any(len(e["out_xs"]) > 1 for e in info)
    z_branches = any(len(e["out_zs"]) > 1 for e in info)

    if print_table:
        print("")
        print("Pauli bits:")
        print("  %s  %s    %-6s  %-6s    %-10s   %-s" %('x', 'z', "x'", "z'", 'Negative', "Negative branch"))
        print(" " + "-"*70)
        for e in info:
            x, z = e["in_x"], e["in_z"]
            ox, oz = e["out_xs"], e["out_zs"]
            neg = 'yes' if (x, z) in neg_input_minterms else 'no'
            nb = 'none'
            if x_branches or z_branches:
                pairs = [(xp,zp) for (a,b,xp,zp) in neg_branch_minterms if a == x and b == z]
                if pairs:
                    if x_branches and z_branches:
                        nb = ' | '.join("x'=%d,z'=%d" % (xp, zp) for (xp, zp) in pairs)
                    elif x_branches:
                        xs = sorted(set(xp for (xp, _) in pairs))
                        nb = ' | '.join("x'=%d" % v for v in xs)
                    elif z_branches:
                        zs = sorted(set(zp for (_, zp) in pairs))
                        nb = ' | '.join("z'=%d" % v for v in zs)
            print("  %d  %d    %-6s  %-6s    %-10s   %-s" % (x, z, str(ox), str(oz), neg, nb))

    if try_simplify:
        try:
            import sympy as sp
            var_in = sp.symbols("x z", boolean=True)

            # x'
            if x_branches:
                mins_x1 = [(e["in_x"], e["in_z"]) for e in info if len(e["out_xs"]) > 1]
                expr_xp = sp.simplify_logic(sp.SOPform(list(var_in), mins_x1), form='dnf')
                print("\n x' := (%s) | x' <-> x" % expr_xp)
            else:
                mins_x1 = [(e["in_x"], e["in_z"]) for e in info if (len(e["out_xs"])==1 and 1 in e["out_xs"])]
                expr_xp = sp.simplify_logic(sp.SOPform(list(var_in), mins_x1), form='dnf')
                print("\n x' := %s" % expr_xp)

            # z'
            if z_branches:
                mins_z1 = [(e["in_x"], e["in_z"]) for e in info if len(e["out_zs"]) > 1]
                expr_zp = sp.simplify_logic(sp.SOPform(list(var_in), mins_z1), form='dnf')
                print(" z' := (%s) | z' <-> z" % expr_zp)
            else:
                mins_z1 = [(e["in_x"], e["in_z"]) for e in info if (len(e["out_zs"])==1 and 1 in e["out_zs"])]
                expr_zp = sp.simplify_logic(sp.SOPform(list(var_in), mins_z1), form='dnf')
                print(" z' := %s" % expr_zp)

            # Sign update
            var_names = ["x","z"]
            if x_branches: var_names.append("x'")
            if z_branches: var_names.append("z'")
            reduced = []
            for (x,z,xp,zp) in neg_branch_minterms:
                row = [x, z]
                if x_branches: row.append(xp)
                if z_branches: row.append(zp)
                reduced.append(row)
            if reduced:
                vars_order = sp.symbols(" ".join(var_names), boolean=True)
                expr = sp.SOPform(list(vars_order), reduced)
                expr_simplified = sp.simplify_logic(expr, form='dnf')
                print("\n s := %s\n" % (expr_simplified))
            else:
                print("\n s :=  r' = r\n")
        except Exception as e:
            print(" (sympy simplification skipped:", e, ")")

def derive_dual_qubits_pauli_constraints(U, tol, print_pauli, print_table, try_simplify):
    gate, U = U

    print("--------[ Gate %s ]--------------------" % gate)

    info = []
    neg_input_minterms = []
    neg_branch_minterms = []

    if print_pauli:
        print(" Pauli strings:")

    for xc, zc, xt, zt in product([0,1], repeat=4):
        P_in = pauli_tensor(xc, zc, xt, zt)
        P_out = U @ P_in @ U.conj().T

        out_xc, out_zc, out_xt, out_zt = set(), set(), set(), set()
        pauli_terms = []
        has_negative_input = False

        for xcp, zcp, xtp, ztp in product([0,1], repeat=4):
            Q = pauli_tensor(xcp, zcp, xtp, ztp)
            coeff = np.trace(Q.conj().T @ P_out) / 4
            if abs(coeff) > tol:
                out_xc.add(xcp); out_zc.add(zcp); out_xt.add(xtp); out_zt.add(ztp)
                if print_pauli:
                    pauli_terms.append((coeff, (xcp, zcp, xtp, ztp)))
                if coeff.real < -tol and abs(coeff.imag) < tol:
                    has_negative_input = True
                    neg_branch_minterms.append((xc, zc, xt, zt, xcp, zcp, xtp, ztp))

        info.append({
            'in_xc': xc, 'in_zc': zc, 'in_xt': xt, 'in_zt': zt,
            'out_xc': out_xc, 'out_zc': out_zc, 'out_xt': out_xt, 'out_zt': out_zt,
        })

        if has_negative_input:
            neg_input_minterms.append((xc, zc, xt, zt))

        if print_pauli:
            inp_label = "%sc ⊗ %st" % (labels[(xc, zc)], labels[(xt, zt)])
            parts = []
            for coeff, (xcp, zcp, xtp, ztp) in pauli_terms:
                label = "%sc ⊗ %st" % (labels[(xcp, zcp)], labels[(xtp, ztp)])
                sign, coeff_str = coeffstring(coeff, tol)
                parts.append('%s%s*%s' % (sign, coeff_str, label))
            expr = ' + '.join(parts).replace('+ -', '- ')
            print('  %s (%s) %s† = %s' % (gate, inp_label, gate, expr))

    xc_branches = any(len(e['out_xc']) > 1 for e in info)
    zc_branches = any(len(e['out_zc']) > 1 for e in info)
    xt_branches = any(len(e['out_xt']) > 1 for e in info)
    zt_branches = any(len(e['out_zt']) > 1 for e in info)
    
    if print_table:
        print("")
        print(" Pauli bits:")
        print(" %3s %3s %3s %3s   %7s %7s %7s %7s    %-10s   %-s" 
            % ('xc', 'zc', 'xt', 'zt', "xc'", "zc'", "xt'", "zt'", 'Negative', "Negative branch"))
        print(" " + "-"*90)
        for e in info:
            ix_c, iz_c, ix_t, iz_t = e['in_xc'], e['in_zc'], e['in_xt'], e['in_zt']
            ox_c, oz_c, ox_t, oz_t = e['out_xc'], e['out_zc'], e['out_xt'], e['out_zt']
            neg = 'yes' if (ix_c, iz_c, ix_t, iz_t) in neg_input_minterms else 'no'
            label = 'none'
            quads = [(ap,bp,cp,dp) for (a,b,c,d,ap,bp,cp,dp) in neg_branch_minterms
                    if (ix_c == a and iz_c == b and ix_t == c and iz_t == d)]
            if quads:
                parts = []
                for (ap,bp,cp,dp) in quads:
                    sub = []
                    if xc_branches: sub.append("xc'=%d" % ap)
                    if zc_branches: sub.append("zc'=%d" % bp)
                    if xt_branches: sub.append("xt'=%d" % cp)
                    if zt_branches: sub.append("zt'=%d" % dp)
                    if sub:
                        parts.append(", ".join(sub))
                if parts:
                    label = " | ".join(sorted(set(parts)))
            print('%3s %3s %3s %3s   %7s %7s %7s %7s     %-10s   %-s' %
                (str(ix_c), str(iz_c), str(ix_t), str(iz_t),
                str(ox_c), str(oz_c), str(ox_t), str(oz_t),
                neg, label))

    if try_simplify:
        try:
            import sympy as sp
            var_in = sp.symbols("xc zc xt zt", boolean=True)

            # xc'
            if xc_branches:
                mins = [(e['in_xc'], e['in_zc'], e['in_xt'], e['in_zt']) for e in info if len(e['out_xc']) > 1]
                expr = sp.simplify_logic(sp.SOPform(list(var_in), mins), form='dnf')
                print("\n xc' := (%s) | xc' <-> xc" % expr)
            else:
                mins = [(e['in_xc'], e['in_zc'], e['in_xt'], e['in_zt']) for e in info if (len(e['out_xc'])==1 and 1 in e['out_xc'])]
                expr = sp.simplify_logic(sp.SOPform(list(var_in), mins), form='dnf')
                print("\n xc' := %s" % expr)

            # zc'
            if zc_branches:
                mins = [(e['in_xc'], e['in_zc'], e['in_xt'], e['in_zt']) for e in info if len(e['out_zc']) > 1]
                expr = sp.simplify_logic(sp.SOPform(list(var_in), mins), form='dnf')
                print(" zc' := (%s) | zc' <-> zc" % expr)  # keep your label style if you prefer zc'
            else:
                mins = [(e['in_xc'], e['in_zc'], e['in_xt'], e['in_zt']) for e in info if (len(e['out_zc'])==1 and 1 in e['out_zc'])]
                expr = sp.simplify_logic(sp.SOPform(list(var_in), mins), form='dnf')
                print(" zc' := %s" % expr)

            # xt'
            if xt_branches:
                mins = [(e['in_xc'], e['in_zc'], e['in_xt'], e['in_zt']) for e in info if len(e['out_xt']) > 1]
                expr = sp.simplify_logic(sp.SOPform(list(var_in), mins), form='dnf')
                print(" xt' := (%s) | xt' <-> xt" % expr)
            else:
                mins = [(e['in_xc'], e['in_zc'], e['in_xt'], e['in_zt']) for e in info if (len(e['out_xt'])==1 and 1 in e['out_xt'])]
                expr = sp.simplify_logic(sp.SOPform(list(var_in), mins), form='dnf')
                print(" xt' := %s" % expr)

            # zt'
            if zt_branches:
                mins = [(e['in_xc'], e['in_zc'], e['in_xt'], e['in_zt']) for e in info if len(e['out_zt']) > 1]
                expr = sp.simplify_logic(sp.SOPform(list(var_in), mins), form='dnf')
                print(" zt' := (%s) | zt' <-> zt" % expr)
            else:
                mins = [(e['in_xc'], e['in_zc'], e['in_xt'], e['in_zt']) for e in info if (len(e['out_zt'])==1 and 1 in e['out_zt'])]
                expr = sp.simplify_logic(sp.SOPform(list(var_in), mins), form='dnf')
                print(" zt' := %s" % expr)

            # sign
            var_names = ["xc","zc","xt","zt"]
            if xc_branches: var_names.append("xc'")
            if zc_branches: var_names.append("zc'")
            if xt_branches: var_names.append("xt'")
            if zt_branches: var_names.append("zt'")
            reduced = []
            for (xci,zci,xti,zti,xcp,zcp,xtp,ztp) in neg_branch_minterms:
                row = [xci,zci,xti,zti]
                if xc_branches: row.append(xcp)
                if zc_branches: row.append(zcp)
                if xt_branches: row.append(xtp)
                if zt_branches: row.append(ztp)
                reduced.append(row)
            if reduced:
                vars_order = sp.symbols(" ".join(var_names), boolean=True)
                expr = sp.SOPform(list(vars_order), reduced)
                expr_simplified = sp.simplify_logic(expr, form='dnf', deep=True, force=True)
                print("\n s := %s\n" % (expr_simplified))
            else:
                print("\n s :=  r' = r\n")
        except Exception as e:
            print(' (sympy simplification skipped: %s)' % e)

def apply_gate(Gate, tolerance=1e-6, print_pauli=True, print_table=True):
    name, U = Gate
    if U.shape == (2,2):
        derive_single_qubit_pauli_constraints(Gate, tol=tolerance, print_pauli=print_pauli, print_table=print_table, try_simplify=True)
    elif U.shape == (4,4):
        derive_dual_qubits_pauli_constraints(Gate, tol=tolerance, print_pauli=print_pauli, print_table=print_table, try_simplify=True)
    else:
        raise ValueError("Gate unitary must be 2x2 or 4x4")

def print_clifford(tolerance=1e-6, print_pauli=True, print_table=True):
    H = ('H', (1/np.sqrt(2)) * np.array([[1, 1],
                                   [1, -1]], dtype=complex))
    S = ('S', np.array([[1, 0],
                  [0, 1j]], dtype=complex))
    CX = ('CX', np.array([
                    [1, 0, 0, 0], 
                    [0, 1, 0, 0], 
                    [0, 0, 0, 1],
                    [0, 0, 1, 0]], dtype=complex))
    apply_gate(H, tolerance=tolerance, print_pauli=print_pauli, print_table=print_table)
    apply_gate(S, tolerance=tolerance, print_pauli=print_pauli, print_table=print_table)
    apply_gate(CX, tolerance=tolerance, print_pauli=print_pauli, print_table=print_table)

def print_non_clifford(tolerance=1e-6, print_pauli=True, print_table=True):
    T = ('T', np.array([
                [1, 0],
                [0, np.exp(1j*np.pi/4)]], dtype=complex))
    CS = ('CS', np.diag([1, 1, 1, 1j]))
    CSqrtX = ('C√X', np.array([
                    [1, 0, 0, 0], 
                    [0, 1, 0, 0], 
                    [0, 0, (1 + 1j) / 2, (1 - 1j) / 2],
                    [0, 0, (1 - 1j) / 2, (1 + 1j) / 2]], dtype=complex))
    apply_gate(T, tolerance=tolerance, print_pauli=print_pauli, print_table=print_table)
    apply_gate(CS, tolerance=tolerance, print_pauli=print_pauli, print_table=print_table)
    apply_gate(CSqrtX, tolerance=tolerance, print_pauli=print_pauli, print_table=print_table)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tolerance', '-t', type=float, default=1e-6)
    parser.add_argument('--print-pauli', dest='print_pauli', action='store_true')
    parser.add_argument('--no-print-pauli', dest='print_pauli', action='store_false')
    parser.set_defaults(print_pauli=True)
    parser.add_argument('--print-table', dest='print_table', action='store_true')
    parser.add_argument('--no-print-table', dest='print_table', action='store_false')
    parser.set_defaults(print_table=True)
    parser.add_argument('--print-clifford', action='store_true')
    parser.add_argument('--print-non-clifford', action='store_true')
    parser.set_defaults(print_clifford=True)
    parser.set_defaults(print_non_clifford=True)
    args = parser.parse_args()

    if args.print_clifford:
        print_clifford(tolerance=args.tolerance, print_pauli=args.print_pauli, print_table=args.print_table)
    if args.print_non_clifford:
        print_non_clifford(tolerance=args.tolerance, print_pauli=args.print_pauli, print_table=args.print_table)

    theta = np.pi/7

    Rx = ('Rx(pi/7)', np.array([
        [np.cos(theta/2), -1j*np.sin(theta/2)],
        [-1j*np.sin(theta/2), np.cos(theta/2)]
    ], dtype=complex))

    Rz = ('Rz(pi/7)', np.array([
        [np.exp(-1j*theta/2), 0],
        [0, np.exp(1j*theta/2)]
    ], dtype=complex))

    apply_gate(Rx, tolerance=1e-8, print_pauli=True, print_table=True)
    apply_gate(Rz, tolerance=1e-8, print_pauli=True, print_table=True)