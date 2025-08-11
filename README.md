# Pauli Deriver
A tool to transform 1/2-qubit unitary into Pauli operators producing:
- Pauli-string decomposition of $\ U P U^\dagger\$ for all Paulis $\ P \$,
- Pauli-bits table $(x,z)$ or $(x_c, z_c, x_t, z_t)$ showing where outputs are deterministic vs. branching,
- Simplified update equations for the output Pauli bits (e.g., $\ x’, z’, \ldots \$) and their collective sign.

# Requirements
```
python >= 3.8
pip install numpy sympy
```

# Usage
Run the script directly with the following usage:
```python
apply_gate.py [-h] [--tolerance TOLERANCE] [--print-pauli] [--no-print-pauli] [--print-table] [--no-print-table] [--print-clifford] [--print-non-clifford]
```

# API
```python
from pauli_tool import apply_gate, print_clifford, print_non_clifford
import numpy as np

# Built-in sets
print_clifford(tolerance=1e-6, print_pauli=True, print_table=True)
print_non_clifford(tolerance=1e-6, print_pauli=True, print_table=True)

# Custim-unitary gates
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
```

## Output sample
```python
--------[ Gate Rx(pi/7) (single qubit) ]--------------------
 Pauli strings:
   Rx(pi/7) (I) Rx(pi/7)† = 1*I
   Rx(pi/7) (Z) Rx(pi/7)† = 0.901*Z - 0.434*Y
   Rx(pi/7) (X) Rx(pi/7)† = 1*X
   Rx(pi/7) (Y) Rx(pi/7)† = 0.434*Z + 0.901*Y

 Pauli bits:
  x  z    x'      z'        Negative     Negative branch
 ----------------------------------------------------------------------
  0  0    {0}     {0}       no           none
  0  1    {0, 1}  {1}       yes          x'=1
  1  0    {1}     {0}       no           none
  1  1    {0, 1}  {1}       no           none

 Update rules:
  x' := (z) | x' <-> x
  z' := z

  s := x' & z & ~x

--------[ Gate Rz(pi/7) (single qubit) ]--------------------
 Pauli strings:
   Rz(pi/7) (I) Rz(pi/7)† = 1*I
   Rz(pi/7) (Z) Rz(pi/7)† = 1*Z
   Rz(pi/7) (X) Rz(pi/7)† = 0.901*X + 0.434*Y
   Rz(pi/7) (Y) Rz(pi/7)† = -0.434*X + 0.901*Y

 Pauli bits:
  x  z    x'      z'        Negative     Negative branch
 ----------------------------------------------------------------------
  0  0    {0}     {0}       no           none
  0  1    {0}     {1}       no           none
  1  0    {1}     {0, 1}    no           none
  1  1    {1}     {0, 1}    yes          z'=0

 Update rules:
  x' := x
  z' := (x) | z' <-> z

  s := x & z & ~z'
```



