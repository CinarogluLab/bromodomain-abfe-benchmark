"""Full-precision Rocklin terms; no implicit choice of counterion convention.

All input charges are in e, lengths nm, volumes nm^3, and residual
integrated potentials in kJ nm^3/(mol e). Returned energies are kcal/mol.
The PB integration domain and MD periodic volume are deliberately separate.
Missing discrete-solvent data stays missing, rather than becoming zero.
"""
import math

K = 138.93545585
XI_LS = -2.837297
XI_CB = -2.38008


def write_precise_pqr(atoms, path):
    """Write APBS whitespace PQR with charges preserved beyond the default 4 decimals."""
    with open(path,'w') as f:
        for serial,a in enumerate(atoms,1):
            x,y,z=a.position
            f.write(f'ATOM {serial} {a.name} {a.resname} {a.resid} '
                    f'{x:.6f} {y:.6f} {z:.6f} {a.charge:.10f} {a.radius:.6f}\n')


def wigner_inverse_nm(box_vectors_nm, *, alpha_scale=5., real_shell=5, reciprocal_shell=12):
    """Ewald regularized self-potential for a general periodic cell (1/nm).

    Box vectors are rows. Includes the uniform neutralizing background.
    The cubic limit is XI_LS/L; convergence must be checked for cell shapes used.
    """
    import itertools
    import numpy as np
    from scipy.special import erfc
    box = np.asarray(box_vectors_nm, dtype=float)
    if box.shape != (3, 3):
        raise ValueError('Expected three box vectors as a 3 by 3 matrix')
    v = abs(float(np.linalg.det(box)))
    if v <= 0:
        raise ValueError('Degenerate periodic box')
    alpha = alpha_scale/v**(1/3)
    def indices(shell):
        a = np.array(list(itertools.product(range(-shell,shell+1),repeat=3)))
        return a[np.any(a!=0,axis=1)]
    r = np.linalg.norm(indices(real_shell)@box,axis=1)
    reciprocal = 2*np.pi*np.linalg.inv(box).T
    k2 = np.sum((indices(reciprocal_shell)@reciprocal)**2,axis=1)
    return float(np.sum(erfc(alpha*r)/r) + 4*np.pi/v*np.sum(np.exp(-k2/(4*alpha**2))/k2)
                 - 2*alpha/np.sqrt(np.pi) - np.pi/(alpha**2*v))


def centered_coulomb_integral_nm2(lengths_nm):
    """Integral of 1/r over a rectangular domain centered on the point charge.

    Integrate z analytically and x,y with adaptive quadrature. This avoids
    imposing the cubic XI_CB reference on an orthorhombic APBS domain.
    """
    from scipy.integrate import quad
    import numpy as np
    a, b, c = np.asarray(lengths_nm, dtype=float)/2
    if min(a,b,c) <= 0:
        raise ValueError('Domain lengths must be positive')
    inner = lambda x: quad(lambda y: math.asinh(c/math.hypot(x,y)),0,b,
                           epsabs=1e-9,epsrel=1e-9)[0]
    return 8*quad(inner,0,a,epsabs=1e-9,epsrel=1e-9)[0]


def terms(*, volume_nm3, q_ligand, q_environment, epsilon, water_count,
          gamma_e_nm2=None, IP=None, IL=None, IL_SLV=None, lattice_inverse_nm=None):
    if not (volume_nm3 > 0 and epsilon > 1 and water_count >= 0):
        raise ValueError('Positive MD volume, epsilon > 1 and nonnegative water count required')
    v, q, p = volume_nm3, q_ligand, q_environment
    dq2 = (p + q)**2 - p**2
    xi = XI_LS/v**(1/3) if lattice_inverse_nm is None else lattice_inverse_nm
    net = -xi * K / 2 * dq2
    out = {'NET': net / 4.184, 'USV': -net*(1-1/epsilon)/4.184,
           'NET_USV': net/epsilon/4.184, 'RIP': None, 'EMP': None,
           'DSC': None, 'ANA': None, 'total': None}
    if gamma_e_nm2 is not None:
        out['DSC'] = -(2*math.pi/3)*K*gamma_e_nm2*q*water_count/v/4.184
    if IP is not None and IL is not None:
        out['RIP'] = ((IP+IL)*(p+q)-IP*p)/v/4.184
    if IL_SLV is not None:
        if q == 0:
            raise ValueError('EMP effective-radius expression requires nonzero ligand charge')
        radius_squared = IL_SLV / (K/2 * (4*math.pi/3)*(1-1/epsilon)*q)
        if radius_squared < 0:
            raise ValueError('Negative effective radius squared: validate PB integrals')
        out['EMP'] = -K/2*(16*math.pi**2/45)*(1-1/epsilon)*dq2*radius_squared**2.5/v**2/4.184
    if all(out[k] is not None for k in ('NET_USV', 'RIP', 'EMP')):
        out['ANA'] = sum(out[k] for k in ('NET_USV', 'RIP', 'EMP'))
        if out['DSC'] is not None:
            out['total'] = out['ANA'] + out['DSC']
    return out


def from_rocklinc(corr, *, gamma_e_nm2, q_environment=None):
    """Read unrounded quantities from an existing object without changing it."""
    import quantities as pq
    potential_units = pq.J*pq.nm**3/(pq.mol*pq.e)
    p = float(corr.protein_netq.rescale(pq.e)) if q_environment is None else q_environment
    out = terms(volume_nm3=float(corr.vol.rescale(pq.nm**3)),
                q_ligand=float(corr.lig_netq.rescale(pq.e)), q_environment=p,
                epsilon=corr.water.epsilon_S, water_count=corr.NS,
                gamma_e_nm2=gamma_e_nm2,
                **{k: float(getattr(corr, k).rescale(potential_units))/1000 for k in ('IP', 'IL', 'IL_SLV')})
    out.update(md_volume_nm3=float(corr.vol.rescale(pq.nm**3)),
               library_integration_volume_nm3=float(corr.apbs_vol.rescale(pq.nm**3)),
               **{k+'_kJ_nm3_per_mol_e': float(getattr(corr, k).rescale(potential_units))/1000 for k in ('IP', 'IL', 'IL_SLV')})
    return out


def integrate_dx(path, *, temperature=298.15):
    """Trapezoidal integral over the nodal APBS domain, plus legacy comparison.

    This is a numerical diagnostic, not a validated PB correction by itself.
    APBS potential units are kBT/e; gridData lengths are Angstrom.
    """
    import numpy as np
    from gridData import Grid
    g = Grid(str(path))
    a = np.asarray(g.grid, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError('Nonfinite APBS potential values')
    spacing = np.abs(np.asarray(g.delta, dtype=float))/10
    # Integrate the last dimension repeatedly, retaining all endpoint half-weights.
    integrated = a
    for axis in (2, 1, 0):
        integrated = np.trapezoid(integrated, dx=spacing[axis], axis=axis)
    kT = 0.0083144621*temperature
    return dict(shape=list(a.shape), spacing_nm=spacing.tolist(),
                domain_lengths_nm=((np.array(a.shape)-1)*spacing).tolist(),
                nodal_domain_volume_nm3=float(np.prod((np.array(a.shape)-1)*spacing)),
                legacy_cell_volume_nm3=float(np.prod(np.array(a.shape)*spacing)),
                integral_kJ_nm3_per_mol_e=float(integrated*kT),
                legacy_integral_kJ_nm3_per_mol_e=float(a.sum()*np.prod(spacing)*kT))


def residuals_from_dx(*, ligand_heterogeneous, ligand_homogeneous, protein_heterogeneous=None,
                      q_ligand_pb, q_protein_pb, epsilon, temperature=298.15):
    """Re-integrate preserved APBS grids with consistent reference domains.

    Reference point charge is centered in the APBS domain. This is the
    centered-domain convention, whose convergence still needs checking on
    protein systems. This function does not validate ion treatment or radii.
    """
    import numpy as np
    het=integrate_dx(ligand_heterogeneous,temperature=temperature)
    hom=integrate_dx(ligand_homogeneous,temperature=temperature)
    if not np.allclose(het['domain_lengths_nm'],hom['domain_lengths_nm'],rtol=1e-6):
        raise ValueError('Heterogeneous and homogeneous reference domains differ')
    reference=centered_coulomb_integral_nm2(het['domain_lengths_nm'])
    IL=het['integral_kJ_nm3_per_mol_e']-K*q_ligand_pb/epsilon*reference
    IL_hom=hom['integral_kJ_nm3_per_mol_e']-K*q_ligand_pb*reference
    IP=0.
    if protein_heterogeneous is not None:
        protein=integrate_dx(protein_heterogeneous,temperature=temperature)
        if not np.allclose(het['domain_lengths_nm'],protein['domain_lengths_nm'],rtol=1e-6):
            raise ValueError('Protein and ligand domains differ')
        IP=protein['integral_kJ_nm3_per_mol_e']-K*q_protein_pb/epsilon*reference
    return dict(IP=IP,IL=IL,IL_SLV=IL-IL_hom,
                pb_volume_nm3=het['nodal_domain_volume_nm3'],
                reference_coulomb_integral_nm2=reference,quadrature='nodal_trapezoidal')
