"""Integrators: rules that move the particles forward by one small time step dt.

Both functions have the same form:

    new_forces, new_potential_energy = step(system, forces, dt, compute_forces)

`forces` are the forces at the current positions. The system is changed in
place. The force function is passed in, so any of the four force versions
can be used (forces_python, forces_newton3, forces_numpy, forces_numba).
"""


def euler_step(system, forces, dt, compute_forces):
    """Forward Euler: the simplest rule, and it does NOT conserve energy.

    Both updates use the values from the start of the step:
        new position = position + velocity * dt
        new velocity = velocity + acceleration * dt
    The total energy slowly grows, step after step.
    """
    masses = system.masses[:, None]
    # Keep this order: positions first, with the old velocities.
    # Updating the velocities first would make this "symplectic Euler",
    # a different method that barely drifts.
    system.positions += system.velocities * dt
    system.velocities += (forces / masses) * dt
    system.positions %= system.box
    return compute_forces(system.positions, system.box, system.sigmas, system.epsilons)


def velocity_verlet_step(system, forces, dt, compute_forces):
    """Velocity Verlet: the standard MD integrator, which conserves energy well.

    The position update is the school formula x = x0 + v*t + 1/2*a*t^2.
    The velocity is then updated with the average of the old and the new
    acceleration. This is the "velocity" form of Verlet's method: positions
    and velocities are known at the same moments (the original "position
    Verlet" works with positions only).
    """
    masses = system.masses[:, None]
    # ZADATAK 3: napišite jedan korak metode velocity Verlet.
    # Ubrzanje je a = forces / masses (masses već ima oblik (N, 1), pa deljenje radi).
    # 1. Pomerite čestice: x_novo = x + v*dt + 1/2*a*dt^2 (menjajte system.positions).
    # 2. Vratite čestice u kutiju: system.positions %= system.box
    # 3. Izračunajte sile na novim pozicijama:
    #    new_forces, potential_energy = compute_forces(
    #        system.positions, system.box, system.sigmas, system.epsilons)
    # 4. Promenite brzine pomoću proseka starog i novog ubrzanja:
    #    v_novo = v + 1/2*(a + a_novo)*dt, gde je a_novo = new_forces / masses.
    # 5. Vratite new_forces, potential_energy.
    # Pogledajte euler_step iznad: isti koraci, za jednostavniju metodu.
    # Dok ovo ne uradite, čestice stoje u mestu (sile se računaju, ali ništa se ne pomera).
    return compute_forces(system.positions, system.box, system.sigmas, system.epsilons)
