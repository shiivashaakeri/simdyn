"""
Visualization utilities for simdyn.

This module provides functions for visualizing simulation results:

Plotting (requires matplotlib)
------------------------------
- plot_states: Time series of states
- plot_controls: Time series of controls
- plot_simulation: Combined states and controls
- plot_phase_portrait: 2D phase space
- plot_phase_vector_field: Vector field for 2-state systems
- plot_trajectory_2d: 2D position trajectory
- plot_trajectory_3d: 3D position trajectory
- plot_unicycle_trajectory: Unicycle with heading arrows
- plot_rocket_trajectory: Rocket 3D trajectory
- plot_energy: Energy over time
- plot_constraints: Constraint values
- plot_comparison: Compare multiple trajectories

Animation (requires matplotlib)
-------------------------------
- animate_pendulum: Swinging pendulum
- animate_cartpole: Cart-pole balancing
- animate_unicycle: Mobile robot
- animate_rocket_2d: Rocket landing (side view)
- animate_point_mass_2d: 2D double integrator

Example
-------
>>> import simdyn as ds
>>> from simdyn.visualization import plot_states, animate_pendulum
>>>
>>> pendulum = ds.create_normalized_pendulum()
>>> x0 = np.array([0.5, 0.0])
>>> t, x, u = pendulum.simulate(x0, lambda t,x: np.array([0.0]), (0,10), 0.02)
>>>
>>> # Static plot
>>> fig, axes = plot_states(t, x, pendulum.state_names)
>>> plt.show()
>>>
>>> # Animation
>>> anim = animate_pendulum(t, x, length=pendulum.params.l)
>>> plt.show()
"""

# Check matplotlib availability
try:
    import matplotlib  # noqa: ICN001, F401

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

if HAS_MATPLOTLIB:
    # Plotting functions
    # Animation functions
    from simdyn.visualization.animate import (
        animate_cartpole,
        animate_pendulum,
        animate_point_mass_2d,
        animate_rocket_2d,
        animate_unicycle,
        save_animation,
    )
    from simdyn.visualization.plotters import (
        plot_comparison,
        plot_constraints,
        plot_controls,
        plot_energy,
        plot_phase_portrait,
        plot_phase_vector_field,
        plot_rocket_trajectory,
        plot_simulation,
        plot_states,
        plot_trajectory_2d,
        plot_trajectory_3d,
        plot_unicycle_trajectory,
    )

    __all__ = [
        "animate_cartpole",
        # Animation
        "animate_pendulum",
        "animate_point_mass_2d",
        "animate_rocket_2d",
        "animate_unicycle",
        "plot_comparison",
        "plot_constraints",
        "plot_controls",
        "plot_energy",
        "plot_phase_portrait",
        "plot_phase_vector_field",
        "plot_rocket_trajectory",
        "plot_simulation",
        # Plotting
        "plot_states",
        "plot_trajectory_2d",
        "plot_trajectory_3d",
        "plot_unicycle_trajectory",
        "save_animation",
    ]
else:
    # Empty exports when matplotlib not available
    __all__ = []

    def _matplotlib_required(*args, **kwargs):  # noqa: ARG001
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")

    # Create stub functions that raise helpful errors
    plot_states = _matplotlib_required
    plot_controls = _matplotlib_required
    plot_simulation = _matplotlib_required
    plot_phase_portrait = _matplotlib_required
    plot_phase_vector_field = _matplotlib_required
    plot_trajectory_2d = _matplotlib_required
    plot_trajectory_3d = _matplotlib_required
    plot_unicycle_trajectory = _matplotlib_required
    plot_rocket_trajectory = _matplotlib_required
    plot_energy = _matplotlib_required
    plot_constraints = _matplotlib_required
    plot_comparison = _matplotlib_required
    animate_pendulum = _matplotlib_required
    animate_cartpole = _matplotlib_required
    animate_unicycle = _matplotlib_required
    animate_rocket_2d = _matplotlib_required
    animate_point_mass_2d = _matplotlib_required
    save_animation = _matplotlib_required
