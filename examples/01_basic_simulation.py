#!/usr/bin/env python3
"""
Example 01: Basic Simulation

Demonstrates fundamental simdyn usage:
- Creating dynamical systems
- Running simulations
- Using visualization utilities

Systems: Pendulum, Unicycle, Double Integrator
Outputs saved to: examples/outputs/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import simdyn as ds
from simdyn.visualization import (
    animate_pendulum,
    animate_unicycle,
    plot_energy,
    plot_phase_portrait,
    plot_phase_vector_field,
    plot_simulation,
    plot_states,
    plot_trajectory_2d,
    plot_unicycle_trajectory,
)

# Output directory
OUTPUT_DIR = Path(__file__).parent / "outputs"


def pendulum_example():
    """Pendulum free oscillation with phase portrait and animation."""
    print("=" * 60)
    print("Pendulum Free Oscillation")
    print("=" * 60)

    # Create normalized pendulum (m=l=g=1)
    pendulum = ds.create_normalized_pendulum()
    print(f"System: {pendulum}")
    print(f"Natural period: {pendulum.params.period:.3f}s")

    # Initial condition: 45° from bottom
    x0 = pendulum.pack_state(theta=np.deg2rad(45), omega=0.0)

    # Simulate 3 periods with no control
    t_final = 3 * pendulum.params.period
    t, x, u = pendulum.simulate(x0, lambda t, x: np.array([0.0]), (0, t_final), dt=0.01)

    # Check energy conservation
    E0 = pendulum.energy(x[0])["total"]
    Ef = pendulum.energy(x[-1])["total"]
    print(f"Energy drift: {abs(Ef - E0):.2e}")

    # --- Plots ---

    # State trajectories
    fig1, _ = plot_states(t, x, pendulum.state_names, title="Pendulum States")
    fig1.savefig(OUTPUT_DIR / "01a_pendulum_states.png", dpi=150)
    print("Saved: 01a_pendulum_states.png")

    # Phase portrait
    fig2, ax2 = plot_phase_portrait(x, xlabel="θ [rad]", ylabel="ω [rad/s]", title="Pendulum Phase Portrait")
    fig2.savefig(OUTPUT_DIR / "01a_pendulum_phase.png", dpi=150)
    print("Saved: 01a_pendulum_phase.png")

    # Vector field with trajectory
    fig3, ax3 = plot_phase_vector_field(pendulum, xlim=(-np.pi, np.pi), ylim=(-3, 3), n_grid=20)
    ax3.plot(x[:, 0], x[:, 1], "r-", lw=2, label="Trajectory")
    ax3.legend()
    ax3.set_title("Pendulum Vector Field")
    fig3.savefig(OUTPUT_DIR / "01a_pendulum_vectorfield.png", dpi=150)
    print("Saved: 01a_pendulum_vectorfield.png")

    # Energy plot
    fig4, _ = plot_energy(t, x, pendulum, title="Pendulum Energy")
    fig4.savefig(OUTPUT_DIR / "01a_pendulum_energy.png", dpi=150)
    print("Saved: 01a_pendulum_energy.png")

    # Animation (subsample for smaller file)
    anim = animate_pendulum(  # noqa: F841
        t[::2],
        x[::2],
        length=pendulum.params.l,
        title="Pendulum",
        interval=20,
        save_path=str(OUTPUT_DIR / "01a_pendulum.gif"),
    )

    plt.close("all")


def unicycle_example():
    """Unicycle following a circular path."""
    print("\n" + "=" * 60)
    print("Unicycle Circular Trajectory")
    print("=" * 60)

    # Create unicycle
    unicycle = ds.create_unicycle(v_bounds=(-2.0, 2.0), omega_bounds=(-2.0, 2.0))
    print(f"System: {unicycle}")

    # Start at origin facing +x
    x0 = unicycle.pack_state(position=np.array([0.0, 0.0]), heading=0.0)

    # Circle: v=1, R=2 -> omega=0.5
    R, v = 2.0, 1.0
    omega = v / R

    # Simulate one full circle
    t_final = 2 * np.pi * R / v
    t, x, u = unicycle.simulate(x0, lambda t, x: np.array([v, omega]), (0, t_final), dt=0.02)

    print(f"Circle completed in {t_final:.2f}s")
    print(f"Return error: {np.linalg.norm(x[-1, :2] - x[0, :2]):.4f}")

    # --- Plots ---

    # Trajectory with heading arrows
    fig1, ax1 = plot_unicycle_trajectory(x, title="Unicycle Circular Path", n_arrows=10, arrow_scale=0.3)
    # Reference circle
    theta = np.linspace(0, 2 * np.pi, 100)
    ax1.plot(R * np.sin(theta), R * (1 - np.cos(theta)), "g--", alpha=0.5, label="Ideal")
    ax1.legend()
    fig1.savefig(OUTPUT_DIR / "01b_unicycle_trajectory.png", dpi=150)
    print("Saved: 01b_unicycle_trajectory.png")

    # States over time
    fig2, _ = plot_states(t, x, unicycle.state_names, title="Unicycle States")
    fig2.savefig(OUTPUT_DIR / "01b_unicycle_states.png", dpi=150)
    print("Saved: 01b_unicycle_states.png")

    # Animation
    anim = animate_unicycle(  # noqa: F841
        t[::2], x[::2], title="Unicycle", interval=40, save_path=str(OUTPUT_DIR / "01b_unicycle.gif")
    )

    plt.close("all")


def double_integrator_example():
    """2D double integrator with PD control to origin."""
    print("\n" + "=" * 60)
    print("Double Integrator PD Control")
    print("=" * 60)

    # Create 2D system
    system = ds.DoubleIntegrator2D()
    print(f"System: {system}")

    # Initial: position (2,-1), velocity (1, 0.5)
    x0 = np.array([2.0, -1.0, 1.0, 0.5])
    x_target = np.zeros(4)

    # PD controller
    Kp, Kd = 2.0, 3.0

    def controller(t, x):  # noqa: ARG001
        return Kp * (x_target[:2] - x[:2]) + Kd * (x_target[2:] - x[2:])

    # Simulate
    t, x, u = system.simulate(x0, controller, (0, 5), dt=0.02)

    print(f"Initial: pos=({x0[0]:.1f}, {x0[1]:.1f})")
    print(f"Final:   pos=({x[-1, 0]:.4f}, {x[-1, 1]:.4f})")

    # --- Plots ---

    # 2D trajectory
    fig1, ax1 = plot_trajectory_2d(
        x, idx_x=0, idx_y=1, xlabel="X [m]", ylabel="Y [m]", title="Double Integrator Trajectory"
    )
    ax1.plot(0, 0, "k*", ms=15, label="Target")
    ax1.legend()
    fig1.savefig(OUTPUT_DIR / "01c_double_integrator_trajectory.png", dpi=150)
    print("Saved: 01c_double_integrator_trajectory.png")

    # Full simulation view
    fig2, _ = plot_simulation(
        t, x, u, state_names=["px", "py", "vx", "vy"], control_names=["ax", "ay"], title="Double Integrator Simulation"
    )
    fig2.savefig(OUTPUT_DIR / "01c_double_integrator_simulation.png", dpi=150)
    print("Saved: 01c_double_integrator_simulation.png")

    plt.close("all")


def main():
    print("#" * 60)
    print("# simdyn Example 01: Basic Simulation")
    print("#" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"\nOutputs: {OUTPUT_DIR.absolute()}\n")

    pendulum_example()
    unicycle_example()
    double_integrator_example()

    print("\n" + "=" * 60)
    print(f"Done! Check {OUTPUT_DIR.name}/ for outputs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
