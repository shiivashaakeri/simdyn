"""
Plotting utilities for simdyn.

This module provides functions for visualizing simulation results:
- Time series plots
- Phase portraits
- 2D and 3D trajectory plots
- Constraint visualization
- Energy plots
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Check for matplotlib availability
try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    Figure = Any
    Axes = Any


def _check_matplotlib():
    """Raise error if matplotlib not available."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")


# =============================================================================
# Time Series Plots
# =============================================================================


def plot_states(
    t: np.ndarray,
    x: np.ndarray,
    state_names: Optional[List[str]] = None,
    title: str = "State Trajectories",
    figsize: Tuple[float, float] = (10, 8),
    sharex: bool = True,
    grid: bool = True,
) -> Tuple[Figure, np.ndarray]:
    """
    Plot state trajectories over time.

    Parameters
    ----------
    t : np.ndarray, shape (N,)
        Time array.
    x : np.ndarray, shape (N, n_state)
        State trajectory.
    state_names : list of str, optional
        Names for each state. Defaults to x_0, x_1, etc.
    title : str
        Figure title.
    figsize : tuple
        Figure size (width, height).
    sharex : bool
        Share x-axis across subplots.
    grid : bool
        Show grid.

    Returns
    -------
    fig : Figure
        Matplotlib figure.
    axes : ndarray of Axes
        Array of subplot axes.

    Example
    -------
    >>> t, x, u = system.simulate(x0, controller, (0, 10), dt=0.01)
    >>> fig, axes = plot_states(t, x, system.state_names)
    >>> plt.show()
    """
    _check_matplotlib()

    n_state = x.shape[1]

    if state_names is None:
        state_names = [f"x_{i}" for i in range(n_state)]

    # Determine subplot layout
    n_cols = min(2, n_state)
    n_rows = int(np.ceil(n_state / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=sharex)
    axes = np.atleast_1d(axes).flatten()

    for i in range(n_state):
        axes[i].plot(t, x[:, i], linewidth=1.5)
        axes[i].set_ylabel(state_names[i])
        if grid:
            axes[i].grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(n_state, len(axes)):
        axes[i].set_visible(False)

    # Set x-label on bottom row
    for i in range(n_cols):
        idx = (n_rows - 1) * n_cols + i
        if idx < len(axes):
            axes[idx].set_xlabel("Time [s]")

    fig.suptitle(title)
    fig.tight_layout()

    return fig, axes


def plot_controls(
    t: np.ndarray,
    u: np.ndarray,
    control_names: Optional[List[str]] = None,
    title: str = "Control Inputs",
    figsize: Tuple[float, float] = (10, 6),
    sharex: bool = True,
    grid: bool = True,
    bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[Figure, np.ndarray]:
    """
    Plot control inputs over time.

    Parameters
    ----------
    t : np.ndarray, shape (N,)
        Time array (should be len(u) or len(u)+1).
    u : np.ndarray, shape (N-1, n_control) or (N, n_control)
        Control trajectory.
    control_names : list of str, optional
        Names for each control.
    title : str
        Figure title.
    figsize : tuple
        Figure size.
    sharex : bool
        Share x-axis.
    grid : bool
        Show grid.
    bounds : tuple of arrays, optional
        (lower, upper) bounds to show as dashed lines.

    Returns
    -------
    fig : Figure
    axes : ndarray of Axes
    """
    _check_matplotlib()

    n_control = u.shape[1]

    # Adjust time array if needed
    t_u = t[:-1] if len(t) == len(u) + 1 else t[:len(u)]

    if control_names is None:
        control_names = [f"u_{i}" for i in range(n_control)]

    n_cols = min(2, n_control)
    n_rows = int(np.ceil(n_control / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=sharex)
    axes = np.atleast_1d(axes).flatten()

    for i in range(n_control):
        axes[i].plot(t_u, u[:, i], linewidth=1.5)
        axes[i].set_ylabel(control_names[i])

        if bounds is not None:
            lb, ub = bounds
            if np.isfinite(lb[i]):
                axes[i].axhline(lb[i], color="r", linestyle="--", alpha=0.7, label="bounds")
            if np.isfinite(ub[i]):
                axes[i].axhline(ub[i], color="r", linestyle="--", alpha=0.7)

        if grid:
            axes[i].grid(True, alpha=0.3)

    for i in range(n_control, len(axes)):
        axes[i].set_visible(False)

    for i in range(n_cols):
        idx = (n_rows - 1) * n_cols + i
        if idx < len(axes):
            axes[idx].set_xlabel("Time [s]")

    fig.suptitle(title)
    fig.tight_layout()

    return fig, axes


def plot_simulation(
    t: np.ndarray,
    x: np.ndarray,
    u: np.ndarray,
    state_names: Optional[List[str]] = None,
    control_names: Optional[List[str]] = None,
    title: str = "Simulation Results",
    figsize: Tuple[float, float] = (12, 8),
) -> Tuple[Figure, Tuple[np.ndarray, np.ndarray]]:
    """
    Plot both states and controls from a simulation.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    x : np.ndarray
        State trajectory.
    u : np.ndarray
        Control trajectory.
    state_names : list, optional
        State names.
    control_names : list, optional
        Control names.
    title : str
        Figure title.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig : Figure
    (state_axes, control_axes) : tuple of axes arrays
    """
    _check_matplotlib()

    n_state = x.shape[1]
    n_control = u.shape[1]

    if state_names is None:
        state_names = [f"x_{i}" for i in range(n_state)]
    if control_names is None:
        control_names = [f"u_{i}" for i in range(n_control)]

    # Layout: states on left, controls on right
    n_rows = max(n_state, n_control)

    fig, axes = plt.subplots(n_rows, 2, figsize=figsize, sharex=True)

    # Time for controls
    t_u = t[:-1] if len(t) == len(u) + 1 else t[: len(u)]

    # Plot states
    for i in range(n_state):
        axes[i, 0].plot(t, x[:, i], "b-", linewidth=1.5)
        axes[i, 0].set_ylabel(state_names[i])
        axes[i, 0].grid(True, alpha=0.3)

    for i in range(n_state, n_rows):
        axes[i, 0].set_visible(False)

    # Plot controls
    for i in range(n_control):
        axes[i, 1].plot(t_u, u[:, i], "r-", linewidth=1.5)
        axes[i, 1].set_ylabel(control_names[i])
        axes[i, 1].grid(True, alpha=0.3)

    for i in range(n_control, n_rows):
        axes[i, 1].set_visible(False)

    # Labels
    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")
    axes[0, 0].set_title("States")
    axes[0, 1].set_title("Controls")

    fig.suptitle(title)
    fig.tight_layout()

    return fig, (axes[:, 0], axes[:, 1])


# =============================================================================
# Phase Portraits
# =============================================================================


def plot_phase_portrait(
    x: np.ndarray,
    idx_x: int = 0,
    idx_y: int = 1,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: str = "Phase Portrait",
    figsize: Tuple[float, float] = (8, 6),
    ax: Optional[Axes] = None,
    show_start: bool = True,
    show_end: bool = True,
    **plot_kwargs,
) -> Tuple[Figure, Axes]:
    """
    Plot 2D phase portrait.

    Parameters
    ----------
    x : np.ndarray, shape (N, n_state)
        State trajectory.
    idx_x : int
        State index for x-axis.
    idx_y : int
        State index for y-axis.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    title : str
        Plot title.
    figsize : tuple
        Figure size.
    ax : Axes, optional
        Existing axes to plot on.
    show_start : bool
        Mark start point.
    show_end : bool
        Mark end point.
    **plot_kwargs
        Additional arguments to plt.plot().

    Returns
    -------
    fig : Figure
    ax : Axes
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Default plot style
    if "linewidth" not in plot_kwargs:
        plot_kwargs["linewidth"] = 1.5

    ax.plot(x[:, idx_x], x[:, idx_y], **plot_kwargs)

    if show_start:
        ax.plot(x[0, idx_x], x[0, idx_y], "go", markersize=10, label="Start")
    if show_end:
        ax.plot(x[-1, idx_x], x[-1, idx_y], "ro", markersize=10, label="End")

    ax.set_xlabel(xlabel or f"x[{idx_x}]")
    ax.set_ylabel(ylabel or f"x[{idx_y}]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if show_start or show_end:
        ax.legend()

    return fig, ax


def plot_phase_vector_field(
    system,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    u: np.ndarray = None,
    n_grid: int = 20,
    ax: Optional[Axes] = None,
    figsize: Tuple[float, float] = (8, 6),
    **quiver_kwargs,
) -> Tuple[Figure, Axes]:
    """
    Plot vector field for a 2-state system.

    Parameters
    ----------
    system : DynamicalSystem
        Must have n_state = 2.
    xlim : tuple
        (min, max) for first state.
    ylim : tuple
        (min, max) for second state.
    u : np.ndarray, optional
        Control input (default: zeros).
    n_grid : int
        Number of grid points per axis.
    ax : Axes, optional
        Existing axes.
    figsize : tuple
        Figure size.
    **quiver_kwargs
        Additional arguments to ax.quiver().

    Returns
    -------
    fig : Figure
    ax : Axes
    """
    _check_matplotlib()

    if system.n_state != 2:
        raise ValueError(f"Vector field requires n_state=2, got {system.n_state}")

    if u is None:
        u = np.zeros(system.n_control)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Create grid
    x1 = np.linspace(xlim[0], xlim[1], n_grid)
    x2 = np.linspace(ylim[0], ylim[1], n_grid)
    X1, X2 = np.meshgrid(x1, x2)

    # Compute vector field
    DX1 = np.zeros_like(X1)
    DX2 = np.zeros_like(X2)

    for i in range(n_grid):
        for j in range(n_grid):
            x = np.array([X1[i, j], X2[i, j]])
            x_dot = system.f(x, u)
            DX1[i, j] = x_dot[0]
            DX2[i, j] = x_dot[1]

    # Normalize for better visualization
    magnitude = np.sqrt(DX1**2 + DX2**2)
    magnitude = np.maximum(magnitude, 1e-10)
    DX1_norm = DX1 / magnitude
    DX2_norm = DX2 / magnitude

    # Plot
    ax.quiver(X1, X2, DX1_norm, DX2_norm, magnitude, cmap="viridis", **quiver_kwargs)
    ax.set_xlabel(system.state_names[0] if hasattr(system, "state_names") else "x[0]")
    ax.set_ylabel(system.state_names[1] if hasattr(system, "state_names") else "x[1]")
    ax.set_title("Phase Vector Field")

    return fig, ax


# =============================================================================
# 2D Trajectory Plots
# =============================================================================


def plot_trajectory_2d(
    x: np.ndarray,
    idx_x: int = 0,
    idx_y: int = 1,
    xlabel: str = "X",
    ylabel: str = "Y",
    title: str = "2D Trajectory",
    figsize: Tuple[float, float] = (8, 6),
    ax: Optional[Axes] = None,
    show_start: bool = True,
    show_end: bool = True,
    equal_aspect: bool = True,
    **plot_kwargs,
) -> Tuple[Figure, Axes]:
    """
    Plot 2D position trajectory.

    Parameters
    ----------
    x : np.ndarray
        State trajectory.
    idx_x : int
        Index for x position.
    idx_y : int
        Index for y position.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    title : str
        Title.
    figsize : tuple
        Figure size.
    ax : Axes, optional
        Existing axes.
    show_start : bool
        Mark start.
    show_end : bool
        Mark end.
    equal_aspect : bool
        Equal axis scaling.
    **plot_kwargs
        Arguments to plt.plot().

    Returns
    -------
    fig : Figure
    ax : Axes
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    if "linewidth" not in plot_kwargs:
        plot_kwargs["linewidth"] = 2

    ax.plot(x[:, idx_x], x[:, idx_y], **plot_kwargs)

    if show_start:
        ax.plot(x[0, idx_x], x[0, idx_y], "go", markersize=10, label="Start")
    if show_end:
        ax.plot(x[-1, idx_x], x[-1, idx_y], "ro", markersize=10, label="End")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if equal_aspect:
        ax.set_aspect("equal")

    if show_start or show_end:
        ax.legend()

    return fig, ax


def plot_unicycle_trajectory(
    x: np.ndarray,
    title: str = "Unicycle Trajectory",
    figsize: Tuple[float, float] = (8, 8),
    ax: Optional[Axes] = None,
    n_arrows: int = 10,
    arrow_scale: float = 0.3,
) -> Tuple[Figure, Axes]:
    """
    Plot unicycle trajectory with heading arrows.

    Parameters
    ----------
    x : np.ndarray, shape (N, 3)
        State trajectory [x, y, θ].
    title : str
        Title.
    figsize : tuple
        Figure size.
    ax : Axes, optional
        Existing axes.
    n_arrows : int
        Number of heading arrows to draw.
    arrow_scale : float
        Arrow length scale.

    Returns
    -------
    fig : Figure
    ax : Axes
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Plot path
    ax.plot(x[:, 0], x[:, 1], "b-", linewidth=2, label="Path")
    ax.plot(x[0, 0], x[0, 1], "go", markersize=10, label="Start")
    ax.plot(x[-1, 0], x[-1, 1], "ro", markersize=10, label="End")

    # Draw heading arrows
    indices = np.linspace(0, len(x) - 1, n_arrows, dtype=int)
    for i in indices:
        theta = x[i, 2]
        dx = arrow_scale * np.cos(theta)
        dy = arrow_scale * np.sin(theta)
        ax.arrow(
            x[i, 0], x[i, 1], dx, dy, head_width=0.1 * arrow_scale, head_length=0.05 * arrow_scale, fc="red", ec="red"
        )

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig, ax


# =============================================================================
# 3D Trajectory Plots
# =============================================================================


def plot_trajectory_3d(
    x: np.ndarray,
    idx_x: int = 0,
    idx_y: int = 1,
    idx_z: int = 2,
    xlabel: str = "X",
    ylabel: str = "Y",
    zlabel: str = "Z",
    title: str = "3D Trajectory",
    figsize: Tuple[float, float] = (10, 8),
    ax: Optional[Axes] = None,
    show_start: bool = True,
    show_end: bool = True,
    show_projection: bool = False,
    **plot_kwargs,
) -> Tuple[Figure, Axes]:
    """
    Plot 3D trajectory.

    Parameters
    ----------
    x : np.ndarray
        State trajectory.
    idx_x, idx_y, idx_z : int
        Indices for x, y, z positions.
    xlabel, ylabel, zlabel : str
        Axis labels.
    title : str
        Title.
    figsize : tuple
        Figure size.
    ax : Axes3D, optional
        Existing 3D axes.
    show_start : bool
        Mark start.
    show_end : bool
        Mark end.
    show_projection : bool
        Show projection onto xy plane.
    **plot_kwargs
        Arguments to plot().

    Returns
    -------
    fig : Figure
    ax : Axes3D
    """
    _check_matplotlib()

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    if "linewidth" not in plot_kwargs:
        plot_kwargs["linewidth"] = 2

    ax.plot(x[:, idx_x], x[:, idx_y], x[:, idx_z], **plot_kwargs)

    if show_start:
        ax.scatter([x[0, idx_x]], [x[0, idx_y]], [x[0, idx_z]], c="green", s=100, label="Start")
    if show_end:
        ax.scatter([x[-1, idx_x]], [x[-1, idx_y]], [x[-1, idx_z]], c="red", s=100, label="End")

    if show_projection:
        z_min = ax.get_zlim()[0]
        ax.plot(x[:, idx_x], x[:, idx_y], z_min * np.ones(len(x)), "k--", alpha=0.3, linewidth=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)

    if show_start or show_end:
        ax.legend()

    return fig, ax


def plot_rocket_trajectory(
    x: np.ndarray,
    title: str = "Rocket Trajectory",
    figsize: Tuple[float, float] = (10, 8),
    ax: Optional[Axes] = None,
    show_glide_slope: bool = False,
    gamma_gs: Optional[float] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot rocket trajectory (3-DoF or 6-DoF).

    Assumes state has position at indices 1:4 (after mass).

    Parameters
    ----------
    x : np.ndarray
        Rocket state trajectory.
    title : str
        Title.
    figsize : tuple
        Figure size.
    ax : Axes3D, optional
        Existing 3D axes.
    show_glide_slope : bool
        Draw glide slope cone.
    gamma_gs : float, optional
        Glide slope angle [rad].

    Returns
    -------
    fig : Figure
    ax : Axes3D
    """
    _check_matplotlib()

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure

    # Extract positions (indices 1, 2, 3 for rocket states)
    # x-axis is altitude (up), y is East, z is North
    alt = x[:, 1]
    east = x[:, 2]
    north = x[:, 3]

    # Plot trajectory
    ax.plot(east, north, alt, "b-", linewidth=2, label="Trajectory")
    ax.scatter([east[0]], [north[0]], [alt[0]], c="green", s=100, label="Start")
    ax.scatter([east[-1]], [north[-1]], [alt[-1]], c="red", s=100, label="Land")
    ax.scatter([0], [0], [0], c="black", s=150, marker="*", label="Target")

    # Glide slope cone
    if show_glide_slope and gamma_gs is not None:
        theta = np.linspace(0, 2 * np.pi, 50)
        max_alt = np.max(alt)
        altitudes = np.linspace(0, max_alt, 10)
        for z in altitudes[1::2]:
            r = z * np.tan(gamma_gs)
            ax.plot(r * np.cos(theta), r * np.sin(theta), z * np.ones(50), "g--", alpha=0.3)

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_zlabel("Altitude [m]")
    ax.set_title(title)
    ax.legend()

    return fig, ax


# =============================================================================
# Energy Plots
# =============================================================================


def plot_energy(
    t: np.ndarray,
    x: np.ndarray,
    system,
    title: str = "Energy",
    figsize: Tuple[float, float] = (10, 6),
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot energy over time for a system.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    x : np.ndarray
        State trajectory.
    system : DynamicalSystem
        Must have energy() method.
    title : str
        Title.
    figsize : tuple
        Figure size.
    ax : Axes, optional
        Existing axes.

    Returns
    -------
    fig : Figure
    ax : Axes
    """
    _check_matplotlib()

    if not hasattr(system, "energy"):
        raise ValueError("System must have an energy() method")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Compute energies
    energies = [system.energy(xi) for xi in x]

    # Check what keys are available
    if isinstance(energies[0], dict):
        KE = [e.get("kinetic", 0) for e in energies]
        PE = [e.get("potential", 0) for e in energies]
        total = [e.get("total", 0) for e in energies]

        ax.plot(t, KE, label="Kinetic", linewidth=1.5)
        ax.plot(t, PE, label="Potential", linewidth=1.5)
        ax.plot(t, total, "k--", label="Total", linewidth=2)
    else:
        ax.plot(t, energies, "k-", linewidth=2, label="Energy")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Energy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig, ax


# =============================================================================
# Constraint Plots
# =============================================================================


def plot_constraints(
    t: np.ndarray,
    values: Dict[str, np.ndarray],
    title: str = "Constraints",
    figsize: Tuple[float, float] = (10, 6),
) -> Tuple[Figure, Axes]:
    """
    Plot constraint values over time.

    Negative values indicate satisfaction.

    Parameters
    ----------
    t : np.ndarray
        Time array.
    values : dict
        Dictionary of constraint name -> array of values.
    title : str
        Title.
    figsize : tuple
        Figure size.

    Returns
    -------
    fig : Figure
    ax : Axes
    """
    _check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    for name, vals in values.items():
        t_plot = t[: len(vals)]
        ax.plot(t_plot, vals, label=name, linewidth=1.5)

    ax.axhline(0, color="k", linestyle="--", alpha=0.5, label="Constraint boundary")
    ax.fill_between(t, 0, ax.get_ylim()[1], alpha=0.1, color="red", label="Violated")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Constraint Value (negative = satisfied)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig, ax


# =============================================================================
# Comparison Plots
# =============================================================================


def plot_comparison(
    t_list: List[np.ndarray],
    x_list: List[np.ndarray],
    labels: List[str],
    state_idx: int = 0,
    ylabel: Optional[str] = None,
    title: str = "Comparison",
    figsize: Tuple[float, float] = (10, 6),
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """
    Compare multiple trajectories.

    Parameters
    ----------
    t_list : list of arrays
        Time arrays.
    x_list : list of arrays
        State trajectories.
    labels : list of str
        Labels for each trajectory.
    state_idx : int
        Which state to plot.
    ylabel : str, optional
        Y-axis label.
    title : str
        Title.
    figsize : tuple
        Figure size.
    ax : Axes, optional
        Existing axes.

    Returns
    -------
    fig : Figure
    ax : Axes
    """
    _check_matplotlib()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    for t, x, label in zip(t_list, x_list, labels):
        ax.plot(t, x[:, state_idx], linewidth=1.5, label=label)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel(ylabel or f"State[{state_idx}]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    return fig, ax
