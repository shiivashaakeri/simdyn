"""
Animation utilities for simdyn.

This module provides functions for creating animations of dynamical systems:
- Pendulum swing animation
- Cart-pole animation
- Unicycle/vehicle animation
- Rocket trajectory animation

Requires matplotlib with animation support.
"""

from typing import Any, Optional, Tuple

import numpy as np

# Check for matplotlib availability
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.patches import Circle, Polygon, Rectangle

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    Figure = Any
    Axes = Any
    FuncAnimation = Any


def _check_matplotlib():
    """Raise error if matplotlib not available."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for animation. Install with: pip install matplotlib")


# =============================================================================
# Pendulum Animation
# =============================================================================


def animate_pendulum(
    t: np.ndarray,
    x: np.ndarray,
    length: float = 1.0,
    title: str = "Pendulum Animation",
    figsize: Tuple[float, float] = (6, 6),
    interval: int = 20,
    trail_length: int = 50,
    save_path: Optional[str] = None,
) -> FuncAnimation:
    """
    Create animation of a pendulum.

    Parameters
    ----------
    t : np.ndarray, shape (N,)
        Time array.
    x : np.ndarray, shape (N, 2)
        State trajectory [θ, ω].
    length : float
        Pendulum length.
    title : str
        Animation title.
    figsize : tuple
        Figure size.
    interval : int
        Milliseconds between frames.
    trail_length : int
        Number of past positions to show as trail.
    save_path : str, optional
        Path to save animation (e.g., 'pendulum.gif').

    Returns
    -------
    anim : FuncAnimation
        Matplotlib animation object.

    Example
    -------
    >>> t, x, u = pendulum.simulate(x0, controller, (0, 10), dt=0.02)
    >>> anim = animate_pendulum(t, x, length=pendulum.params.l)
    >>> plt.show()
    """
    _check_matplotlib()

    # Compute bob positions
    # θ = 0 is down, θ = π is up
    theta = x[:, 0]
    bob_x = length * np.sin(theta)
    bob_y = -length * np.cos(theta)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-1.5 * length, 1.5 * length)
    ax.set_ylim(-1.5 * length, 1.5 * length)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)

    # Draw pivot
    pivot = Circle((0, 0), 0.02 * length, color="black", zorder=5)
    ax.add_patch(pivot)

    # Initialize plot elements
    (rod,) = ax.plot([], [], "k-", linewidth=2, zorder=3)
    bob = Circle((0, 0), 0.08 * length, color="blue", zorder=4)
    ax.add_patch(bob)
    (trail,) = ax.plot([], [], "b-", alpha=0.3, linewidth=1)
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, verticalalignment="top")

    def init():
        rod.set_data([], [])
        bob.center = (0, -length)
        trail.set_data([], [])
        time_text.set_text("")
        return rod, bob, trail, time_text

    def animate(i):
        # Update rod
        rod.set_data([0, bob_x[i]], [0, bob_y[i]])

        # Update bob
        bob.center = (bob_x[i], bob_y[i])

        # Update trail
        start = max(0, i - trail_length)
        trail.set_data(bob_x[start : i + 1], bob_y[start : i + 1])

        # Update time
        time_text.set_text(f"t = {t[i]:.2f}s")

        return rod, bob, trail, time_text

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(t), interval=interval, blit=True)

    if save_path:
        anim.save(save_path, writer="pillow", fps=1000 // interval)
        print(f"Saved animation to {save_path}")

    return anim


# =============================================================================
# Cart-Pole Animation
# =============================================================================


def animate_cartpole(
    t: np.ndarray,
    x: np.ndarray,
    cart_width: float = 0.4,
    cart_height: float = 0.2,
    pole_length: float = 1.0,
    track_length: float = 4.8,
    title: str = "Cart-Pole Animation",
    figsize: Tuple[float, float] = (10, 6),
    interval: int = 20,
    save_path: Optional[str] = None,
) -> FuncAnimation:
    """
    Create animation of a cart-pole system.

    Parameters
    ----------
    t : np.ndarray, shape (N,)
        Time array.
    x : np.ndarray, shape (N, 4)
        State trajectory [cart_x, cart_v, θ, ω].
    cart_width : float
        Cart width.
    cart_height : float
        Cart height.
    pole_length : float
        Full pole length.
    track_length : float
        Total track length.
    title : str
        Animation title.
    figsize : tuple
        Figure size.
    interval : int
        Milliseconds between frames.
    save_path : str, optional
        Path to save animation.

    Returns
    -------
    anim : FuncAnimation
    """
    _check_matplotlib()

    # Extract states
    cart_x = x[:, 0]
    theta = x[:, 2]  # θ = 0 is up

    # Pole tip positions (θ = 0 is up)
    pole_tip_x = cart_x + pole_length * np.sin(theta)
    pole_tip_y = cart_height / 2 + pole_length * np.cos(theta)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    margin = 0.5
    ax.set_xlim(-track_length / 2 - margin, track_length / 2 + margin)
    ax.set_ylim(-0.5, pole_length + 0.5)
    ax.set_aspect("equal")
    ax.set_title(title)

    # Draw track
    ax.axhline(0, color="brown", linewidth=3, zorder=1)
    ax.axvline(-track_length / 2, color="red", linestyle="--", alpha=0.5)
    ax.axvline(track_length / 2, color="red", linestyle="--", alpha=0.5)

    # Initialize cart
    cart = Rectangle((0, 0), cart_width, cart_height, facecolor="gray", edgecolor="black", zorder=3)
    ax.add_patch(cart)

    # Initialize wheels
    wheel_radius = cart_height / 4
    wheel1 = Circle((0, 0), wheel_radius, color="black", zorder=4)
    wheel2 = Circle((0, 0), wheel_radius, color="black", zorder=4)
    ax.add_patch(wheel1)
    ax.add_patch(wheel2)

    # Initialize pole
    (pole,) = ax.plot([], [], "b-", linewidth=4, zorder=5)

    # Pole tip marker
    tip = Circle((0, 0), 0.05, color="red", zorder=6)
    ax.add_patch(tip)

    # Time text
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, verticalalignment="top", fontsize=12)

    def init():
        cart.set_xy((-cart_width / 2, 0))
        wheel1.center = (-cart_width / 4, 0)
        wheel2.center = (cart_width / 4, 0)
        pole.set_data([], [])
        tip.center = (0, pole_length + cart_height / 2)
        time_text.set_text("")
        return cart, wheel1, wheel2, pole, tip, time_text

    def animate(i):
        cx = cart_x[i]

        # Update cart
        cart.set_xy((cx - cart_width / 2, 0))

        # Update wheels
        wheel1.center = (cx - cart_width / 4, 0)
        wheel2.center = (cx + cart_width / 4, 0)

        # Update pole
        pole_base_y = cart_height / 2
        pole.set_data([cx, pole_tip_x[i]], [pole_base_y, pole_tip_y[i]])

        # Update tip
        tip.center = (pole_tip_x[i], pole_tip_y[i])

        # Update time
        time_text.set_text(f"t = {t[i]:.2f}s\nθ = {np.rad2deg(theta[i]):.1f}°")

        return cart, wheel1, wheel2, pole, tip, time_text

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(t), interval=interval, blit=True)

    if save_path:
        anim.save(save_path, writer="pillow", fps=1000 // interval)
        print(f"Saved animation to {save_path}")

    return anim


# =============================================================================
# Unicycle Animation
# =============================================================================


def animate_unicycle(
    t: np.ndarray,
    x: np.ndarray,
    robot_radius: float = 0.2,
    title: str = "Unicycle Animation",
    figsize: Tuple[float, float] = (8, 8),
    interval: int = 50,
    trail_length: int = 100,
    save_path: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> FuncAnimation:
    """
    Create animation of a unicycle robot.

    Parameters
    ----------
    t : np.ndarray, shape (N,)
        Time array.
    x : np.ndarray, shape (N, 3)
        State trajectory [x, y, θ].
    robot_radius : float
        Robot body radius.
    title : str
        Animation title.
    figsize : tuple
        Figure size.
    interval : int
        Milliseconds between frames.
    trail_length : int
        Number of past positions to show.
    save_path : str, optional
        Path to save animation.
    xlim, ylim : tuple, optional
        Axis limits.

    Returns
    -------
    anim : FuncAnimation
    """
    _check_matplotlib()

    pos_x = x[:, 0]
    pos_y = x[:, 1]
    theta = x[:, 2]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Set limits
    if xlim is None:
        margin = 1.0
        xlim = (pos_x.min() - margin, pos_x.max() + margin)
    if ylim is None:
        margin = 1.0
        ylim = (pos_y.min() - margin, pos_y.max() + margin)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(title)

    # Initialize robot body
    body = Circle((0, 0), robot_radius, facecolor="blue", edgecolor="black", alpha=0.7, zorder=3)
    ax.add_patch(body)

    # Heading indicator (line from center)
    (heading,) = ax.plot([], [], "r-", linewidth=2, zorder=4)

    # Trail
    (trail,) = ax.plot([], [], "b-", alpha=0.3, linewidth=1, zorder=1)

    # Time text
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, verticalalignment="top")

    def init():
        body.center = (pos_x[0], pos_y[0])
        heading.set_data([], [])
        trail.set_data([], [])
        time_text.set_text("")
        return body, heading, trail, time_text

    def animate(i):
        # Update body position
        body.center = (pos_x[i], pos_y[i])

        # Update heading indicator
        dx = robot_radius * 1.5 * np.cos(theta[i])
        dy = robot_radius * 1.5 * np.sin(theta[i])
        heading.set_data([pos_x[i], pos_x[i] + dx], [pos_y[i], pos_y[i] + dy])

        # Update trail
        start = max(0, i - trail_length)
        trail.set_data(pos_x[start : i + 1], pos_y[start : i + 1])

        # Update time
        time_text.set_text(f"t = {t[i]:.2f}s\nθ = {np.rad2deg(theta[i]):.1f}°")

        return body, heading, trail, time_text

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(t), interval=interval, blit=True)

    if save_path:
        anim.save(save_path, writer="pillow", fps=1000 // interval)
        print(f"Saved animation to {save_path}")

    return anim


# =============================================================================
# Rocket Animation (2D side view)
# =============================================================================


def animate_rocket_2d(
    t: np.ndarray,
    x: np.ndarray,
    u: np.ndarray,
    rocket_height: float = 0.5,
    rocket_width: float = 0.15,
    title: str = "Rocket Landing Animation",
    figsize: Tuple[float, float] = (10, 10),
    interval: int = 20,
    trail_length: int = 100,
    save_path: Optional[str] = None,
) -> FuncAnimation:
    """
    Create 2D side-view animation of rocket landing.

    Shows altitude (x-axis of state) vs horizontal position.

    Parameters
    ----------
    t : np.ndarray, shape (N,)
        Time array.
    x : np.ndarray, shape (N, 7) or (N, 14)
        Rocket state trajectory.
    u : np.ndarray, shape (N-1, 3)
        Thrust history.
    rocket_height : float
        Visual rocket height.
    rocket_width : float
        Visual rocket width.
    title : str
        Animation title.
    figsize : tuple
        Figure size.
    interval : int
        Milliseconds between frames.
    trail_length : int
        Trail length.
    save_path : str, optional
        Path to save.

    Returns
    -------
    anim : FuncAnimation
    """
    _check_matplotlib()

    # Extract positions (indices 1=altitude, 2=horizontal)
    alt = x[:, 1]  # Up (x in UEN)
    horiz = x[:, 2]  # East (y in UEN)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    margin = 2.0
    ax.set_xlim(horiz.min() - margin, horiz.max() + margin)
    ax.set_ylim(-0.5, alt.max() + margin)
    ax.set_aspect("equal")
    ax.set_xlabel("Horizontal Position [m]")
    ax.set_ylabel("Altitude [m]")
    ax.set_title(title)

    # Ground
    ax.axhline(0, color="brown", linewidth=3, zorder=1)
    ax.fill_between(ax.get_xlim(), -0.5, 0, color="saddlebrown", alpha=0.3)

    # Target
    ax.plot(0, 0, "r*", markersize=20, zorder=2, label="Target")

    # Initialize rocket body (simplified as rectangle)
    rocket = Rectangle((0, 0), rocket_width, rocket_height, facecolor="gray", edgecolor="black", zorder=5)
    ax.add_patch(rocket)

    # Thrust flame (triangle)
    flame = Polygon([(0, 0), (0, 0), (0, 0)], closed=True, facecolor="orange", edgecolor="red", zorder=4)
    ax.add_patch(flame)

    # Trail
    (trail,) = ax.plot([], [], "b-", alpha=0.3, linewidth=1, zorder=1)

    # Time and info text
    time_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, verticalalignment="top", fontsize=10, family="monospace"
    )

    def init():
        rocket.set_xy((horiz[0] - rocket_width / 2, alt[0]))
        flame.set_xy([(0, 0), (0, 0), (0, 0)])
        trail.set_data([], [])
        time_text.set_text("")
        return rocket, flame, trail, time_text

    def animate(i):
        # Update rocket position
        rocket.set_xy((horiz[i] - rocket_width / 2, alt[i]))

        # Update thrust flame
        if i < len(u):
            T_mag = np.linalg.norm(u[i])
            flame_len = T_mag * 0.05  # Scale thrust to flame length

            # Flame points (below rocket)
            cx = horiz[i]
            base_y = alt[i]
            flame.set_xy([(cx - rocket_width / 3, base_y), (cx + rocket_width / 3, base_y), (cx, base_y - flame_len)])
        else:
            flame.set_xy([(0, 0), (0, 0), (0, 0)])

        # Update trail
        start = max(0, i - trail_length)
        trail.set_data(horiz[start : i + 1], alt[start : i + 1])

        # Update text
        mass = x[i, 0]
        vel_down = -x[i, 4]  # Descent rate
        info = f"t = {t[i]:.2f}s\nAlt = {alt[i]:.2f}m\nDescent = {vel_down:.2f}m/s\nMass = {mass:.3f}kg"
        time_text.set_text(info)

        return rocket, flame, trail, time_text

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(t), interval=interval, blit=True)

    if save_path:
        anim.save(save_path, writer="pillow", fps=1000 // interval)
        print(f"Saved animation to {save_path}")

    return anim


# =============================================================================
# Double Integrator Animation (2D)
# =============================================================================


def animate_point_mass_2d(
    t: np.ndarray,
    x: np.ndarray,
    target: Optional[np.ndarray] = None,
    title: str = "Point Mass Animation",
    figsize: Tuple[float, float] = (8, 8),
    interval: int = 50,
    trail_length: int = 100,
    save_path: Optional[str] = None,
) -> FuncAnimation:
    """
    Create animation of a 2D point mass (double integrator).

    Parameters
    ----------
    t : np.ndarray, shape (N,)
        Time array.
    x : np.ndarray, shape (N, 4)
        State trajectory [px, py, vx, vy].
    target : np.ndarray, optional
        Target position [x, y].
    title : str
        Animation title.
    figsize : tuple
        Figure size.
    interval : int
        Milliseconds between frames.
    trail_length : int
        Trail length.
    save_path : str, optional
        Path to save.

    Returns
    -------
    anim : FuncAnimation
    """
    _check_matplotlib()

    pos_x = x[:, 0]
    pos_y = x[:, 1]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    margin = 1.0
    ax.set_xlim(pos_x.min() - margin, pos_x.max() + margin)
    ax.set_ylim(pos_y.min() - margin, pos_y.max() + margin)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(title)

    # Target
    if target is not None:
        ax.plot(target[0], target[1], "r*", markersize=20, label="Target")

    # Point mass
    point = Circle((0, 0), 0.1, facecolor="blue", edgecolor="black", zorder=3)
    ax.add_patch(point)

    # Velocity vector
    (vel_arrow,) = ax.plot([], [], "g-", linewidth=2, zorder=2)

    # Trail
    (trail,) = ax.plot([], [], "b-", alpha=0.3, linewidth=1, zorder=1)

    # Time text
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, verticalalignment="top")

    def init():
        point.center = (pos_x[0], pos_y[0])
        vel_arrow.set_data([], [])
        trail.set_data([], [])
        time_text.set_text("")
        return point, vel_arrow, trail, time_text

    def animate(i):
        # Update position
        point.center = (pos_x[i], pos_y[i])

        # Update velocity arrow
        vx, vy = x[i, 2], x[i, 3]
        scale = 0.3  # velocity arrow scale
        vel_arrow.set_data([pos_x[i], pos_x[i] + scale * vx], [pos_y[i], pos_y[i] + scale * vy])

        # Update trail
        start = max(0, i - trail_length)
        trail.set_data(pos_x[start : i + 1], pos_y[start : i + 1])

        # Update time
        time_text.set_text(f"t = {t[i]:.2f}s")

        return point, vel_arrow, trail, time_text

    anim = FuncAnimation(fig, animate, init_func=init, frames=len(t), interval=interval, blit=True)

    if save_path:
        anim.save(save_path, writer="pillow", fps=1000 // interval)
        print(f"Saved animation to {save_path}")

    return anim


# =============================================================================
# Helper Functions
# =============================================================================


def save_animation(anim: FuncAnimation, path: str, fps: int = 30, dpi: int = 100):
    """
    Save animation to file.

    Parameters
    ----------
    anim : FuncAnimation
        Animation object.
    path : str
        Output path (.gif, .mp4, etc.)
    fps : int
        Frames per second.
    dpi : int
        Resolution.
    """
    _check_matplotlib()

    if path.endswith(".gif"):
        writer = "pillow"
    elif path.endswith(".mp4"):
        writer = "ffmpeg"
    else:
        writer = "pillow"

    anim.save(path, writer=writer, fps=fps, dpi=dpi)
    print(f"Saved animation to {path}")
