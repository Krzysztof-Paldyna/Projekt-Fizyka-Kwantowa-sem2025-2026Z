# -*- coding: utf-8 -*-
"""
Kwanty.py
Kornelia Leśniewska, Krzysztof Pałdyna
"""
#domek bibliotek
import numpy as np
import scipy.special as special
import scipy.constants as constants
import plotly.graph_objects as go

# Physical constants:
epsilon_0 = constants.epsilon_0
h_bar = constants.hbar
e_0 = constants.elementary_charge
m_p = constants.proton_mass
m_e = constants.electron_mass
m_0 = m_p * m_e / (m_p + m_e)

# Convert to spherical coordinates:
def spherical_convertion(x, y, z):
  r = np.sqrt(x**2 + y**2 + z**2)
  theta = np.arccos(z / (r + 1e-20))
  phi = np.arctan2(y, x)
  return r, theta, phi

def compute_orbital(n, l, m, grid_res=50):
  # creates a 3D coordinate space:
  limit = n * (n + 4)
  x = np.linspace(-limit, limit, grid_res)
  y = np.linspace(-limit, limit, grid_res)
  z = np.linspace(-limit, limit, grid_res)
  X, Y, Z = np.meshgrid(x, y, z)
  r, theta, phi = spherical_convertion(X, Y, Z)

  # angular component - spherical harmonics:
  angular = special.sph_harm_y(l, m, theta, phi)

  # radial component:
  a_0 = 4 * np.pi * epsilon_0 * h_bar**2 / ((e_0**2) * m_0)
  #rho = 2 * r / (n * a_0)
  rho = 2 * r / (n)

  radial_wave_function = rho**l * special.genlaguerre(n - l - 1, 2*l + 1)(rho) * np.exp(-rho/2)


  # normalisation:
  #normalisation = 2/(n**2) * np.sqrt(special.factorial(n - l - 1) / ((special.factorial(n+l))**3)) * (1 / np.sqrt(a_0 ** 3))
  normalisation = np.sqrt((2/(n))**3 * special.factorial(n - l - 1) / (2*n * special.factorial(n+l)))
  radial = normalisation * radial_wave_function

  # combined function:
  psi = radial * angular

  return X, Y, Z, psi

def plot_fig(X, Y, Z, prob):
  fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=prob.flatten(),
        isomin=prob.max() * 0.01,
        isomax=prob.max(),
        opacity=0.25,
        surface_count=25,
        colorscale='Plasma',
        caps=dict(x_show=False, y_show=False, z_show=False)))
  fig.update_layout(
        scene=dict(
            bgcolor="black"
        ),
        template="plotly_dark"
  )
  fig.show()

def plot_superposition(states):
  # compute multiple psi for the given sets of n l m
  PSI = []
  for state in states:
    X, Y, Z, psi = compute_orbital(state[0], state[1], state[2])
    PSI.append(psi)

  # probability density:
  superpos = 0
  for state in PSI:
    superpos = superpos + state
  prob = np.abs(superpos)**2 / len(PSI)

  # plot figure
  plot_fig(X, Y, Z, prob)

def plot_animated(n1, l1, m1, n2, l2, m2):
  X, Y, Z, psi_1 = compute_orbital(n1, l1, m1)
  X, Y, Z, psi_2 = compute_orbital(n2, l2, m2)
  limit = n1 * (n1 + 4)

  frames = []
  times = np.linspace(0, 60, 20)
  for t in times:
    # probability density:
    T_1 = np.exp(-1j * t / (n1*n1))
    #T_2 = np.exp(-1j * t / (n2*n2))
    prob = np.abs((psi_1*T_1 + psi_2)/np.sqrt(2))**2
    frames.append(go.Frame(
      data=go.Isosurface(
          x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
          value=prob.flatten(),
          isomin=prob.max() * 0.1,
          isomax=prob.max(),
          surface_count=10,
          opacity=0.25,
          colorscale='Plasma'
      )))
  fig = go.Figure(
      data=frames[0].data,
      layout=go.Layout(
          scene=dict(xaxis=dict(range=[-limit, limit]),
                      yaxis=dict(range=[-limit, limit]),
                      zaxis=dict(range=[-limit, limit])),
          updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None])])]
      ),
      frames=frames)
  fig.update_layout(
      scene=dict(
          bgcolor="black"
      ),
      template="plotly_dark")
  fig.show()

states = np.array([[4, 1, 0]])
plot_superposition(states)

states = np.array([[3, 2, 0], [4, 2, 1], [4, 3, -3]])
plot_superposition(states)

plot_animated(3, 1, 1, 3, 2, 0)