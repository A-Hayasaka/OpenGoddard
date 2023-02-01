# -*- coding: utf-8 -*-
# Copyright 2017 Interstellar Technologies Inc. All Rights Reserved.

from __future__ import print_function
from re import U
import numpy as np
import matplotlib.pyplot as plt
from OpenGoddard.optimize import Problem, Guess, Condition, Dynamics


class Rocket:
    def __init__(self):
        self.tf = 3.32
        self.Thrust = 0.1405
        self.mdot = 0.0749
        self.Rinit = 1.0
        self.Vinit = 1.0


def dynamics(prob, obj, section):
    r = prob.states(0, section)
    theta = prob.states(1, section)
    u = prob.states(2, section)
    v = prob.states(3, section)
    m = prob.states(4, section)
    gamma = prob.controls(0, section)

    dx = Dynamics(prob, section)
    dx[0] = u
    dx[1] = v / r
    dx[2] = v**2 / r - 1 / r**2 + obj.Thrust / m * np.sin(gamma)
    dx[3] = -u * v / r + obj.Thrust / m * np.cos(gamma)
    dx[4] = -0.0749 * np.ones(len(m))
    return dx()


def equality(prob, obj):
    r = prob.states_all_section(0)
    theta = prob.states_all_section(1)
    u = prob.states_all_section(2)
    v = prob.states_all_section(3)
    m = prob.states_all_section(4)
    gamma = prob.controls_all_section(0)
    time_final = prob.time_final_all_section()

    result = Condition()

    # event condition
    result.equal(r[0], obj.Rinit)
    result.equal(theta[0], 0.0)
    result.equal(u[0], 0.0)
    result.equal(v[0], obj.Vinit)
    result.equal(m[0], 1.0)
    result.equal(u[-1], 0.0)
    result.equal(v[-1], np.sqrt(1.0 / r[-1]))
    result.equal(time_final[-1], obj.tf)

    return result()


def inequality(prob, obj):

    result = Condition()

    return result()


def cost(prob, obj):
    r = prob.states_all_section(0)
    # return -m[-1]
    # ==== Caution ====
    # cost function should be near 1.0
    return -r[-1]


def cost_derivative(prob, obj):
    jac = Condition(prob.number_of_variables)
    index_rf = prob.index_states(1, 0, -1)
    jac.change_value(index_rf, -1.0)
    return jac()


# ========================
# plt.close("all")
plt.ion()
# Program Starting Point
time_init = [0.0, 3.32]
n = [30]
num_states = [5]
num_controls = [1]
max_iteration = 30

flag_savefig = True
savefig_file = "08b_Rocket_Ascent_ConstThrust/"

# ------------------------
# set OpenGoddard class for algorithm determination
prob = Problem(time_init, n, num_states, num_controls, max_iteration)

# ------------------------
# create instance of operating object
obj = Rocket()


# ========================
# Initial parameter guess

# altitude profile
R_init = Guess.linear(prob.time_all_section, 1.0, 1.2)
# theta
Theta_init = Guess.linear(prob.time_all_section, 0.0, 1.0)

# velocity
U_init = Guess.linear(prob.time_all_section, 0.0, 0.0)
V_init = Guess.linear(prob.time_all_section, 1.0, 0.7)

# mass profile
M_init = Guess.cubic(prob.time_all_section, 1.0, 0.0, 0.5, 0.0)

# thrust angle profile
Gamma_init = Guess.linear(prob.time_all_section, 0.001, 0.001)

# plt.show()

# ========================
# Substitution initial value to parameter vector to be optimized
# non dimensional values (Divide by scale factor)
prob.set_states_all_section(0, R_init)
prob.set_states_all_section(1, Theta_init)
prob.set_states_all_section(2, U_init)
prob.set_states_all_section(3, V_init)
prob.set_states_all_section(4, M_init)
prob.set_controls_all_section(0, Gamma_init)

prob.set_states_bounds_all_section(0, 1.0, None)

prob.set_controls_bounds_all_section(0, -np.pi, np.pi)

# ========================
# Main Process
# Assign problem to SQP solver
prob.dynamics = [dynamics]
prob.knot_states_smooth = []
prob.cost = cost
# prob.cost_derivative = cost_derivative
prob.equality = equality
prob.inequality = inequality


def display_func():
    R = prob.states_all_section(0)
    print("max altitude: {0:.5f}".format(R[-1]))


prob.solve(obj, display_func, ftol=1e-8)

# ========================
# Post Process
# ------------------------
# Convert parameter vector to variable
r = prob.states_all_section(0)
theta = prob.states_all_section(1)
u = prob.states_all_section(2)
v = prob.states_all_section(3)
m = prob.states_all_section(4)
gamma = prob.controls_all_section(0)
time = prob.time_update()


# ------------------------
# Visualizetion
plt.figure()
plt.title("Altitude profile")
plt.plot(time, r, marker="o", label="Altitude")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time")
plt.ylabel("Altitude")
if flag_savefig:
    plt.savefig(savefig_file + "altitude" + ".png")

plt.figure()
plt.title("Velocity")
plt.plot(time, u, marker="o", label="u")
plt.plot(time, v, marker="o", label="v")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time")
plt.ylabel("Velocity")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "velocity" + ".png")

plt.figure()
plt.title("Mass")
plt.plot(time, m, marker="o", label="Mass")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time")
plt.ylabel("Mass")
if flag_savefig:
    plt.savefig(savefig_file + "mass" + ".png")

plt.figure()
plt.title("Control")
plt.plot(time, gamma, marker="o", label="gamma")
for line in prob.time_knots():
    plt.axvline(line, color="k", alpha=0.5)
plt.grid()
plt.xlabel("time")
plt.ylabel("Thrust angle")
plt.legend(loc="best")
if flag_savefig:
    plt.savefig(savefig_file + "control" + ".png")

plt.show()
