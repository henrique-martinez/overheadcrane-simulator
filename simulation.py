import time
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from IPython.display import clear_output

class state:
    m = 1
    M = 5
    D = 0.2
    g = 9.8

    def __init__(self, x0, dx0, l0, dl0, theta0, dtheta0):
        self.x = x0
        self.dx = dx0
        self.l = l0
        self.dl = dl0
        self.theta = theta0
        self.dtheta = dtheta0

        self.ddx = 0
        self.ddl = 0
        self.ddtheta = 0

        self.f1 = 0
        self.f2 = -self.m*self.g
        self.df1 = 0
        self.df2 = 0

    def sigmoid(self, lamb, t):
        return (1-np.exp(-lamb*t))/(1+np.exp(-lamb*t))
    
    def T(self):
        return 0.5*self.M*self.dx**2 + 0.5*self.m*(self.l**2*self.dtheta**2 + self.dx**2 - 2*self.l*self.dx*self.dtheta*np.cos(self.theta)+self.dl**2-2*self.dl*self.dx*np.sin(self.theta))
    
    def V(self):
        return -self.m*self.g*self.l*np.cos(self.theta)
    
    def positions(self):
        return (self.x, 0), (self.x+self.l*np.sin(self.theta), -self.l*np.cos(self.theta))
    
    def model(self, f1, f2):
        self.ddx = (f1 - self.D*self.dx + f2*np.sin(self.theta))/self.M
        self.ddl = self.g*np.cos(self.theta) + self.l*self.dtheta**2 + (f1 - self.D*self.dx)*np.sin(self.theta)/self.M + f2*(self.M + self.m*np.sin(self.theta)**2)/(self.M*self.m)
        self.ddtheta = np.cos(self.theta)*(f1 - self.D*self.dx + f2*np.sin(self.theta))/(self.M*self.l) - 2*self.dl*self.dtheta/self.l - self.g*np.sin(self.theta)/self.l
        # print(self.ddx, self.ddl, self.ddtheta)

    def control(self, mode, dt):
        if mode == "zero":
            return 0, 0
        if mode == "pendulo":
            f1 = 0
            f2 = (self.m*self.M)/(self.M + self.m*np.sin(self.theta)**2)*((self.D*self.dx*np.sin(self.theta)/self.M)-(self.l*self.dtheta**2 + self.g*np.cos(self.theta)))
            return f1, f2
        if mode == "sliding":
            lamb = 2
            x_d, l_d = 0.5, 0.5
            epsilon1, k1 = 5, 5
            epsilon2, k2 = 3, 28
            a_x, b_x, a_t, b_t = 20*10, 8, 7*1, 140*10
            s1 = self.ddx + a_x * self.dx + b_x * (self.x - x_d) + a_t * self.dtheta + b_t * self.theta
            a_l, b_l = 15*2, 36*0.1
            s2 = self.ddl + a_l * self.dl + b_l * (self.l - l_d)

            df1 = -np.sin(self.theta)*self.df2 + (self.D/self.M) * (self.f1 - self.D * self.dx + self.f2 * np.sin(self.theta)) - self.dtheta*np.cos(self.theta)*self.f2 - self.M*(a_x*self.ddx + b_x*self.dx + a_t*self.ddtheta + b_t*self.dtheta + epsilon1*self.sigmoid(lamb, s1)+k1*s1)
            df2 = -self.df1*self.m*np.sin(self.theta)/(self.M + self.m*np.sin(self.theta)**2) - ((self.M*self.m)/(self.M + self.m*np.sin(self.theta)**2))*(((self.f2*self.dtheta*np.sin(2*self.theta))/self.M)+((self.M*(self.f1-self.D*self.dx)*self.dtheta*np.cos(self.theta)-self.D*(self.f1 - self.D * self.dx + self.f2 * np.sin(self.theta))*np.sin(self.theta))/self.M**2)-self.g*self.dtheta*np.sin(self.theta)+self.dl*self.dtheta**2+2*self.l*self.dtheta*self.ddtheta+a_l*self.ddl+b_l*self.dl+epsilon2*self.sigmoid(lamb, s2)+k2*s2)

            self.df1 = df1
            self.df2 = df2

            self.f1 = self.f1 + self.df1 * dt
            self.f2 = self.f2 + self.df2 * dt

            return self.f1, self.f2
        return 0, 0

    def integrate(self, dt):
        f1, f2 = self.control("sliding", dt)
        self.model(f1, f2)
        x = self.x + self.dx * dt
        dx = self.dx + self.ddx * dt
        l = self.l + self.dl * dt
        dl = self.dl + self.ddl * dt
        theta = self.theta + self.dtheta * dt
        dtheta = self.dtheta + self.ddtheta * dt
        return state(x, dx, l, dl, theta, dtheta)

x0 = 0
dx0 = -0.2
l0 = 0.35
dl0 = 0
theta0 = 0.5
dtheta0 = 0

v0 = state(x0, dx0, l0, dl0, theta0, dtheta0)
v = [v0]

dt = 0.001
tf = 20
t = np.linspace(0, tf, int(tf/dt))
for _ in range(len(t)):
    v.append(v[-1].integrate(dt))

t = [dt*i for i in range(len(v))]
x = [i.x for i in v]
dx = [i.dx for i in v]
theta = [i.theta for i in v]
dtheta = [i.dtheta for i in v]
l = [i.l for i in v]
dl = [i.dl for i in v]
T = [i.T() for i in v]
V = [i.V() for i in v]
positions = [i.positions() for i in v]


# printing positions
'''
fig, ax = plt.subplots()
positions = [positions[i] for i in range(0, len(positions), 200)]
for i in range(0, len(positions)):
    pos = positions[i]
    ax.plot([pos[0][0], pos[1][0]], [pos[0][1], pos[1][1]], color=cm.viridis(i/len(positions)), marker='o')
#ax.scatter([pos[0][0] for pos in positions], [pos[0][1] for pos in positions], c=[i/len(positions) for i in range(len(positions))])
#ax.scatter([pos[1][0] for pos in positions], [pos[1][1] for pos in positions], c=[i/len(positions) for i in range(len(positions))])
ax.axis('equal')
ax.set_ylabel("x [m]")
ax.set_xlabel("y [m]")
ax.legend()
plt.show()
'''

# animate positions
plt.ion()
fig, ax = plt.subplots()
positions = [positions[i] for i in range(0, len(positions), 200)]
pos = positions[0]
baseline, = ax.plot([-0.6, 0.6], [0, 0], color='black', marker='.')
line, = ax.plot([pos[0][0], pos[1][0]], [pos[0][1], pos[1][1]], color=cm.viridis(0), marker='o', linestyle='dashed')
for i in range(0, len(positions)):
    pos = positions[i]
    line, = ax.plot([pos[0][0], pos[1][0]], [pos[0][1], pos[1][1]], color=cm.viridis(0), marker='o', linestyle='dashed')
    line.set_xdata([pos[0][0], pos[1][0]])
    line.set_ydata([pos[0][1], pos[1][1]])
    line.set_color(cm.viridis(i/len(positions)))
    #ax.plot([pos[0][0], pos[1][0]], [pos[0][1], pos[1][1]], color=cm.viridis(i/len(positions)), marker='o')
    ax.axis('equal')
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    plt.xlim([-0.6, 0.6])
    plt.ylim([-1, 0.2])
    fig.canvas.draw()
    fig.canvas.flush_events()
plt.ioff()

# plots
fig, axs = plt.subplots(3,2)
fig.suptitle('Simulação')
axs[0,0].plot(t, x)
axs[0,0].set_ylabel("$x$ [m]")
axs[0,0].set_xlabel("tempo [s]")
axs[0,1].plot(t, dx)
axs[0,1].set_ylabel("$dx/dt$ [m/s]")
axs[0,1].set_xlabel("tempo [s]")
axs[1,0].plot(t, theta)
axs[1,0].set_ylabel("$\\theta$ [rad]")
axs[1,0].set_xlabel("tempo [s]")
axs[1,1].plot(t, dtheta)
axs[1,1].set_ylabel("$d\\theta/dt$ [rad/s]")
axs[1,1].set_xlabel("tempo [s]")
axs[2,0].plot(t, l)
axs[2,0].set_ylabel("$l$ [m]")
axs[2,0].set_xlabel("tempo [s]")
axs[2,1].plot(t, dl)
axs[2,1].set_ylabel("$dl/dt$ [m/s]")
axs[2,1].set_xlabel("tempo [s]")

fig, ax = plt.subplots()
ax.plot(t, T, label="T")
ax.plot(t, V, label="V")
ax.plot(t, [T[i]+V[i] for i in range(len(t))], label="T+V")
ax.set_ylabel("energia [J]")
ax.set_xlabel("tempo [s]")
ax.legend()
plt.show()