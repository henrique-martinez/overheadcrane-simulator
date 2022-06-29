import time
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from IPython.display import clear_output

class state:
    m = 0.5
    M = 1.0
    D = 10.0
    g = 9.8

    def __init__(self, x0, dx0, l0, dl0, theta0, dtheta0):
        self.x = x0
        self.dx = dx0
        self.l = l0
        self.dl = dl0
        self.theta = theta0
        self.dtheta = dtheta0
    
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

    def control(self, mode):
        if mode == "zero":
            return 0, 0
        if mode == "pendulo":
            f1 = 0
            f2 = (self.m*self.M)/(self.M + self.m*np.sin(self.theta)**2)*((self.D*self.dx*np.sin(self.theta)/self.M)-(self.l*self.dtheta**2 + self.g*np.cos(self.theta)))
            return f1, f2
        if mode == "sliding":
            f1 = 0
            f2 = 0
            return f1, f2
        return 0, 0

    def integrate(self, dt):
        f1, f2 = self.control("pendulo")
        self.model(f1, f2)
        x = self.x + self.dx * dt
        dx = self.dx + self.ddx * dt
        l = self.l + self.dl * dt
        dl = self.dl + self.ddl * dt
        theta = self.theta + self.dtheta * dt
        dtheta = self.dtheta + self.ddtheta * dt
        return state(x, dx, l, dl, theta, dtheta)

x0 = 0
dx0 = 0.1
l0 = 0.35
dl0 = 0
theta0 = 0.05
dtheta0 = 0

v0 = state(x0, dx0, l0, dl0, theta0, dtheta0)
v = [v0]

dt = 0.001
tf = 5
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
positions = [positions[i] for i in range(0, len(positions), 100)]
pos = positions[0]
line, = ax.plot([pos[0][0], pos[1][0]], [pos[0][1], pos[1][1]], color=cm.viridis(0), marker='o')
for i in range(0, len(positions)):
    pos = positions[i]
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

#######################################################
'''import pygame 

pygame.init()

surface = pygame.display.set_mode((800,400))

trolley = pygame.Rect(400+v0.x, 30, 60, 40)
charge = pygame.Rect(400+v0.x-1000*v0.l*np.sin(v0.theta), 1000*v0.l*np.cos(v0.theta), 10, 10)


for s in v:
    surface.fill((0,0,0))
    pygame.draw.rect(surface, (255,0,0), trolley)
    pygame.draw.rect(surface, (0,255,0), charge)
    trolley.center = (400+s.x, 30)
    charge.center = (400+s.x-1000*s.l*np.sin(s.theta), 1000*s.l*np.cos(s.theta))
    pygame.display.update()
    pygame.time.delay(int(dt*1000))
    # input()'''
