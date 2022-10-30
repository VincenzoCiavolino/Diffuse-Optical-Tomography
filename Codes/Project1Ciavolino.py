import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
                                                        #------PARAMETERS--------
v = 3*(10**10)
mu_a_0 = 0.10
mu_s = 10
r = 0.1
zp = 1
t = np.arange(10**-9,5*10**-8,10**-12)
D=1/(3*mu_s)
mu_eff=np.sqrt(3*0.1*10)
labels1 = []
labels2 = []
             
                                                        #------EQUATIONS--------
def PHI_hom(mu_a_0,D):
    return (v/((4*np.pi*v*D*t)**(3/2)))*np.exp(-((r**2)/(4*D*v*t))-(mu_a_0*t*v))

def Contr(mu_a_0,D):
    return (-(2*0.1)/(4*np.pi*D*zp))*np.exp(4*zp**2/(-4*D*v*t))

def PHI_hom_2D(r1,r2):
    return (1/(4*np.pi*D)*np.sqrt(r1**2+r2**2))*np.exp(-mu_eff*np.sqrt(r1**2+r2**2))

def delta_PHI_2D(r1,r2,xp,yp):
    return (-mu_a_0/((4*np.pi*D)**2)*np.sqrt(abs(r1-xp)**2+abs(r2-yp)**2))*np.exp(-mu_eff*np.sqrt(abs(r1-xp)**2+abs(r2-yp)**2))

def PHI_hom_2Dt(r1,r2,t):
    return (v/((4*np.pi*v*D*t)**(3/2)))*np.exp(-((r1**2+r2**2)/(4*D*v*t))-(mu_a_0*t*v))

def delta_PHI_2Dt(r1,r2,xp,yp,t):
    return (((v**2)*-2*mu_a_0)/(((4*np.pi*v*D)**(5/2))*np.sqrt(abs(r1-xp)**2+abs(r2-yp)**2)*t**(3/2)))*np.exp(-(mu_a_0*v*t)-(4*(abs(r1-xp)**2+abs(r2-yp)**2)/(4*D*v*t)))


                                                        #------TEMPORAL UNPERTURBED--------
for j in range(1,6):
     f1 = plt.figure(1)
     plt.plot(t,PHI_hom(mu_a_0,D/j))
     labels1.append(r"$\mu_a^0$=%.1f$mm^{-1}$, $\mu_s^{'}$=%d$mm^{-1}$"  %(mu_a_0,j/(3*D)))
plt.title("r=%.2fcm" %(r))
plt.legend(labels1)
plt.ylabel(r"$\phi^0$ $[cm^{-2}s^{-1}]$")
plt.xlabel("time(ns)")
plt.xticks(t[:10000],list(range(0,12)))

                                                        #------TEMPORAL CONTRAST--------
for j in range(1,6):
    zp += 1/j
    f2 = plt.figure(2)
    plt.plot(t,Contr(mu_a_0,D))

#f3 = plt.figure(3)
#plt.contourf(PHI_hom_2D(r1,r2))
#plt.colorbar()
                                                        #------2D STEADY STATE PERTURBED--------

L=10
N=100
f4 = plt.figure(4)
x=y = np.linspace(-L/2, L/2, N)
r1,r2= np.meshgrid(x,y)

ax1 = plt.axes([0.1, 0.18, 0.8, 0.65])
xax = plt.axes([0.1, 0.045, 0.8, 0.035])
yax = plt.axes([0.1, 0.002, 0.8, 0.035])

slidex = Slider(xax,'x',-5,5,valinit=0)
slidey = Slider(yax,'y',-5,5,valinit=0)
xp,yp = (slidex.val,slidey.val)


plt.axes(ax1)
im1 = plt.imshow(PHI_hom_2D(r1,r2)+delta_PHI_2D(r1,r2,xp,yp),extent=(-5,5,5,-5))
plt.title("2D Map of steady-state perturbed fluence \nas function of inhomogeneity position: $\Phi^{pert}(x,y)=\Phi^{0}(x,y)+\delta\Phi^{a}(x,y)$\n$\mu_a^{0} =0.1 cm^{-1}, \mu_s^{'}=1cm^{-1}$")
plt.ylabel("y (cm)")
plt.xlabel("x (cm)")
plt.colorbar()
scat1 = plt.scatter(xp,yp,s=1,c='r')

def update(a):
    im1.set_data(PHI_hom_2D(r1,r2)+delta_PHI_2D(r1,r2,slidex.val,slidey.val))
    f4.canvas.draw()
    scat1.set_offsets([slidex.val,slidey.val])
    f4.canvas.draw()

slidex.on_changed(update)
slidey.on_changed(update)

                                                        #------2D TEMPORAL UNPERTURBED--------
f5 = plt.figure(5)

ax2 = plt.axes([0.1, 0.2, 0.8, 0.65])
tax = plt.axes([0.1, 0.06, 0.8, 0.04])
slidet = Slider(tax,'t',10**-13,8*10**-10,valinit=10**-13)

plt.axes(ax2)
im2 = plt.imshow(PHI_hom_2Dt(r1,r2,slidet.val),extent=(-5,5,5,-5))
plt.title("2D Map of time resolved unperturbed fluence\n$\mu_a^{0} =0.1 cm^{-1}, \mu_s^{'}=1cm^{-1}$")
plt.ylabel("y (cm)")
plt.xlabel("x (cm)")
plt.colorbar()
scat2 = plt.scatter(0,0,s=1,c='r')

def update(a):
    im2.set_data(PHI_hom_2Dt(r1,r2,slidet.val))
    plt.scatter(0,0,s=1,c='r')
    f5.canvas.draw()

slidet.on_changed(update)

                                                        #------2D TEMPORAL PERTURBED--------
f6 = plt.figure(6)

ax3 = plt.axes([0.1, 0.18, 0.8, 0.65])
xax3 = plt.axes([0.1, 0.06, 0.8, 0.025])
yax3 = plt.axes([0.1, 0.031, 0.8, 0.025])
tax3 = plt.axes([0.1, 0.002, 0.8, 0.025])
slidex3 = Slider(xax3,'x',-5,5,valinit=0)
slidey3 = Slider(yax3,'y',-5,5,valinit=0)
xp3,yp3 = (slidex3.val,slidey3.val)
slidet3 = Slider(tax3,'t',10**-13,8*10**-10,valinit=10**-13)

plt.axes(ax3)
im3 = plt.imshow(PHI_hom_2Dt(r1,r2,slidet3.val)+delta_PHI_2Dt(r1,r2,slidex3.val,slidey3.val,slidet3.val),extent=(-5,5,5,-5))
plt.title("2D Map of time resolved perturbed fluence\n$\Phi^{pert}(x,y,t)=\Phi^{0}(x,y,t)+\delta\Phi^{a}(x,y,t)$\n$\mu_a^{0} =0.1 cm^{-1}, \mu_s^{'}=1cm^{-1}$")
plt.ylabel("y (cm)")
plt.xlabel("x (cm)")
plt.colorbar()
scat3 = plt.scatter(xp3,yp3,s=1,c='r')

def update(a):
    im3.set_data(PHI_hom_2Dt(r1,r2,slidet3.val)+delta_PHI_2Dt(r1,r2,slidex3.val,slidey3.val,slidet3.val))
    f6.canvas.draw()
    scat3.set_offsets([slidex3.val,slidey3.val])
    f6.canvas.draw()

slidex3.on_changed(update)
slidey3.on_changed(update)
slidet3.on_changed(update)

plt.show()
