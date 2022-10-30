import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RadioButtons, Slider
from numpy import inf
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.colors as color
 

                                             #------------------ PARAMETERS -----------------
v = 3.0*(10**10)
mu_a_0 = 0.10
mu_s = 10
dt = np.arange(10**-10,1.1*10**-8,10**-9)
D=1/(3*mu_s)
M = []
w = []
det=[]
vox=[]
N = 8   
                                             #------------- COORDINATES CREATION -------------
for k in range(N):
    for j in range(N):
        for i in range(N):
            if k == 0:
                det.append((j,i))
            vox.append((j,i,k))

                                             #-------------- SENSITIVITY MATRIX --------------

def Contr(x_sd,y_sd,z_sd,x_p,y_p,z_p,dt):
    ro = np.sqrt(abs(x_sd-x_p)**2+abs(y_sd-y_p)**2+abs(z_sd-z_p)**2)
    return float((1)/(2*np.pi*D*ro))*np.exp(ro**2/(-D*v*dt)) 

for x_sd,y_sd in det:
    for t in dt:
        for x_p,y_p,z_p in vox:
            w.append(Contr(x_sd,y_sd,0,x_p,y_p,z_p,t))    # 1D sensitivity matrix
W = np.array(w).reshape(len(dt)*len(det),len(vox))        # 2D sensitivity matrix
W[W==inf]=5
#W=W/5
def ExtractSensitivityProfile(t,sd,pl):
    row = sd*11+t
    SensitivityProfile = (W[row,pl*64:(pl+1)*64].reshape(8,8))
    return SensitivityProfile

                                             #--------------  MEASURES  --------------

A = [0]*512
A[0]=0.1
M = W.dot(np.array(A)).reshape(len(det),len(dt))
P = 10**6

def ExtractTPSF(sd,xp,yp,zp):
    A2 = [0]*512
    rp = xp+yp*8+zp*64
    A2[rp]=0.1
    M2 = W.dot(np.array(A2)).reshape(len(det),len(dt)) 

    M3 = (P*(1+M2)-P)/P

    M_noise = (np.random.poisson(P*(1+M))-np.random.poisson(P))/np.random.poisson(P)

    TPSF = M3[sd,:]
    return TPSF

def ExtractTomography(tslot,xp,yp,zp):
    A1 = [0]*512
    rp = xp+yp*8+zp*64
    A1[rp]=0.1
    M1 = W.dot(np.array(A1)).reshape(len(det),len(dt)) 
    Tomography = (M1[:,tslot].reshape(8,8))#/M.max()
    return Tomography

                                             #--------------  PLOTS  --------------

print(ExtractTPSF(0,0,4,2))

x,y,z = np.indices((N,N,N))
Vox1 = (x > N) & (y > N) & (z > N)
Vox1[0,0,0]=True
Vox2 = (x > N) & (y > N) & (z > N)
Vox2[:,:,0]=True
Vox3 = (x > N) & (y > N) & (z > N)
Vox3[0,0,0]=True

                                              
f4 = plt.figure(4)   
ax6 = f4.add_subplot(1, 2, 1,projection='3d')  
sc = ax6.scatter(0.5,0.5, color='g')
ax6.voxels(Vox3, facecolors='r', edgecolor='gray')                                 # 3D plot 
ax6.set_xticks(range(8))
ax6.set_yticks(range(8))
ax6.set_zticks(range(8))          
plt.title('Voxel Space')
xpax1 = plt.axes([0.03, 0.06, 0.45, 0.02])
slidexp1 = Slider(xpax1,'$x_p$',0,7,valinit=0,valstep=1)
ypax1 = plt.axes([0.03, 0.035, 0.45, 0.02])
slideyp1 = Slider(ypax1,'$y_p$',0,7,valinit=0,valstep=1)
zpax1 = plt.axes([0.03, 0.01, 0.45, 0.02])
slidezp1 = Slider(zpax1,'$z_p$',0,7,valinit=0,valstep=1)

ax5 = f4.add_subplot(1, 2, 2)
ax5.bar(range(11),ExtractTPSF(0,0,0,0))                                                  # TPSF
plt.title('Contrast TPSF')
plt.xlabel('Time (ns)')
sd1ax = plt.axes([0.56, 0.005, 0.4, 0.020])
slidesd1 = Slider(sd1ax,'SD Pair',0,63,valinit=0,valstep=1)


f1 = plt.figure(1)
ax1 = f1.add_subplot(1, 2, 1,projection='3d')
ax1.voxels(Vox1, facecolors='r', edgecolor='gray')                                 # 3D plot 
ax1.set_xticks(range(8))
ax1.set_yticks(range(8))
ax1.set_zticks(range(8))
plt.title('Voxel Space')
for e in range(8):
    for u in range(8):
        ax1.scatter(u+0.5,e+0.5,0, color='g')
xpax = plt.axes([0.03, 0.06, 0.45, 0.02])
slidexp = Slider(xpax,'$x_p$',0,7,valinit=0,valstep=1)
ypax = plt.axes([0.03, 0.035, 0.45, 0.02])
slideyp = Slider(ypax,'$y_p$',0,7,valinit=0,valstep=1)
zpax = plt.axes([0.03, 0.01, 0.45, 0.02])
slidezp = Slider(zpax,'$z_p$',0,7,valinit=0,valstep=1)

ax2 = f1.add_subplot(1, 2, 2)
im1 = ax2.imshow(ExtractTomography(7,0,0,0),norm = color.Normalize(vmin=0.,vmax=1.),vmax=0.65,vmin=0)                         # Tomography slice
#plt.title('Tomograghy slice (X,Y) plane')
tax1 = plt.axes([0.56, 0.025, 0.4, 0.020])
slidet1 = Slider(tax1,'GATE',0,10,valinit=0,valstep=1)


                                            
f2 = plt.figure(2)
ax3 = f2.add_subplot(1, 2, 1,projection='3d')
ax3.voxels(Vox2, facecolors='r', edgecolor='gray')                                 # 3D plot 
ax3.set_xticks(range(8))
ax3.set_yticks(range(8))
ax3.set_zticks(range(8))
plt.title('Voxel Space')
ax3.scatter(0.5,0.5,0, color='g')
plax = plt.axes([0.03, 0.03, 0.4, 0.02])
slidepl = Slider(plax,'$z_p$',0,7,valinit=0,valstep=1)

ax4 = f2.add_subplot(1, 2, 2)
im = plt.imshow(W[0,:64].reshape(8,8),vmax=1,vmin=0)                                #Sens. Matr. plot
plt.colorbar()
plt.title('Sensitivity Profile')
tax = plt.axes([0.56, 0.03, 0.4, 0.020])
slidet = Slider(tax,'GATE',0,10,valinit=0,valstep=1)
sdax = plt.axes([0.56, 0.005, 0.4, 0.020])
slidesd = Slider(sdax,'SD Pair',0,63,valinit=0,valstep=1)

                                             #--------------  WIDGETS  --------------

def update(a):
    im.set_data(ExtractSensitivityProfile(int(slidet.val),int(slidesd.val),int(slidepl.val)))
slidet.on_changed(update)
slidesd.on_changed(update)

def update4(a):
    Vox3[:,:,:]=False
    Vox3[int(slidexp1.val),int(slideyp1.val),int(slidezp1.val)]=True
    ax6.cla()
    ax6.set_xticks(range(8))
    ax6.set_yticks(range(8))
    ax6.set_zticks(range(8)) 
    ax6.scatter(int(slidesd1.val%8)+0.5,int(slidesd1.val/8)+0.5, color='g')
    ax6.voxels(Vox3, facecolors='r', edgecolor='gray')                                 # 3D plot 
    ax5.cla()
    ax5.bar(range(11),ExtractTPSF(int(slidesd1.val),int(slidexp1.val),int(slideyp1.val),int(slidezp1.val)))  
    plt.title('Contrast TPSF')
    f4.canvas.draw()
slidesd1.on_changed(update4)
slideyp1.on_changed(update4)
slidexp1.on_changed(update4)
slidezp1.on_changed(update4)

def update1(a):
    im1.set_data(ExtractTomography(int(slidet1.val),int(slidexp.val),int(slideyp.val),int(slidezp.val)))
slidet1.on_changed(update1)

def update2(a):
    Vox1[:,:,:]=False
    Vox1[int(slidexp.val),int(slideyp.val),int(slidezp.val)]=True
    ax1.cla()
    ax1.set_xticks(range(8))
    ax1.set_yticks(range(8))
    ax1.set_zticks(range(8)) 
    for e in range(8):
        for u in range(8):
            ax1.scatter(u+0.5,e+0.5,0, color='g')
    ax1.voxels(Vox1, facecolors='r', edgecolor='gray')
    im1.set_data(ExtractTomography(int(slidet1.val),int(slidexp.val),int(slideyp.val),int(slidezp.val)))
    f1.canvas.draw()
slideyp.on_changed(update2)
slidexp.on_changed(update2)
slidezp.on_changed(update2)

def update3(a):
    Vox2[:,:,:]=False
    Vox2[:,:,int(slidepl.val)]=True
    ax3.cla()
    ax3.set_xticks(range(8))
    ax3.set_yticks(range(8))
    ax3.set_zticks(range(8))
    ax3.scatter(int(slidesd.val%8)+0.5,int(slidesd.val/8)+0.5, color='g')
    ax3.voxels(Vox2, facecolors='r', edgecolor='gray')
    im.set_data(ExtractSensitivityProfile(int(slidet.val),int(slidesd.val),int(slidepl.val)))
    f2.canvas.draw()
slidepl.on_changed(update3)
slidesd.on_changed(update3)


                                             #-------------- --------- --------------
f9 = plt.figure(9)
plt.plot(M)

plt.show()





