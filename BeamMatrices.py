import numpy as np
from scipy import optimize

def dispersionRelSol(sol, *data):
    g, d, T = data
    L = sol
    return (L-(g/2/np.pi*T*T*np.tanh(2*np.pi/L*d)))

def getWaveLen(g, d, T):
    data = (g, d, T)
    root = optimize.fsolve(dispersionRelSol, g/2/np.pi*T*T ,args=data)
    return(root[0])

class LinearWave2D:
    # LinearWave2D(rhoW, g, d, T, H, phi=0, x0=0, msg=True)
    # rhoW, gravity, water-depth, time-period, wave-height, wave-phase, x0, printMsg
    def __init__(self, rhoW, g, d, T, H, phi=0.0, x0=0.0, msg=True):
        self.rhoW = rhoW
        self.g = g
        self.d = d
        self.x0 = x0
        self.phi = phi                
        self.T = T        
        self.H = H                
        self.L = getWaveLen(self.g, d, T)        
        self.k = 2*np.pi/self.L
        self.w = 2*np.pi/self.T

        if(msg):
            print('Wave-Length L = ', self.L)
            print('d/L = ', self.d / self.L)
        
    def wavePhase(self, t, x):
        return self.k*(x-self.x0) - self.w*t + self.phi
        
    
    def waveElevation(self, t, x):
        et = self.H / 2 * np.cos( self.wavePhase(t, x) )
        return et
    
    
    def pressureTotPoi(self, t, x, z):
        et = self.waveElevation(t, x)
        if(z>0):
            if(z>et):
                pdyn = 0.0*et
            else:
                pdyn = et-z
        else:
            if(z<et):
                Kp = np.cosh( self.k* (self.d + z) ) / np.cosh( self.k* self.d )
                pdyn = Kp*et - z
            else:
                pdyn=0.0*et
        return self.rhoW * self.g * pdyn
    
    def pressureTot(self, t, x, z):
        PdynRes = [self.pressureTotPoi(t,ix,iz) for ix, iz in zip(x,z)]        
        return PdynRes
    
    
    def pressureDynPoi(self, t, x, z):
        et = self.waveElevation(t, x)
        if(z>0):
            if(z>et):
                pdyn = 0.0*et
            else:
                pdyn = et-z
        else:
            if(z<et):
                Kp = np.cosh( self.k* (self.d + z) ) / np.cosh( self.k* self.d )
                pdyn = Kp*et                
            else:
                pdyn=0.0*et
        return self.rhoW * self.g * pdyn
    
    def pressureDyn(self, t, x, z):
        PdynRes = [self.pressureDynPoi(t,ix,iz) for ix, iz in zip(x,z)]        
        return PdynRes
    
    
    def particleVelPoi(self, t, x, zin):        
        et = self.waveElevation(t, x)   
        # # Wheeler stretching
        # z = self.d * ( self/d + zin ) / ( self/d + et ) - self.d     
        # Not applying Wheeler Stretching
        z = zin
        if(z>0):
            vx = 0.0 * et
            vz = 0.0 * et
        else:
            mag = self.H/2 * self.w
            vx =  mag * np.cosh( self.k*(self.d + z) ) / np.sinh( self.k * self.d )
            vx = vx * np.cos( self.wavePhase(t, x) )
            vz =  mag * np.sinh( self.k*(self.d + z) ) / np.sinh( self.k * self.d )
            vz = vz * np.sin( self.wavePhase(t, x) )
        return vx, vz
    
    
    def particleAccPoi(self, t, x, zin):
        et = self.waveElevation(t, x)
        # # Wheeler stretching
        # z = self.d * ( self/d + zin ) / ( self/d + et ) - self.d     
        # Not applying Wheeler Stretching
        z = zin
        if(z>0):
            vx = 0.0 * et
            vz = 0.0 * et
        else:
            mag = self.H/2 * self.w**2
            vx =  mag * np.cosh( self.k*(self.d + z) ) / np.sinh( self.k * self.d )
            vx = vx * np.sin( self.wavePhase(t, x) )
            vz =  mag * np.sinh( self.k*(self.d + z) ) / np.sinh( self.k * self.d )
            vz = -vz * np.cos( self.wavePhase(t, x) )
        return vx, vz

    
    def particleVelMax(self, x, zin):
        et = self.H  / 2
        # # Wheeler stretching
        # z = self.d * ( self/d + zin ) / ( self/d + et ) - self.d     
        # Not applying Wheeler Stretching
        z = zin
        mag = self.H/2 * self.w
        vx =  mag * np.cosh( self.k*(self.d + z) ) / np.sinh( self.k * self.d )            
        vz =  mag * np.sinh( self.k*(self.d + z) ) / np.sinh( self.k * self.d )
        if(z>0):
            vx = vx * 0.0
            vz = vz * 0.0
        return vx, vz


    def particleAccMax(self, x, zin):        
        et = self.H  / 2
        # # Wheeler stretching
        # z = self.d * ( self/d + zin ) / ( self/d + et ) - self.d     
        # Not applying Wheeler Stretching
        z = zin
        mag = self.H/2 * self.w**2
        vx =  mag * np.cosh( self.k*(self.d + z) ) / np.sinh( self.k * self.d )
        vz =  mag * np.sinh( self.k*(self.d + z) ) / np.sinh( self.k * self.d )
        if(z>0):
            vx = vx * 0.0
            vz = vz * 0.0
        return vx, vz
            
    
# LinearWave2D(rhoW, g, d, T, H, phi=0, x0=0, msg=True)
# rhoW, gravity, water-depth, time-period, wave-height, wave-phase, x0, printMsg




# class LinearWaveDeep2D:
#     # LinearWaveDeep2D(rhoW, g, T, H, phi=0, x0=0, msg=True)
#     # rhoW, gravity, time-period, wave-height, wave-phase, x0, printMsg
#     def __init__(self, rhoW, g, T, H, phi=0.0, x0=0.0, msg=True):
#         self.rhoW = rhoW
#         self.g = g
#         self.T = T
#         self.H = H
#         self.phi = phi
#         self.x0 = x0
#         self.L = g/2.0/np.pi * T**2
#         self.k = 2*np.pi/self.L
#         self.w = 2*np.pi/self.T
#         if(msg):
#             print('Wave-Length L = ', self.L)
        
#     def wavePhase(self, t, x):
#         return self.k*(x-self.x0) - self.w*t + self.phi
        
    
#     def waveElevation(self, t, x):
#         et = self.H / 2 * np.cos( self.wavePhase(t, x) )
#         return et
    
    
#     def pressureTotPoi(self, t, x, z):
#         et = self.waveElevation(t, x)
#         if(z>0):
#             if(z>et):
#                 pdyn = 0.0*et
#             else:
#                 pdyn = et-z
#         else:
#             if(z<et):
#                 Kp = np.exp( self.k * z )
#                 pdyn = Kp*et - z
#             else:
#                 pdyn=0.0*et
#         return self.rhoW * self.g * pdyn
    
#     def pressureTot(self, t, x, z):
#         PdynRes = [self.pressureTotPoi(t,ix,iz) for ix, iz in zip(x,z)]        
#         return PdynRes
    
    
#     def pressureDynPoi(self, t, x, z):
#         et = self.waveElevation(t, x)
#         if(z>0):
#             if(z>et):
#                 pdyn = 0.0
#             else:
#                 pdyn = et-z
#         else:
#             if(z<et):
#                 Kp = np.exp( self.k * z )
#                 pdyn = Kp*et
#             else:
#                 pdyn=0.0*et
#         return self.rhoW * self.g * pdyn
    
#     def pressureDyn(self, t, x, z):
#         PdynRes = [self.pressureDynPoi(t,ix,iz) for ix, iz in zip(x,z)]        
#         return PdynRes
    
    
#     def particleVelPoi(self, t, x, z):
#         et = self.waveElevation(t, x)
#         if(z>0):
#             vx = 0.0 * et
#             vz = 0.0 * et
#         else:
#             mag = self.H/2 * self.w
#             vx =  mag * np.exp( self.k * z )
#             vx = vx * np.cos( self.wavePhase(t, x) )
#             vz =  mag * np.exp( self.k * z )
#             vz = vz * np.sin( self.wavePhase(t, x) )
#         return vx, vz
    
    
#     def particleAccPoi(self, t, x, z):
#         et = self.waveElevation(t, x)
#         if(z>0):
#             vx = 0.0 * et
#             vz = 0.0 * et
#         else:
#             mag = self.H/2 * self.w**2
#             vx =  mag * np.exp( self.k * z )
#             vx = vx * np.sin( self.wavePhase(t, x) )
#             vz =  mag * np.exp( self.k * z )
#             vz = -vz * np.cos( self.wavePhase(t, x) )
#         return vx, vz

    
#     def particleVelMax(self, x, z):
#         mag = self.H/2 * self.w
#         vx =  mag * np.exp( self.k * z )
#         vz =  mag * np.exp( self.k * z )
#         if(z>0):
#             vx = vx * 0.0
#             vz = vz * 0.0
#         return vx, vz


#     def particleAccMax(self, x, z):        
#         mag = self.H/2 * self.w**2
#         vx =  mag * np.exp( self.k * z )
#         vz =  mag * np.exp( self.k * z )
#         if(z>0):
#             vx = vx * 0.0
#             vz = vz * 0.0
#         return vx, vz  
    
# # LinearWaveDeep2D(rhoW, g, T, H, phi=0, x0=0, msg=True)
# # rhoW, gravity, time-period, wave-height, wave-phase, x0, printMsg






def Beam3DMatrices(m, EA, EI, GJ, Im, NodeCoord):
# Inputs:
# m         - mass per unit length [kg/m]
# EA        - axial stiffness [N]
# EI        - bending stiffness [N.m2]
# NodeCoord - ([xl, yl, zl], [xr, yr, zr])
#           - left (l) and right (r) node coordinates

    # 1 - calculate length of beam (L) and orientation alpha
    xl = NodeCoord[0][0]    # x-coordinate of left node
    yl = NodeCoord[0][1]    # y-coordinate of left node
    zl = NodeCoord[0][2]    # z-coordinate of left node
    xr = NodeCoord[1][0]    # x-coordinate of right node
    yr = NodeCoord[1][1]    # y-coordinate of rigth node
    zr = NodeCoord[1][2]    # z-coordinate of rigth node
    L = np.sqrt((xr - xl)**2 + (yr - yl)**2 + (zr - zl)**2)    # length
    
    # 2 - calculate transformation matrix T
    C = (xr-xl)/L  # cosine of angle between beam and global X axis
    S = (yr-yl)/L  # sine of angle between beam and global X axis
    # T in this is different from T in the above Beam 2D function
    T = np.array([[C, S, 0], [-S, C, 0], [0, 0, 1]])
    
    # Only support rotation in XY plane
    if(abs(zr - zl) > 1e-6):
        print("Error, Only supports rotation in XY plane")
        T = np.array([[1, 0, 0], [0, 1, 0], [0,0,1]])        
    
    T = np.asarray(np.bmat([[T, np.zeros((3,3))], [np.zeros((3, 3)), T]]))    
    T = np.asarray(np.bmat([[T, np.zeros((6,6))], [np.zeros((6, 6)), T]]))    
    # print(T)

    # 3 - calculate local stiffness and matrices
    L2 = L*L
    L3 = L*L2
    K = np.array([[EA/L, 0, 0, 0, 0, 0, -EA/L, 0, 0, 0, 0, 0], 
                  [0, 12*EI/L3, 0, 0, 0, 6*EI/L2, 0, -12*EI/L3, 0, 0, 0, 6*EI/L2], 
                  [0, 0, 12*EI/L3, 0, -6*EI/L2, 0, 0, 0, -12*EI/L3, 0, -6*EI/L2, 0], 
                  [0, 0, 0, GJ/L, 0, 0, 0, 0, 0, -GJ/L, 0, 0],
                  [0, 0, -6*EI/L2, 0, 4*EI/L, 0, 0, 0, 6*EI/L2, 0, 2*EI/L, 0], 
                  [0, 6*EI/L2, 0, 0, 0, 4*EI/L, 0, -6*EI/L2, 0, 0, 0, 2*EI/L], 
                  [-EA/L, 0, 0, 0, 0, 0, EA/L, 0, 0, 0, 0, 0], 
                  [0, -12*EI/L3, 0, 0, 0, -6*EI/L2, 0, 12*EI/L3, 0, 0, 0, -6*EI/L2], 
                  [0, 0, -12*EI/L3, 0, 6*EI/L2, 0, 0, 0, 12*EI/L3, 0, 6*EI/L2, 0], 
                  [0, 0, 0, -GJ/L, 0, 0, 0, 0, 0, GJ/L, 0, 0],                  
                  [0, 0, -6*EI/L2, 0, 2*EI/L, 0, 0, 0, 6*EI/L2, 0, 4*EI/L, 0],
                  [0, 6*EI/L2, 0, 0, 0, 2*EI/L, 0, -6*EI/L2, 0, 0, 0, 4*EI/L]])    
    
    M = np.array([[140, 0, 0, 0, 0, 0, 70, 0, 0, 0, 0, 0], 
                  [0, 156, 0, 0, 0, 22*L, 0, 54, 0, 0, 0, -13*L], 
                  [0, 0, 156, 0, 22*L, 0, 0, 0, 54, 0, 13*L, 0], 
                  [0, 0, 0, 140*Im, 0, 0, 0, 0, 0, 70*Im, 0, 0],
                  [0, 0, 22*L, 0, 4*L2, 0, 0, 0, -13*L, 0, -3*L2, 0], 
                  [0, 22*L, 0, 0, 0, 4*L2, 0, 13*L, 0, 0, 0, -3*L2], 
                  [70, 0, 0, 0, 0, 0, 140, 0, 0, 0, 0, 0], 
                  [0, 54, 0, 0, 0, 13*L, 0, 156, 0, 0, 0, -22*L], 
                  [0, 0, 54, 0, -13*L, 0, 0, 0, 156, 0, 22*L, 0], 
                  [0, 0, 0, 70*Im, 0, 0, 0, 0, 0, 140*Im, 0, 0],
                  [0, 0, 13*L, 0, -3*L2, 0, 0, 0, 22*L, 0, 4*L2, 0],
                  [0, -13*L, 0, 0, 0, -3*L2, 0, -22*L, 0, 0, 0, 4*L2]])
    
    M_added = 443000 # kg/m, added mass per unit length for offshore wind turbine blades, 
    M_added = M_added * L / 420 * np.array([
                [0, 0,   0,   0, 0,       0,    0, 0,   0,   0, 0,       0],
                [0, 156, 0,   0, 0,       22*L, 0, 54,  0,   0, 0,      -13*L],
                [0, 0,   156, 0, 22*L,    0,    0, 0,   54,  0, 13*L,    0],
                [0, 0,   0,   0, 0,       0,    0, 0,   0,   0, 0,       0],
                [0, 0,   22*L,0, 4*L**2,  0,    0, 0,  -13*L,0,-3*L**2, 0],
                [0, 22*L,0,   0, 0,       4*L**2,0,13*L,0,   0, 0,      -3*L**2],
                [0, 0,   0,   0, 0,       0,    0, 0,   0,   0, 0,       0],
                [0, 54,  0,   0, 0,       13*L, 0, 156, 0,   0, 0,      -22*L],
                [0, 0,   54,  0,-13*L,    0,    0, 0,   156, 0, 22*L,   0],
                [0, 0,   0,   0, 0,       0,    0, 0,   0,   0, 0,       0],
                [0, 0,   13*L,0,-3*L**2,  0,    0, 0,   22*L,0,4*L**2,  0],
                [0,-13*L,0,   0, 0,      -3*L**2,0,-22*L,0,  0, 0,       4*L**2]])


    M = m*L/420 * M + M_added
    
    Q = np.array([[140, 0, 0, 0, 0, 0, 70, 0, 0, 0, 0, 0], 
                  [0, 156, 0, 0, 0, 22*L, 0, 54, 0, 0, 0, -13*L], 
                  [0, 0, 156, 0, 22*L, 0, 0, 0, 54, 0, 13*L, 0], 
                  [0, 0, 0, 140, 0, 0, 0, 0, 0, 70, 0, 0],
                  [0, 0, 22*L, 0, 4*L2, 0, 0, 0, -13*L, 0, -3*L2, 0], 
                  [0, 22*L, 0, 0, 0, 4*L2, 0, 13*L, 0, 0, 0, -3*L2], 
                  [70, 0, 0, 0, 0, 0, 140, 0, 0, 0, 0, 0], 
                  [0, 54, 0, 0, 0, 13*L, 0, 156, 0, 0, 0, -22*L], 
                  [0, 0, 54, 0, -13*L, 0, 0, 0, 156, 0, 22*L, 0], 
                  [0, 0, 0, 70, 0, 0, 0, 0, 0, 140, 0, 0],
                  [0, 0, 13*L, 0, -3*L2, 0, 0, 0, 22*L, 0, 4*L2, 0],
                  [0, -13*L, 0, 0, 0, -3*L2, 0, -22*L, 0, 0, 0, 4*L2]])
    Q = L/420 * Q


    Cdamp = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    
   


    # F_static = 648000 * L /2 # N,static for the half of the length
    # Fv_inertia = particleVelPoi(self, t, x, zin)[0] * * L / 2 # N, inertia for the half of the length
    # F = np.array([0, F_inertia + F_drag + F_static, F_inertia + F_drag, 0, 0, 0, 
    #                0, F_inertia + F_drag + F_static, F_inertia + F_drag, 0, 0, 0])
    

    # 4 - rotate the matrices
    K = np.matmul(np.transpose(T), np.matmul(K, T))
    M = np.matmul(np.transpose(T), np.matmul(M, T))
    Q = np.matmul(np.transpose(T), np.matmul(Q, T))
    Cdamp = np.matmul(np.transpose(T), np.matmul(Cdamp, T))
    return M, K, Q, Cdamp