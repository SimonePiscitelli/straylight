import numpy as np
import astropy.constants as ac
import random
from scipy.interpolate import interp1d
from scipy.integrate import quad
from astropy.io import fits
from matplotlib import pyplot as plt


from skimage.draw import polygon, disk
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy import units as u

def DefParWolterPolyForm(FocalLenght, r0):
    z0 = -FocalLenght
    
    alfa = (1/4)*np.arctan(r0/z0)   #angolo di inclinazione tra parabola e asse: perché 1/4?
    a0 = 1
    a1 = (np.tan(alfa))*(-2)
    a2 = 0
    r_par_square = np.array([a2, r0*a1, (r0**2)*a0])     

    PoliFrontPolTot = np.array([0, 0, 0, *r_par_square])
    return PoliFrontPolTot
    

#funzione polinomiale dell'iperbole
def DefParWolterHypForm(FocalLenght, r0):
    z0 = -FocalLenght
    F = z0/(2*r0)
    alfa = (1/4)*np.arctan(r0/z0)
    
    beta = alfa + (np.arctan(2/F))/2
    b0 = 1.
    b1 = np.tan(beta)*(-2)
    b2 = 2*r0*np.tan(beta)/(z0+r0*(1/np.tan(2*alfa)))

    r_hyp_square = [b2, r0*b1, r0**2*b0]
    PoliRearPolTot = [0,0 ,0, *r_hyp_square]

    return PoliRearPolTot


def check_intersect_pol(x_ph, y_ph, z_ph, PhVelX, PhVelY, PhVelZ, PolTot, LimBot, LimTop):
    
    MaxSol = -100000000
    z0 = MaxSol

    c1 = np.round(((x_ph * PhVelX) + (y_ph * PhVelY)) / PhVelZ,4)
    c2 = np.round((PhVelX ** 2 + PhVelY ** 2) / PhVelZ ** 2,4)
    p0 = np.round(x_ph ** 2 + y_ph ** 2 - 2 * z_ph * c1 + z_ph ** 2 * c2,4)
    p1 = np.round(2 * c1 - 2 * z_ph * c2,4)
    p2 = c2

    pol_intercept = np.zeros(len(PolTot))
    #pol_intercept[-3:] = [p2, p1[0], p0[0]]
    #pol_intercept[-3:] = [p0[0], p1[0], p2]
    pol_intercept[-3:] = p2, p1, p0

    PolTot = np.asarray(PolTot)
    pol_for_roots = -PolTot + pol_intercept
    z_int = np.roots(pol_for_roots)
    NumSol = len(z_int)
    MaxSol = -100000000

    for k in range(NumSol):
        RealSol = np.isreal(z_int[k])
        if RealSol == True:
            if LimBot < z_int[k] <= LimTop:
                z0 = z_int[k]
                MaxSol = max(MaxSol, z0)

    return MaxSol

def cart2pol(x,y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y,x)
    return r,theta



def RayPropagate(z_check,x0,y0,z0,vx,vy,vz):

    # x = x0 + vx * t;
    # y = y0 + vy * t;
    # z = z0 + vz * t;
    
    z = z_check
    t = (z-z0)/vz
    x = x0 + vx*t
    y = y0 + vy*t

    return x,y,z

def Calc_NormSupPol(pol_sup_square, z_ph,r_ph,theta_ph):
    
    #calcolo la derivata del polinomio (iperbole o parabola)
    deriv_pol = np.polyder(pol_sup_square)

    #calcolo il valore della tangente alla curva nel punto di riflessione
    tang =  np.polyval(deriv_pol,z_ph)

    #faccio una cosa che non capisco
    tang_par = (1/2) * (1/r_ph) * tang

    #calcolo un angolo che avrà a che fare con la riflessione
    ang_tang = np.arctan(tang_par)

    #definisco la normale
    NormSup = np.zeros(3)
    NormSup[0]= -np.cos(theta_ph)*np.sqrt(1-np.sin(ang_tang)**2)
    NormSup[1] = -np.sin(theta_ph)*np.sqrt(1-np.sin(ang_tang)**2)
    NormSup[2] = np.sin(ang_tang)

    return NormSup


def PhReflex(v_init,NormSup):

    #genero il vettore velocità post riflessione 
    v_out = np.zeros((1,3))

    v_out = v_init - 2*NormSup * np.dot(v_init, NormSup)

    return v_out


    
#function [vx,vy,vz] = PhReflex(v_init,NormSup)
 #   % vector reflexted, enter (3,1) --> exit (1,3)
 #   v_out(:,1) = v_init(:,1) -2 * NormSup(:,1) * dot(v_init(:,1),NormSup(:,1)); 
 #   vx = v_out(1,1);
 #   vy = v_out(2,1);
 #   vz = v_out(3,1);
#end
        


def deformation_matrix_gen(t,amp,zp,zh,freq,spec_index,method):
    l = len(t)

    whs = []
    wps = []

    if method==1:
        shift1=np.random.uniform(0,2*np.pi)
        shift2=np.random.uniform(0,2*np.pi)
        for i in range(len(t)):
            #voglio introdurre una deformazione che sia somma di sinusoidi a frequenza differente
            #definisco un numero di frequenze che deciderà a che livello saranno presenti le deformazioni
        
            w1 = []
            w2 = []
            
            n=len(freq)
            amps = amp*freq**(-spec_index)
            
            for j in range(n):
                w1.append(amps[j]*np.sin(2*np.pi*freq[j]*zp/max(zp) + shift1))
                w2.append(amps[j]*np.sin(2*np.pi*freq[j]*zh/max(abs(zh)) + shift2))
                
            w1 = np.asarray(w1)
            w2 = np.asarray(w2)
            
            w1_tot = np.sum(w1, axis = 0)
            w2_tot = np.sum(w2, axis = 0)
            
            wps.append(w1_tot)
            whs.append(w2_tot)
           
                
    return np.array([wps,whs])


def assign_energy(k,scaled_threshold,dE):
    #assegno un'energia ai fotoni ed una riflettività
    for j in range(len(scaled_threshold)):
        if k<scaled_threshold[j]:
            low = min(dE[j])
            high = max(dE[j])
            EPh=np.random.uniform(low,high)
            break
        else:
            continue
    return EPh


def calculate_PSD(freq,Kn,n):
    return Kn*freq**(-n)


def baffle_shadow(h,r,theta,iota,height,n,mirror):
    #limiti angolari

    #considero un semipiano di centro iota
    alpha = iota + np.pi/2
    beta = iota + np.pi*3/2

    #inverto gli estremi
    #alpha2=alpha+np.pi
    #beta2=beta+np.pi
    
    phi = np.random.uniform(alpha,beta,n)
    #phi2 = np.random.uniform(beta,alpha+2*np.pi,n)
    #phi2 = np.random.uniform(alpha,beta,M)
    phi2 = phi+np.pi
    
    
    theta_rad = theta/180*np.pi
    #shadows
    shadow1 = height[h]*np.tan(theta_rad)
    shadow2 = height[h-1]*np.tan(theta_rad)

    #specchio interno
    if mirror==1:
        phi_norm = abs(phi) - alpha
        x_baff = (r[h-1]*np.cos(phi) - shadow1*np.sin(phi_norm)*(np.cos(iota)/abs(np.cos(iota))))
        y_baff = (r[h-1]*np.sin(phi) - shadow1*np.sin(phi_norm)*(np.sin(iota)/abs(np.sin(iota))))
        
    #specchio esterno
    else:
        phi_norm2 = abs(phi) + (np.pi-alpha)
        x_baff = (r[h]*np.cos(phi2) + shadow2*np.sin(phi_norm2)*(np.cos(iota)/abs(np.cos(iota))))
        y_baff = (r[h]*np.sin(phi2) + shadow2*np.sin(phi_norm2)*(np.sin(iota)/abs(np.sin(iota))))
        
    return x_baff, y_baff
    
    
def deg_to_rad(angle):
    return angle/180*np.pi

def rad_to_deg(angle):
    return angle/np.pi*180


def deg2arcs(angle):
    return angle*3600

def arcs2deg(angle):
    return angle/3600

#funzione polinomiale della parabola
def DefParWolterPolyFormShifted(FocalLenght, r0, delta, omega,t):
    
    #---- defining variable r
    n=10_000
    t_tmp = np.random.uniform(0,np.pi*2,n)
    t_tmp[-1] = t
    
    x = r0*np.cos(t_tmp) + delta*np.cos(omega)
    y = r0*np.sin(t_tmp) + delta*np.sin(omega)
    
    r_new = np.sqrt(x**2 + y**2)
    f = interp1d(t_tmp,r_new)
    r0 = f(t)
    
    #----- original function
    z0 = -FocalLenght
    
    alfa = (1/4)*np.arctan(r0/z0)   #angolo di inclinazione tra parabola e asse: perché 1/4? E' una configurazione che massimizza l'efficienza degli specchi
    a0 = 1
    a1 = (np.tan(alfa))*(-2) 
    a2 = 0

    PoliFrontPolTot = []
   
    r_par_square = np.array([a2, r0*a1, (r0**2)*a0])     #a2 definisce un'intersezione con l'origine
    PoliFrontPolTot.append(np.array([0, 0, *r_par_square]))
    return PoliFrontPolTot


#funzione polinomiale dell'iperbole
def DefHypWolterPolyFormShifted(FocalLenght, r0, delta, omega, t):
    
    #---- defining variable r
    n=10_000
    t_tmp = np.random.uniform(0,2*np.pi,n)
    t_tmp[-1] = t

    x = r0*np.cos(t_tmp) + delta*np.cos(omega)
    y = r0*np.sin(t_tmp) + delta*np.sin(omega)
    
    r_new = np.sqrt(x**2 + y**2)
    f = interp1d(t_tmp,r_new)
    r0 = f(t)
    
    #---- original function
    z0 = -FocalLenght
    F = z0/(2*r0)
    alfa = (1/4)*np.arctan(r0/z0)

    beta = alfa + np.arctan(1/(2*F))/2
    b0 = 1
    b1 = np.tan(beta)*(-2) 
    b2 = 2*r0*np.tan(beta)/(z0+r0/np.tan(2*alfa)) 

    PoliRearPolTot = []
    
    r_hyp_square = np.array([b2, r0*b1, (r0**2)*b0])
    PoliRearPolTot.append(np.array([0,0, *r_hyp_square]))

    return PoliRearPolTot



def generate_random_point(t, center1, radius1, center2, radius2):
        """
        Genera un punto casuale all'interno della regione compresa tra due circonferenze.

        Args:
        t: Angolo compreso tra 0 e 2*pi che definisce la posizione del punto sulla circonferenza.
        center1: Coordinate del primo centro (array-like).
        radius1: Raggio della prima circonferenza (float).
        center2: Coordinate del secondo centro (array-like).
        radius2: Raggio della seconda circonferenza (float).

        Returns:
        Coordinate del punto casuale (array-like).
        """

        # Controlla se i parametri sono validi
        if not (0 <= t <= 2*np.pi):
            raise ValueError("L'angolo t deve essere compreso tra 0 e 2*pi.")
        if radius1 <= 0 or radius2 <= 0:
            raise ValueError("I raggi delle circonferenze devono essere positivi.")

        # Calcola la distanza tra i centri
        delta_center = center2 - center1

        # Definisce i parametri per la circonferenza equivalente
        center_eq = (center1 + center2) / 2
        radius_eq = np.linalg.norm(delta_center) / 2

        # Calcola l'angolo di offset
        alpha = np.arctan2(delta_center[1], delta_center[0])

        # Genera un punto casuale sulla circonferenza equivalente
        phi = t + alpha
        x = radius_eq * np.cos(phi)
        y = radius_eq * np.sin(phi)

        # Sposta il punto nella posizione corretta
        x = x + center_eq[0] - center1[0]
        y = y + center_eq[1] - center1[1]

        # Genera un raggio casuale all'interno dell'intervallo [0, min(radius1, radius2)]
        random_radius = random.uniform(0, min(radius1, radius2))

        # Calcola le coordinate finali del punto
        x_final = x * random_radius / radius_eq
        y_final = y * random_radius / radius_eq

        # Controlla se il punto si trova all'interno della regione
        if np.linalg.norm([x_final, y_final] - center1) <= radius1 and np.linalg.norm([x_final, y_final] - center2) <= radius2:
            return [x_final, y_final]
        else:
            raise ValueError("Impossibile generare un punto all'interno della regione.")


            
#voglio creare una funzione che restituisca l'istogramma

def histo_shells(xh,yh,MinMirror,MaxMirror,det_dim,phi_deg,width_deg):
    '''
        This function determines the histogram of the radial profile of straylights in a given direction.
        xh and yh must be lists of coordinates of singular shells: each list is composed by arrays of coordinates
        of the shells.
        
        MinMirror and MaxMirror represents the first and last shell one wants to visualize.
        
        det_dim represents the dimension of the detector which is suppose to be a square
        
        
    '''
    
    #create two empty lists for change of reference
    rs = []
    thetas = []

    for i in range(MinMirror,MaxMirror):
        NumShell = i - MinMirror

        #index to select points in the detector area
        idxc = (xh[NumShell] < det_dim) & (xh[NumShell] > -det_dim) & (yh[NumShell] < det_dim) & (yh[NumShell] > -det_dim)

        #change of reference
        xs_sh = xh[NumShell][idxc] - det_dim
        ys_sh = yh[NumShell][idxc] + det_dim

        #xs_sh_trans.append(xs_sh)
        #ys_sh_trans.append(ys_sh)

        #polar coordinates
        rho_xh, theta_xh = cart2pol(xs_sh,ys_sh)

        #defining an angle in rad
        phi = deg_to_rad(phi_deg)

        #defining a width for the angle
        width = deg_to_rad(width_deg)

        #defining an index
        i_angle = (theta_xh < phi+width) & (theta_xh > phi-width)

        #filter of the rays
        rs_tmp = rho_xh[i_angle]
        thetas_tmp = theta_xh[i_angle]
        
        rs.append(rs_tmp)
        thetas.append(thetas)
    
    #agglomerate the shells    
    fin_rs = []
    for i in range(len(rs)):
        for j in range(len(rs[i])):
            fin_rs.append(rs[i][j])
            
    histogram = plt.hist(fin_rs, density=True, histtype = 'step', bins = 30)
    return histogram



def matrix_to_coords(threshold, final_matrix):
    # Trova gli indici in cui il valore è maggiore della soglia
    y_im, x_im = np.where(final_matrix > threshold)  # Restituisce array di coordinate

    # Concatena le coordinate in un array di shape (N, 2)
    coords_im = np.column_stack((x_im, y_im))

    return coords_im.T



def rotation_matrix(x,y,iota0):
    #iota0_rad = deg_to_rad(iota0)
    iota = iota0 - np.pi
    rot_matrix = np.array([[np.cos(iota),-np.sin(iota)],[np.sin(iota),np.cos(iota)]])
    coords = np.array([x,y])
    return rot_matrix@coords

def rotation_matrix_coords(x,y,gamma):
    x_f = x*np.cos(gamma) - y*np.sin(gamma)
    y_f = x*np.sin(gamma) + y*np.cos(gamma)    
    return x_f, y_f


def find_perpendicular_line(iota):
    """
    Trova il coefficiente angolare della retta che passa per l'origine e 
    ha come perpendicolare la retta che interseca il punto (cos(iota), sin(iota)).
    """
    m_perpendicular = np.tan(iota)  # Coefficiente angolare della perpendicolare
    m_line = -1 / m_perpendicular if m_perpendicular != 0 else np.inf  # Coefficiente angolare della retta
    return m_line

def reflect_across_line(x, y, iota):
    """
    Riflette il punto (x, y) rispetto alla retta passante per l'origine e con coefficiente angolare trovato.
    """
    m = find_perpendicular_line(iota)
    
    if np.isinf(m):  # Se la retta è verticale (caso speciale)
        return -x, y  # Riflesso rispetto all'asse y

    # Formula per la riflessione rispetto a una retta y = mx
    d = (x + m*y) / (1 + m**2)
    x_reflected = 2*d - x
    y_reflected = 2*d*m - y

    return x_reflected, y_reflected


def shift_reference_system(x,y,x_coord_opt,y_coord_opt,quadrant):

    if quadrant==1:
        x_fin = x + x_coord_opt
        y_fin = y + y_coord_opt
    
    if quadrant==2:
        x_fin = x - x_coord_opt
        y_fin = y + y_coord_opt
        
    if quadrant==3:
        x_fin = x - x_coord_opt
        y_fin = y - y_coord_opt
        
    if quadrant==4:
        x_fin = x + x_coord_opt
        y_fin = y - y_coord_opt
   
    return x_fin,y_fin

def defining_spider(num_spider,gap,len_spider,iota_rad,NumPh,direction):
    
    #creo degli intervalli
    phi_intervals = []
    for j in range(len(num_spider)):
        t_min = len_spider*j
        t_max = (j+1)*(len_spider)#-gap
        #tmp_phi = np.linspace(t_min,t_max,100)
        phi_intervals.append(np.array([t_min,t_max]))

    phi_intervals = np.asarray(phi_intervals)
    
    for j in range(len(phi_intervals)):
        
            t_min = phi_intervals[j][0]
            t_max = phi_intervals[j][1]

            if iota_rad > np.pi:
                #iota_tmp = iota_267_rad - np.pi
                iota_tmp = iota_rad - np.pi
            else:
                #iota_tmp = iota_267_rad + np.pi
                iota_tmp = iota_rad + np.pi

            if iota_tmp<t_max and iota_tmp>t_min:
                int_index = j
                #print('questo è il tuo intervallo: %i'%j)
                
                
    if direction=='cw':

        #calcolo la distanza tra l'estremo inferiore dell'intervallo e il mio angolo sottratto del gap
        dgap = abs(iota_tmp - (phi_intervals[int_index][1]))
        
        phi = []
        for i in range(NumPh):
            num_spider = np.arange(0,16)
            #num_spider = np.arange(15,15+17)
            gap = 0.045
            #gap = 0
            len_spider = np.pi/8 - gap
            j = int(np.random.uniform(0,len(num_spider)))
            #t = np.random.uniform(min_phi,max_phi + len_spider) + (len_spider+gap)*num_spider[j]
            t = np.random.uniform(0,len_spider) + (len_spider+gap)*num_spider[j] - (len_spider - dgap) #+ dgap
            #t2 = np.random.uniform(0,len_spider) + (len_spider+gap)*num_spider[j]
    
            phi.append(t)
    
        phi = np.asarray(phi)
        
    #dgap = abs(iota_tmp - (phi_intervals[int_index+1][0]))
    
    if direction=='ccw':
        #calcolo la distanza tra l'estremo inferiore dell'intervallo e il mio angolo sottratto del gap
        dgap = phi_intervals[int_index][1] - iota_tmp - gap
        
        phi = []
        for i in range(NumPh):
            num_spider = np.arange(0,16)
            #num_spider = np.arange(15,15+17)
            gap = 0.045
            #gap = 0
            len_spider = np.pi/8 - gap
            j = int(np.random.uniform(0,len(num_spider)))
            #t = np.random.uniform(min_phi,max_phi + len_spider) + (len_spider+gap)*num_spider[j]
            t = np.random.uniform(0,len_spider) + (len_spider+gap)*num_spider[j] + gap - dgap
            #t2 = np.random.uniform(0,len_spider) + (len_spider+gap)*num_spider[j]
    
            phi.append(t)
        
        phi = np.asarray(phi)
        
    return phi



def king_profile(r, r0, alpha):
    """Calcola il profilo di King.

    Args:
        r: Distanza dal centro della PSF
        r0: Raggio caratteristico
        alpha: Parametro di forma

    Returns:
        Intensità a distanza r
    """
    return (1 + (r/r0)**2)**(-alpha)

def eef(R, r0, alpha):
    """Calcola l'Energy Encircled Fraction.

    Args:
        R: Raggio del cerchio
        r0: Raggio caratteristico della PSF
        alpha: Parametro di forma

    Returns:
        EEF
    """

    # Integrale al numeratore
    def integrand_num(r):
        return r * king_profile(r, r0, alpha)
    num, _ = quad(integrand_num, 0, R)

    # Integrale al denominatore (normalizzazione)
    def integrand_den(r):
        return r * king_profile(r, r0, alpha)
    den, _ = quad(integrand_den, 0, np.inf)

    return num / den



def coords_to_matrix(xs,ys,det_dim):
    
        #x = []
        #y = []
        
        #for i in range(len(xs)):
        #    for j in range(len(xs[i])):
        #        x.append(xs[i][j])
        #        y.append(ys[i][j])
        #x = np.asarray(x)
        #y = np.asarray(y)

        
        idxc = (xs < det_dim) & (xs > -det_dim) & (ys < det_dim) & (ys > -det_dim)
        
        #definisco le nuove coordinate
        xs_sh = xs[idxc]
        ys_sh = ys[idxc]
        
        #creo i punti
        couples = np.array([xs_sh,ys_sh]).T

        #creo una matrice delle dimensioni dell'immagine fits
        zoom=200
        fits_mask = np.zeros((2*zoom,2*zoom))

        #definisco la grandezza del dominio della mappa
        interval = 2*det_dim #mm
        #bin = interval/2*zoom
    
    
        # Normalizza le coordinate nell'intervallo [0, 1]
        x_norm = (xs_sh + det_dim) / interval  # Normalizzazione tra 0 e 1
        y_norm = (ys_sh + det_dim) / interval

        # Trasforma le coordinate normalizzate nelle coordinate della matrice 400x400
        img_size = 400
        x_img = np.floor(x_norm * img_size).astype(int)
        y_img = np.floor(y_norm * img_size).astype(int)

        # Crea un'immagine FITS vuota di dimensioni 400x400
        fits_image = np.zeros((img_size, img_size))


        # Assegna i valori alle coordinate nella matrice dell'immagine
        for i in range(len(x_img)):
            fits_image[y_img[i], x_img[i]] += 1  # Incrementa il valore del pixel
            
        return fits_image
    

def save_matrix_to_fits(matrix, filename, metadata=None):
    """
    Salva una matrice in un file FITS.

    Parameters:
    - matrix (numpy.ndarray): La matrice da salvare.
    - filename (str): Il nome del file FITS da creare (es. "output.fits").
    - metadata (dict, opzionale): Dizionario di metadati da aggiungere all'header del file FITS.
    """
    # Creare l'oggetto PrimaryHDU
    hdu = fits.PrimaryHDU(matrix)

    # Aggiungere i metadati, se forniti
    if metadata:
        header = hdu.header
        for key, value in metadata.items():
            header[key] = value

    # Scrivere il file FITS
    hdu.writeto(filename, overwrite=True)
    print(f"File FITS salvato con successo: {filename}")



def transform_image_to_matrix(threshold,fitsfile,origin,pa,detsize):

    #apro il file fits
    with fits.open(fitsfile) as hdul:
        data = hdul[0].data
        #hdul.close()
        
        #pa_pnt = hdul[0].header['PA_PNT']

    if  np.pi < pa <= 3*np.pi/2:
        dpa = pa - 3*np.pi/2
    if 0 < pa <= np.pi/2:
        dpa = pa - np.pi/2

    #definisco il taglio
    zoom_x = 210
    zoom_y = 210

    x_img = origin[0]
    y_img = origin[1]

    roi_x_start, roi_x_end = x_img - zoom_x, x_img + zoom_x # Modifica questi valori secondo le tue preferenze
    roi_y_start, roi_y_end = y_img - zoom_y, y_img + zoom_y

    # Estrai l'immagine zoomata
    zoomed_image = data[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

    m = matrix_to_coords(threshold,zoomed_image)
    x = m[0]
    y = m[1]
    
    x_rot, y_rot = rotation_matrix_coords(x,y,dpa)

    return x_rot,y_rot



def rescale_matrix(x_rot,y_rot,detsize):

    detdim_x = detsize[0]/2
    detdim_y = detsize[1]/2

    max_dim_x = np.max(x_rot)
    max_dim_y = np.max(y_rot)

    min_dim_x = np.min(x_rot)
    min_dim_y = np.min(y_rot)

    shorten_factor = 0.96

    #x = (x_rot - min_dim_x)/(max_dim_x - min_dim_x)*(2*det_dim*shorten_factor) - det_dim*shorten_factor
    #y = (y_rot - min_dim_y)/(max_dim_y - min_dim_y)*(2*det_dim) - det_dim

    x = (x_rot - min_dim_x)/(max_dim_x - min_dim_x)*(2*detdim_x) - detdim_x
    y = (y_rot - min_dim_y)/(max_dim_y - min_dim_y)*(2*detdim_y) - detdim_y

    return x,y





def extract_from_region(region_file, data, header):

    wcs = WCS(header)
    markers = ['\"', '\'']

    mask_polygon = np.zeros(data.shape, dtype=bool)
    
    # Reading the region file
    with open(region_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            # This part handles both polygon and circle regions
            if line.startswith(('polygon', 'circle')):
                region_type = 'polygon' if line.startswith('polygon') else 'circle'
                
                # Split line by '#' to ignore comments, then by '(' and ')'
                coords_str = line.split('#')[0].strip().split('(')[-1].split(')')[0]
                
                # Find the unit marker at the end of the radius
                marker = ''
                if region_type == 'circle':
                    if coords_str.endswith('\"'):
                        marker = 'arcsec'
                    elif coords_str.endswith('\''):
                        marker = 'arcmin'
                    
                    # Clean up the string to get just the numbers
                    coords_str = coords_str.replace('\"', '').replace('\'', '')
                
                coords = list(map(float, coords_str.split(',')))
                

            else: 
                continue

            # Creating the mask
            if region_type == 'polygon':
                x_coords_fk5 = coords[::2]
                y_coords_fk5 = coords[1::2]
                skycoords = SkyCoord(x_coords_fk5, y_coords_fk5, unit="deg")
                x_coords, y_coords = wcs.world_to_pixel(skycoords)
                rr, cc = polygon(y_coords, x_coords, shape=data.shape)

                mask_polygon[rr, cc] = 1


            elif region_type == 'circle':
                x_center_fk5, y_center_fk5, radius_val = coords
                
                # Convert radius from celestial units to degrees
                if marker == 'arcsec':
                    radius_deg = radius_val * u.arcsec.to(u.deg)
                elif marker == 'arcmin':
                    radius_deg = radius_val * u.arcmin.to(u.deg)
                else: # Assumes degrees if no unit marker is found
                    radius_deg = radius_val
                    
                print(f'x center fk5: {x_center_fk5}')
                print(f'y center fk5: {y_center_fk5}')

                # Convert celestial coordinates to pixel coordinates
                skycoords = SkyCoord(x_center_fk5, y_center_fk5, unit="deg")

                print(f'skycoords: {skycoords}')

                x_center, y_center = wcs.world_to_pixel(skycoords)

                # Calculate the pixel scale of the image (degrees per pixel)
                # This is needed to convert the radius from degrees to pixels
                #pixel_scale_deg = np.sqrt(wcs.pixel_scale_matrix[0,0]**2 + wcs.pixel_scale_matrix[1,0]**2)
                #radius_pixels = radius_deg / pixel_scale_deg

                # Restituisce le scale in gradi/pixel per X e Y, tenendo conto della rotazione
                scales = proj_plane_pixel_scales(wcs) 
                # Usando la scala media (o geometrica) per il raggio
                pixel_scale_deg = np.mean(scales)
                radius_pixels = radius_deg / pixel_scale_deg
                
                rr, cc = disk((y_center, x_center), radius_pixels, shape=data.shape)

                mask_polygon[rr, cc] = 1

    return mask_polygon
    