import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser


from analytical_framework import (
    parametric_straylights_x,
    parametric_straylights_y,
    rotation_matrix
)

def return_reflectivity(map_data, teta, energy, show_plot=False):
    formatted_ref_data = []
    for i in range(len(map_data)):
        string = map_data.loc[i][0]
        substring = string.split()
        nums = [float(sub) for sub in substring]
        formatted_ref_data.append(np.asarray(nums))

    angles = formatted_ref_data[0]
    energies = np.asarray(formatted_ref_data[1:]).T[0]
    reflectivity_matrix = np.asarray(formatted_ref_data[1:]).T[1:]

    len_energies, len_angles = reflectivity_matrix.shape

    teta = np.clip(teta, 0, max(angles))
    energy = np.clip(energy, 0, max(energies))

    index_angles = np.round(teta * (len_angles - 1) / max(angles)).astype(int)
    index_energies = np.round(energy * (len_energies - 1) / max(energies)).astype(int)

    return reflectivity_matrix[index_energies, index_angles]


def trace_stray_light_1R_parametric(
    r_shells, 
    thicknesses, 
    theta_deg, 
    iota_deg, 
    gamma_deg=0.0, 
    xi_deg=0.0,
    delta_deg=0.0,
    epsilon_deg=0.0,
    energy_keV=1.0, 
    apply_reflectivity=True, 
    map_ref=None, 
    apply_geometry = True,
    focal_length=7500.0, 
    L=300.0, 
    num_photons=100000,
    PLOT=True
):
    theta_rad = np.radians(theta_deg)
    iota_rad = np.radians(iota_deg)
    gamma_rad = np.radians(gamma_deg)
    xi_rad = np.radians(xi_deg)
    delta_rad = np.radians(delta_deg)
    epsilon_rad = np.radians(epsilon_deg)
    
    kx = np.sin(theta_rad) * np.cos(iota_rad)
    ky = np.sin(theta_rad) * np.sin(iota_rad)
    kz = -np.cos(theta_rad)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    num_shells = len(r_shells) - 1
    colors = plt.cm.plasma(np.linspace(0, 1, num_shells))

    for n in range(num_shells):
        thick_outer = float(thicknesses[n])
        thick_inner = float(thicknesses[n+1])

        # Raggi netti al netto degli spessori
        R0_outer_net = float(r_shells[n]) - thick_outer / 2.0
        R0_inner_net = float(r_shells[n+1]) + thick_inner / 2.0
        
        alpha_out = R0_outer_net / (4.0 * focal_length)
        alpha_in  = R0_inner_net / (4.0 * focal_length)
        beta_out  = 3.0 * alpha_out
        beta_in   = 3.0 * alpha_in
        
        # --- 1. FILTRO DI SELEZIONE GEOMETRICA E SPESSORE CORONA ---
        r_top_outer = R0_outer_net + L * np.tan(alpha_out)
        r_top_inner = R0_inner_net + L * np.tan(alpha_in)
        
        if r_top_outer <= r_top_inner:
            continue

        # Generazione radiale uniforme all'interno dello spessore della corona
        r_gen = np.sqrt(np.random.uniform(r_top_inner**2, r_top_outer**2, num_photons))
        phi_gen = np.random.uniform(0, 2 * np.pi, num_photons)

        if apply_geometry:
        
            x0 = r_gen * np.cos(phi_gen)
            y0 = r_gen * np.sin(phi_gen)
            z0 = np.full(num_photons, L)
            
            # Passaggio Z = 0
            t_IP = (0.0 - z0) / kz
            x_IP = x0 + t_IP * kx
            y_IP = y0 + t_IP * ky
            r_IP = np.sqrt(x_IP**2 + y_IP**2)
            
            mask_pass_IP = (r_IP < R0_outer_net) & (r_IP > R0_inner_net)
            
            # Riflessione Iperbole
            tan_beta_avg = np.tan((beta_out + beta_in) / 2.0)
            k_trans = (kx * np.cos(phi_gen) + ky * np.sin(phi_gen)) / kz
            denom = k_trans - tan_beta_avg
            denom[np.abs(denom) < 1e-9] = 1e-9
            
            z_hit = (R0_outer_net - r_IP) / denom
            mask_hit_hyperbola = (z_hit <= 0.0) & (z_hit >= -L)
            
            t_hit = (z_hit - z0) / kz
            x_hit = x0 + t_hit * kx
            y_hit = y0 + t_hit * ky
            phi_hit = np.arctan2(y_hit, x_hit)
            
            beta_avg = (beta_out + beta_in) / 2.0
            nx = -np.cos(beta_avg) * np.cos(phi_hit)
            ny = -np.cos(beta_avg) * np.sin(phi_hit)
            nz = np.full(num_photons, np.sin(beta_avg))
            
            dot_prod = kx * nx + ky * ny + kz * nz
            k_ref_z = kz - 2.0 * dot_prod * nz
            
            # Passaggio Z = -L
            t_exit = (-L - z_hit) / k_ref_z
            x_exit = x_hit + t_exit * (kx - 2.0 * dot_prod * nx)
            y_exit = y_hit + t_exit * (ky - 2.0 * dot_prod * ny)
            r_exit = np.sqrt(x_exit**2 + y_exit**2)
            
            r_exit_max = R0_outer_net - L * np.tan(beta_out)
            r_exit_min = R0_inner_net - L * np.tan(beta_in)
            mask_pass_exit = (r_exit <= r_exit_max) & (r_exit >= r_exit_min)

            passed = mask_pass_IP & mask_hit_hyperbola & mask_pass_exit

        else:

            passed = np.ones(num_photons, dtype=bool)

        # --- 2. FILTRO DI RIFLETTIVITÀ ---
        if apply_reflectivity and map_ref is not None and np.sum(passed) > 0:
            sin_grazing = np.abs(dot_prod[passed])
            grazing_angle_deg = np.degrees(np.arcsin(sin_grazing))
            
            probs = return_reflectivity(map_ref, grazing_angle_deg, energy_keV)
            random_draw = np.random.uniform(0.0, 1.0, size=np.sum(passed))
            reflect_mask = random_draw < probs
            
            passed_indices = np.where(passed)[0]
            passed[passed_indices[~reflect_mask]] = False

        # --- 3. VALUTAZIONE EQUAZIONI PARAMETRICHE E ROTAZIONE FINALE ---
        if np.sum(passed) > 0:

            if apply_geometry:
                phi_passed = phi_hit[passed]
                r0_photons = r_IP[passed]
            else:
                phi_passed = phi_gen[passed]
                r0_photons = r_gen[passed]
            
            beta0_photons = 3.0 * (r0_photons / (4.0 * focal_length))
            
            phi_passed_rel = phi_passed - iota_rad + np.pi
            
            x_raw_param = parametric_straylights_x(
                theta=theta_rad, 
                iota=0.0, 
                gamma=gamma_rad[n], 
                xi=xi_rad[n], 
                delta=delta_rad, 
                epsilon=epsilon_rad, 
                r0=r0_photons, 
                beta0=beta0_photons, 
                FocalLenght=focal_length, 
                phi=phi_passed_rel
            )
            y_raw_param = parametric_straylights_y(
                theta=theta_rad, 
                iota=0.0, 
                gamma=gamma_rad[n], 
                xi=xi_rad[n], 
                delta=delta_rad, 
                epsilon=epsilon_rad, 
                r0=r0_photons, 
                beta0=beta0_photons, 
                FocalLenght=focal_length, 
                phi=phi_passed_rel
            )
            
            # ROTAZIONE FINALE DI PI GRECO AGGIUNTIVA (iota_rad + pi + pi = iota_rad + 2*pi)
            # per allineare perfettamente il punto focale nella direzione iota
            x_focal, y_focal = rotation_matrix(x_raw_param, y_raw_param, iota_rad + 2.0 * np.pi)
            
            ax1.scatter(x_focal, y_focal, s=1.2, color=colors[n], alpha=0.7)
            ax2.scatter(x_focal, y_focal, s=2.0, color=colors[n], alpha=0.7)

    # Output Grafico
    if PLOT:
        mode_str = f"Riflettività {'ATTIVA (' + str(energy_keV) + ' keV)' if apply_reflectivity else 'DISATTIVATA'}"
        ax1.set_title(f'Pattern Parametrico Orientato a $\\iota$ (Z = -{focal_length} mm)\n$\\theta = {theta_deg}^\\circ$, $\\iota = {iota_deg}^\\circ$| {mode_str}', fontsize=10)
        ax1.set_xlabel('X Piano Focale (mm)')
        ax1.set_ylabel('Y Piano Focale (mm)')
        ax1.grid(True, linestyle='--', alpha=0.4)
        ax1.set_aspect('equal', 'box')

        ax2.set_title('Zoom Regione Centrale [-30, 30] mm', fontsize=11)
        ax2.set_xlabel('X Piano Focale (mm)')
        ax2.set_ylabel('Y Piano Focale (mm)')
        ax2.set_xlim(-30, 30)
        ax2.set_ylim(-30, 30)
        ax2.grid(True, linestyle='--', alpha=0.4)
        ax2.set_aspect('equal', 'box')

        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=1, vmax=num_shells))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=[ax1, ax2], fraction=0.03, pad=0.08)
        cbar.set_label('Numero Shell', fontsize=11)

        plt.show()

    dataframe = pd.DataFrame({
        'X_focal': x_focal,
        'Y_focal': y_focal,
        'Energy_keV': np.full_like(x_focal, energy_keV),
    })

    dataframe.to_csv(f'Theta{theta_deg}_Iota{iota_deg}_Reflectivity{apply_reflectivity}_Geometry{apply_geometry}_Tilts{ApplyTilts}.csv', index=False)


if __name__ == "__main__":

    parser = ArgumentParser(description="Trace Stray Light Pattern for XMM-Newton Mirrors")

    parser.add_argument('coords', type=float, nargs=2, help="Misalignment parameters: gamma, xi, delta, epsilon (in radians)")
    args = parser.parse_args()

    Theta, Iota = args.coords[0], args.coords[1]


    FocalLength = 7500.0
    L_mirror = 300.0
    
    #Theta = 1.0
    #Iota = 270.0
    
    Gamma = 0.0
    Xi = 0.0
    Energy = 1.0
    ApplyReflectivity = True
    ApplyTilts = False

    try:
        xmm_info = pd.read_csv('XMM_OpticsDesign.csv')
        df_sorted = xmm_info.dropna(subset=['Diameter (mm)']).copy()
        df_sorted['Radius'] = df_sorted['Diameter (mm)'] / 2.0
        df_sorted = df_sorted.sort_values(by='Radius', ascending=False)
        
        r_shells = df_sorted['Radius'].values
        thicknesses = df_sorted['thickness (mm)'].values
    except (FileNotFoundError, KeyError):
        r_shells = np.linspace(350.0, 150.0, 58)
        thicknesses = np.full_like(r_shells, 1.0)

    try:
        ref_map = pd.read_csv('reflexSim_xmm_Au.csv', comment='#')
    except FileNotFoundError:
        ref_map = None
        ApplyReflectivity = False

    
    if ApplyTilts:
        Gamma = np.random.uniform(0,30/3600,len(r_shells)-1)  # gamma in radianti, da 0 a 30 arcsec
        Xi = np.random.uniform(0,360,len(r_shells)-1)  # xi in radianti, da 0 a 360 gradi
    else:
        Gamma = np.zeros(len(r_shells)-1)
        Xi = np.zeros(len(r_shells)-1)


    trace_stray_light_1R_parametric(
        r_shells=r_shells, 
        thicknesses=thicknesses, 
        theta_deg=Theta, 
        iota_deg=Iota, 
        gamma_deg=Gamma,
        xi_deg=Xi,
        energy_keV=Energy,
        apply_reflectivity=ApplyReflectivity,
        map_ref=ref_map,
        apply_geometry = True,
        focal_length=FocalLength, 
        L=L_mirror, 
        num_photons=100000,
        PLOT = False
    )