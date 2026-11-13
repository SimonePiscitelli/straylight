import pandas as pd
import os
import numpy as np
import cv2
from astropy.io import fits
from skimage.draw import polygon, disk
from skimage.restoration import inpaint
from scipy import ndimage
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs.utils import proj_plane_pixel_scales
import argparse
from skimage.feature import canny
from skimage.morphology import disk as disk_ski
from skimage.morphology import remove_small_objects

# Costanti e dizionari di default ereditati dalla logica originaria


APPLY_NORMU8 = False
TRANS_METHOD = 'gamma'

#####################################################################
##### -- FUNZIONI PER LA RIMOZIONE DELLA STRAY LIGHT (da canny) -- ###
#####################################################################

def zoom_in(data, xp, yp, l):
    return data[yp-l:yp+l, xp-l:xp+l]

def fill_chip_gaps(data_image, mask_poly, min_threshold=1e-5, method='biharmonic'):
    """Individua e riempie i pixel vuoti/gap dei chip all'interno della ROI."""
    filled_image = data_image.copy()
    gap_mask = (mask_poly) & (filled_image <= min_threshold)
    
    if not np.any(gap_mask):
        print("[INFO] Nessun chip gap rilevato all'interno della ROI.")
        return filled_image, gap_mask

    print(f"[INFO] Trovati {np.sum(gap_mask)} pixel vuoti (chip gap) nella ROI. Esecuzione Inpainting ({method})...")

    if method == 'biharmonic':
        filled_image = inpaint.inpaint_biharmonic(filled_image, gap_mask)
    elif method == 'telea':
        gap_mask_uint8 = gap_mask.astype(np.uint8) * 255
        norm_factor = filled_image.max() if filled_image.max() > 0 else 1.0
        img_32f = (filled_image / norm_factor).astype(np.float32)
        inpainted_32f = cv2.inpaint(img_32f, gap_mask_uint8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        filled_image = inpainted_32f * norm_factor

    return filled_image, gap_mask

def extract_from_region(region_file, data, header, xp, yp, l, method='keep'):
    wcs = WCS(header)
    mask_polygon = np.zeros(data.shape, dtype=bool)
    regions_found = []

    x_offset = xp - l
    y_offset = yp - l

    if not os.path.exists(region_file):
        return mask_polygon, regions_found

    if method == 'keep':
        with open(region_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line.startswith(('polygon', 'circle')):
                    region_type = 'polygon' if line.startswith('polygon') else 'circle'
                    coords_str = line.split('#')[0].strip().split('(')[-1].split(')')[0]
                    
                    marker = ''
                    if region_type == 'circle':
                        if coords_str.endswith('\"'): marker = 'arcsec'
                        elif coords_str.endswith('\''): marker = 'arcmin'
                        coords_str = coords_str.replace('\"', '').replace('\'', '')
                    
                    coords = list(map(float, coords_str.split(',')))
                elif line.startswith(('-polygon', '-circle')):
                    region_type = '-polygon' if line.startswith('-polygon') else '-circle'
                    coords_str = line.split('#')[0].strip().split('(')[-1].split(')')[0]
                    marker = ''
                    if 'circle' in region_type:
                        if coords_str.endswith('\"'): marker = 'arcsec'
                        elif coords_str.endswith('\''): marker = 'arcmin'
                        coords_str = coords_str.replace('\"', '').replace('\'', '')
                    coords = list(map(float, coords_str.split(',')))
                else:
                    continue
                
                if region_type == 'polygon':
                    x_coords_fk5 = coords[::2]
                    y_coords_fk5 = coords[1::2]
                    skycoords = SkyCoord(x_coords_fk5, y_coords_fk5, unit="deg")
                    x_coords, y_coords = wcs.world_to_pixel(skycoords)
                    rr, cc = polygon(y_coords, x_coords, shape=data.shape)
                    mask_polygon[rr, cc] = True

                    x_local = x_coords - x_offset
                    y_local = y_coords - y_offset

                    regions_found.append({
                        'type': 'polygon',
                        'is_exclusion': False,
                        'x': x_local,
                        'y': y_local
                    })

                elif region_type == 'circle':
                    x_center_fk5, y_center_fk5, radius_val = coords
                    if marker == 'arcsec': radius_deg = radius_val * u.arcsec.to(u.deg)
                    elif marker == 'arcmin': radius_deg = radius_val * u.arcmin.to(u.deg)
                    else: radius_deg = radius_val
                        
                    skycoords = SkyCoord(x_center_fk5, y_center_fk5, unit="deg")
                    x_center, y_center = wcs.world_to_pixel(skycoords)
                    scales = proj_plane_pixel_scales(wcs) 
                    pixel_scale_deg = np.mean(scales)
                    radius_pixels = radius_deg / pixel_scale_deg
                    
                    rr, cc = disk((y_center, x_center), radius_pixels, shape=data.shape)
                    mask_polygon[rr, cc] = True

                    x_local = x_center - x_offset
                    y_local = y_center - y_offset

                    regions_found.append({
                        'type': 'circle',
                        'is_exclusion': False,
                        'x': np.array([x_local]),
                        'y': np.array([y_local]),
                        'r_pixels': radius_pixels
                    })
                    
                elif region_type == '-polygon':
                    x_coords_fk5_negative = coords[::2]
                    y_coords_fk5_negative = coords[1::2]
                    skycoords = SkyCoord(x_coords_fk5_negative, y_coords_fk5_negative, unit="deg")
                    x_coords_neg, y_coords_neg = wcs.world_to_pixel(skycoords)

                    rr_neg, cc_neg = polygon(y_coords_neg, x_coords_neg, shape=data.shape)
                    mask_polygon[rr_neg, cc_neg] = 0

                    x_local = x_coords_neg - x_offset
                    y_local = y_coords_neg - y_offset

                    regions_found.append({'type': 'polygon', 'is_exclusion': True, 'x': np.array([x_local]), 'y': np.array([y_local])})

                elif region_type == '-circle':
                    x_center_fk5_negative, y_center_fk5_negative, radius_val_negative = coords
                    if marker == 'arcsec': radius_deg_negative = radius_val_negative * u.arcsec.to(u.deg)
                    elif marker == 'arcmin': radius_deg_negative = radius_val_negative * u.arcmin.to(u.deg)
                    else: radius_deg_negative = radius_val_negative 
                    skycoords_negative = SkyCoord(x_center_fk5_negative, y_center_fk5_negative, unit="deg")
                    x_center_negative, y_center_negative = wcs.world_to_pixel(skycoords_negative)
                    scales = proj_plane_pixel_scales(wcs) 
                    radius_pixels_negative = radius_deg_negative / np.mean(scales)
                    rr_neg, cc_neg = disk((y_center_negative, x_center_negative), radius_pixels_negative, shape=data.shape)
                    mask_polygon[rr_neg, cc_neg] = 0 

                    x_local = x_center_negative - x_offset
                    y_local = y_center_negative - y_offset
                    
                    regions_found.append({'type': 'circle', 'is_exclusion': True, 'x': np.array([x_local]), 'y': np.array([y_local]), 'r_pixels': radius_pixels_negative})

    elif method == 'anti':
        mask_polygon, regions_found = extract_from_region(region_file, data, header, xp, yp, l, method='keep')
        mask_polygon = ~mask_polygon
        
    return mask_polygon, regions_found

def apply_dynamic_canny_core(
    ima_fits_file, 
    mask_poly, 
    k_high,
    low_ratio,
    s_space,
    mask_bkg=None,
    transform_method=TRANS_METHOD, 
    apply_normU8=APPLY_NORMU8,
    log_scale=100.0, 
    gamma=2
):
    """Applica Canny calcolando le soglie v_min e v_max in base al database."""
    if transform_method == 'log':
        data_transformed = np.log1p(ima_fits_file * log_scale) / np.log1p(log_scale)
    elif transform_method in ['gamma', 'power', 'exponential']:
        data_transformed = np.power(ima_fits_file, gamma)
    else:
        data_transformed = ima_fits_file

    data_polygon = np.where(mask_poly, data_transformed, 0)

    if apply_normU8:
        minus, maximum = np.min(data_transformed), np.max(data_transformed)
        if maximum - minus == 0:
            return np.zeros(data_transformed.shape, dtype=np.uint8)
        
        norm_linear = (data_transformed - minus) / (maximum - minus)    
        normalized_data = (data_polygon * 255).astype(np.uint8)
        
        gx = cv2.Sobel(normalized_data, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(normalized_data, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude = np.hypot(gx, gy)
    else:
        normalized_data = data_polygon.copy()
        gx = ndimage.sobel(normalized_data, axis=0)    
        gy = ndimage.sobel(normalized_data, axis=1)
        grad_magnitude = np.hypot(gx, gy)

    if mask_bkg is not None and np.any(mask_bkg):
        gx_full = cv2.Sobel(data_transformed, cv2.CV_64F, 1, 0, ksize=3)
        gy_full = cv2.Sobel(data_transformed, cv2.CV_64F, 0, 1, ksize=3)
        grad_magnitude_full = np.hypot(gx_full, gy_full)
        grad_noise_sample = grad_magnitude_full[mask_bkg]
    else:
        grad_noise_sample = grad_magnitude[mask_poly & (normalized_data > 0)]

    grad_distribution_ROI = grad_magnitude[grad_magnitude > 0]
    if len(grad_distribution_ROI) == 0:
        grad_distribution_ROI = grad_magnitude

    p_max = np.percentile(grad_distribution_ROI, k_high)
    p_min = low_ratio * p_max

    if apply_normU8:
        edges = cv2.Canny(data_transformed, p_min, p_max)
    else:
        edges = canny(data_transformed, 
                      sigma=s_space,
                      low_threshold=p_min, 
                      high_threshold=p_max)

    return edges

def run_pipeline_execution(data, mask_poly, params, mask_bkg=None, apply_normU8=APPLY_NORMU8):
    edges = apply_dynamic_canny_core(
        data, mask_poly, 
        params['p_max'], params['l_ratio'], 
        params['s_space'],
        mask_bkg=mask_bkg,
        transform_method=TRANS_METHOD, 
        apply_normU8=apply_normU8,
        log_scale=100, 
        gamma=params.get('gamma', 2.0)
    )

    if apply_normU8:
        k_dilation = np.ones((3, 3), np.uint8)
        k_erosion = np.ones((3, 3), np.uint8)
        edges_refined = cv2.dilate(edges.astype(np.uint8), k_dilation, iterations=1)
        edges_refined = cv2.erode(edges_refined, k_erosion, iterations=1)

        contours, _ = cv2.findContours(edges_refined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_arcs = np.zeros_like(data)
        cv2.drawContours(mask_arcs, contours, -1, 255, thickness=cv2.FILLED)
    else:
        edges_refined = edges.copy()
        edges_filtered = remove_small_objects(edges_refined, min_size=int(params['m']), connectivity=params['connectivity'])
        
        n_bin_dilate = int(params['n_iter_dil'])
        if n_bin_dilate > 0:
            struct_element2 = np.ones((params['ksize'], params['ksize']), dtype=bool)
            mask_arcs = ndimage.binary_dilation(
                edges_filtered, 
                structure=struct_element2, 
                iterations=n_bin_dilate
            )
        else:
            mask_arcs = edges_filtered

    mask_arcs = ndimage.binary_fill_holes(mask_arcs)

    result_mask = np.zeros_like(data, dtype=bool)
    result_mask[mask_arcs > 0] = True

    leftover_zoomed = np.where(~result_mask, data, 0)
    leftover_ROI = np.where(mask_poly, leftover_zoomed, 0)
    original_BKG = np.where(mask_bkg, data, 0) if mask_bkg is not None else np.zeros_like(data)
    result_to_show = np.where(~mask_poly, data, 0) + np.where(mask_poly, leftover_zoomed, 0)

    return leftover_zoomed, mask_arcs, leftover_ROI, original_BKG, result_to_show


#####################################################################
##### -- SCRIPT PRINCIPALE DI ESECUZIONE CAS -- ######################
#####################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modulo CAS: Rimozione della stray light pilotata tramite database di parametri.")
    
    parser.add_argument("--ima_fits", required=True, type=str, help="Percorso del file FITS dell'immagine.")
    parser.add_argument("--region_file", required=True, type=str, help="Percorso del file regione DS9 (.reg).")
    parser.add_argument("--database", default="DataBase_CAS.csv", type=str, help="Database CSV delle configurazioni.")
    parser.add_argument("--config", type=str, default=None, help="Nome della configurazione da usare (es. Conf1). Se assente, usa l'ultima.")
    parser.add_argument("--show_results", action="store_true", help="Se attivato, mostra a schermo i risultati dell'analisi.")
    parser.add_argument("--output_dir", default="outputs_cas", type=str, help="Cartella di salvataggio dei file FITS.")

    args = parser.parse_args()

    # 1. Caricamento del database
    if not os.path.exists(args.database):
        raise FileNotFoundError(f"[ERRORE] Il database '{args.database}' non esiste. Esegui prima 'set_params.py'.")

    db = pd.read_csv(args.database)
    if db.empty:
        raise ValueError(f"[ERRORE] Il database '{args.database}' è vuoto.")

    # 2. Selezione della configurazione
    if args.config:
        row = db[db['Configuration'] == args.config]
        if row.empty:
            raise ValueError(f"[ERRORE] Configurazione '{args.config}' non trovata nel database.")
        selected_row = row.iloc[0]
        conf_name = args.config
    else:
        selected_row = db.iloc[-1]
        conf_name = selected_row['Configuration']
        print(f"[INFO] Nessuna configurazione specificata. Utilizzo l'ultima riga: {conf_name}")

    print("=" * 60)
    print(f" Esecuzione pipeline CAS - Configurazione: {conf_name}")
    print("=" * 60)
    for col in db.columns:
        if col != 'Configuration':
            print(f"   {col}: {selected_row[col]}")
    print("=" * 60)

    # Estrazione dei parametri direttamente dal database
    params = {
        'l_ratio': float(selected_row['l_ratio']),
        'p_max': float(selected_row['p_max']),
        's_space': float(selected_row['s_space']),
        'm': float(selected_row['m']),
        'n_iter_dil': int(selected_row['n_iter_dil']),
        'xp': float(selected_row['xp']),
        'yp': float(selected_row['yp']),
        'connectivity': int(selected_row.get('connectivity', 2)),
        'ksize': int(selected_row.get('ksize', 3)),
        'zoom': float(selected_row.get('zoom', 300.0)),
        'gamma': float(selected_row.get('gamma', 2))
    }

    # 3. Apertura immagine FITS
    if not os.path.exists(args.ima_fits):
        raise FileNotFoundError(f"[ERRORE] File FITS '{args.ima_fits}' non trovato.")

    with fits.open(args.ima_fits) as hdul:
        data = hdul[0].data.copy()
        header = hdul[0].header

    # Utilizzo delle coordinate XP, YP e L ricavate dal database o dai default
    xp_val = int(params['xp'])
    yp_val = int(params['yp'])
    zoom_val = int(params['zoom'])

    data_zoomed = zoom_in(data, xp=xp_val, yp=yp_val, l=zoom_val)

    # 4. Estrazione della regione ROI
    mask_roi, _ = extract_from_region(args.region_file, data, header, xp=xp_val, yp=yp_val, l=zoom_val, method='keep')
    mask_roi_zoomed = zoom_in(mask_roi, xp=xp_val, yp=yp_val, l=zoom_val)

    # 5. Esecuzione della pipeline Canny
    leftover_zoomed, mask_arcs, leftover_ROI, original_BKG, result_to_show = run_pipeline_execution(
        data_zoomed, 
        mask_roi_zoomed, 
        params, 
        mask_bkg=None, 
        apply_normU8=APPLY_NORMU8
    )

    # 6. Salvataggio dei risultati in file FITS separati
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = os.path.basename(args.ima_fits).replace('.fits', '')

    clean_img_path = os.path.join(args.output_dir, f"{base_name}_{conf_name}_cleaned.fits")
    mask_path = os.path.join(args.output_dir, f"{base_name}_{conf_name}_mask.fits")

    hdu_clean = fits.PrimaryHDU(result_to_show, header=header)
    hdu_clean.writeto(clean_img_path, overwrite=True)

    hdu_mask = fits.PrimaryHDU(mask_arcs.astype(np.uint8), header=header)
    hdu_mask.writeto(mask_path, overwrite=True)

    print(f"[SUCCESS] Saved in directory '{args.output_dir}':")
    print(f"          -> Cleaned Image: {clean_img_path}")
    print(f"          -> Mask Arcs:    {mask_path}")

    # 7. Visualizzazione opzionale tramite parametro di controllo
    if args.show_results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        im0 = axes[0].imshow(result_to_show, cmap='hot', origin='lower')
        axes[0].set_title(f"Cleaned image ({conf_name})", fontweight='bold')
        axes[0].set_xlabel('Pixel X', fontsize = 14)
        axes[0].set_ylabel('Pixel Y', fontsize = 14)
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(mask_arcs, cmap='gray', origin='lower')
        axes[1].set_title(f"Maschera Archi ({conf_name})", fontweight='bold')
        axes[1].set_xlabel('Pixel X')
        axes[1].set_ylabel('Pixel Y')
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()