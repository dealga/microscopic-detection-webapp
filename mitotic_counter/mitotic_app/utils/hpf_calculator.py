# utils/hpf_calculator.py
from PIL import Image
from PIL.TiffImagePlugin import IFDRational
import os
import tempfile
from django.conf import settings

def extract_strict_tiff_metadata(image_path):
    """Extract metadata from TIFF image"""
    with Image.open(image_path) as img:
        tags = img.tag_v2
        metadata = {
            "ImageWidth": img.width,
            "ImageLength": img.height,
            "XResolution": tags.get(282),
            "YResolution": tags.get(283),
            "ResolutionUnit": tags.get(296)  # 1=None, 2=inches, 3=centimeters
        }
        for key in ["XResolution", "YResolution", "ResolutionUnit"]:
            if metadata[key] is None:
                raise ValueError(f"Missing required metadata field: {key}")
            
            if isinstance(metadata[key], IFDRational):
                metadata[key] = float(metadata[key])
            elif isinstance(metadata[key], tuple):
                metadata[key] = float(metadata[key][0]) / metadata[key][1]
        return metadata

def get_microns_per_pixel(metadata):
    """Calculate microns per pixel based on image metadata"""
    unit = metadata["ResolutionUnit"]
    x_res = metadata["XResolution"]
    y_res = metadata["YResolution"]
    if unit == 2:  # inches
        x_mpp = 25400 / x_res
        y_mpp = 25400 / y_res
    elif unit == 3:  # centimeters
        x_mpp = 10000 / x_res
        y_mpp = 10000 / y_res
    else:
        raise ValueError("Unsupported or undefined ResolutionUnit")
    return x_mpp, y_mpp

def hpf_dimensions_in_pixels(x_mpp, y_mpp):
    """Calculate HPF dimensions in pixels"""
    hpf_area_mm2 = 0.237
    hpf_area_um2 = hpf_area_mm2 * 1e6
    aspect_ratio = 4 / 3
    hpf_width_um = (hpf_area_um2 * aspect_ratio) ** 0.5
    hpf_height_um = hpf_width_um / aspect_ratio
    hpf_width_px = int(hpf_width_um / x_mpp)
    hpf_height_px = int(hpf_height_um / y_mpp)
    return hpf_width_px, hpf_height_px

def estimate_hpf_count(image_width, image_height, step_x, step_y, hpf_w, hpf_h):
    """Estimate number of HPFs in the scanned area"""
    total_scan_area = (image_width // step_x) * (image_height // step_y) * (step_x * step_y)
    hpf_area = hpf_w * hpf_h
    return total_scan_area // hpf_area

def mitoses_per_10_hpf(mitotic_count, hpf_count):
    """Calculate mitoses per 10 HPF"""
    return 0 if hpf_count == 0 else (mitotic_count / hpf_count) * 10

def compute_mitotic_density_from_image(image_path, mitotic_count, step_x, step_y):
    """Compute mitotic density and related metrics from image"""
    # Print for debugging
    print(f"Computing HPF metrics for image: {image_path}")
    print(f"Current mitotic count: {mitotic_count}")
    print(f"Scan parameters: step_x={step_x}, step_y={step_y}")
    
    try:
        metadata = extract_strict_tiff_metadata(image_path)
        width, height = metadata["ImageWidth"], metadata["ImageLength"]
        x_mpp, y_mpp = get_microns_per_pixel(metadata)
        hpf_w, hpf_h = hpf_dimensions_in_pixels(x_mpp, y_mpp)
        hpf_count = estimate_hpf_count(width, height, step_x, step_y, hpf_w, hpf_h)
        density = mitoses_per_10_hpf(mitotic_count, hpf_count)
        
        result = {
            "image": image_path,
            "x_mpp": x_mpp,
            "y_mpp": y_mpp,
            "hpf_size": (hpf_w, hpf_h),
            "total_hpfs": hpf_count,
            "mitoses_per_10_hpf": round(density, 2)
        }
        
        # Print for debugging
        print("HPF calculation results:")
        for key, value in result.items():
            print(f"  {key}: {value}")
            
        return result
    except Exception as e:
        print(f"Error in HPF calculation: {e}")
        raise

def get_tumor_grade(mitoses_per_10_hpf):
    """Determine tumor grade based on mitoses per 10 HPF
    
    Grade 1: 0-7 mitoses per 10 HPF
    Grade 2: 8-14 mitoses per 10 HPF
    Grade 3: 15+ mitoses per 10 HPF
    """
    print(f"Determining tumor grade for {mitoses_per_10_hpf} mitoses per 10 HPF")
    if mitoses_per_10_hpf < 8:
        return 1
    elif mitoses_per_10_hpf < 15:
        return 2
    else:
        return 3