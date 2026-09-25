from pathlib import Path
import base64, io
import numpy as np
import rasterio
from PIL import Image
from matplotlib import colormaps

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

def read_band(path):
    with rasterio.open(path) as ds:
        a = ds.read(1).astype("float64")
        nodata = ds.nodata
        if nodata is not None:
            a[a == nodata] = np.nan
    return a

def rgba_continuous(a, vmin, vmax, cmap_name):
    valid = np.isfinite(a)
    x = np.clip((a - vmin) / (vmax - vmin), 0, 1)
    rgba = (colormaps[cmap_name](np.nan_to_num(x, nan=0.0)) * 255).astype("uint8")
    rgba[..., 3] = np.where(valid, 230, 0).astype("uint8")
    return rgba

def rgba_mask(a, rgb):
    valid = np.isfinite(a) & (a > 0)
    rgba = np.zeros(a.shape + (4,), dtype="uint8")
    rgba[..., 0] = rgb[0]
    rgba[..., 1] = rgb[1]
    rgba[..., 2] = rgb[2]
    rgba[..., 3] = np.where(valid, 210, 0).astype("uint8")
    return rgba

def save_svg(rgba, out_name, maxdim=900):
    im = Image.fromarray(rgba, "RGBA")
    scale = min(1.0, maxdim / max(im.size))
    if scale < 1:
        im = im.resize((max(1, int(im.width*scale)), max(1, int(im.height*scale))), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=70, method=2)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{im.width}" height="{im.height}" viewBox="0 0 {im.width} {im.height}"><image href="data:image/webp;base64,{b64}" width="{im.width}" height="{im.height}"/></svg>'
    (ASSETS / out_name).write_text(svg, encoding="utf-8")

ndvi = read_band(DATA / "Green_Kolkata_NDVI_2025.tif")
save_svg(rgba_continuous(ndvi, -0.20, 0.80, "RdYlGn"), "ndvi_overlay.svg")

lst = read_band(DATA / "Green_Kolkata_LST_2025.tif")
save_svg(rgba_continuous(lst, 24.0, 40.0, "inferno"), "lst_overlay.svg")

green = read_band(DATA / "Green_Kolkata_Green_Cover_2025.tif")
save_svg(rgba_mask(green, (40, 180, 80)), "green_overlay.svg")

water = read_band(DATA / "Green_Kolkata_Water_Bodies_2025.tif")
save_svg(rgba_mask(water, (45, 145, 235)), "water_overlay.svg")

print("Built web raster overlays in assets/")
