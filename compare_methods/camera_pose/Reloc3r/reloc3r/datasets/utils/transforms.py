# transform utilities adapted from DUSt3R
import torchvision.transforms as tvf
from reloc3r.utils.image import ImgNorm

# define the standard image transforms
ColorJitter = tvf.Compose([tvf.ColorJitter(0.5, 0.5, 0.5, 0.1), ImgNorm])