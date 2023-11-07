from .aero import Aero
from .seanet import Seanet
from .discriminators import Discriminator, MultiPeriodDiscriminator, MultiScaleDiscriminator
import yaml
from munch import Munch

def get_model_superresolution():
    args = yaml.safe_load(open('model/superresolution/aero_4-16_512_64.yaml', 'r'))
    args = Munch(args)
    if args.model == 'aero': 
        generator = Aero(**args.aero)
    elif args.model == 'seanet':
        generator = Seanet(**args.seanet)
    return generator

def get_discriminator():
    args = yaml.safe_load(open('model/superresolution/aero_4-16_512_64.yaml', 'r'))
    args = Munch(args)
    models = {}
    if 'adversarial' in args and args.adversarial:
        if 'msd_melgan' in args.discriminator_models:
            discriminator = Discriminator(**args.melgan_discriminator)
            models.update({'msd_melgan': discriminator})
        if 'msd_hifi' in args.discriminator_models:
            msd = MultiScaleDiscriminator(**args.msd)
            models.update({'msd': msd})
        if 'mpd' in args.discriminator_models:
            mpd = MultiPeriodDiscriminator(**args.mpd)
            models.update({'mpd': mpd})
        if 'hifi' in args.discriminator_models:
            mpd = MultiPeriodDiscriminator(**args.mpd)
            msd = MultiScaleDiscriminator(**args.msd)
            models.update({'mpd': mpd, 'msd': msd})
    return models