"""
kalman.apps.calorimeter — sampling-calorimeter domain.

- geometry — :class:`CaloLayer`, :class:`CalorimeterGeometry`, materials;
- sampler  — :class:`CalorimeterSampler` and shower profiles;
- models   — linear / electromagnetic / hadronic reconstruction variants.
"""
from .geometry import (CaloLayer, CalorimeterGeometry, moliere_radius_mm, MATERIALS, MATERIAL_COLORS, CalorimeterState4, CalorimeterState5, CalorimeterObservable2, CalorimeterObservable3)
from .sampler import (CalorimeterSampler, em_shower_profile, hadronic_shower_profile, build_measurements)
from .models import (CalorimeterLinearModel, CalorimeterElectromagneticModel, CalorimeterHadronicModel, ECAL_PRESETS, HCAL_PRESETS)

VARIANTS = {
    "linear": CalorimeterLinearModel,
    "electromagnetic": CalorimeterElectromagneticModel,
    "hadronic": CalorimeterHadronicModel,
}

__all__ = [
    "CaloLayer",
    "CalorimeterGeometry",
    "moliere_radius_mm",
    "MATERIALS",
    "MATERIAL_COLORS",
    "ECAL_PRESETS",
    "HCAL_PRESETS",
    "CalorimeterState4",
    "CalorimeterState5",
    "CalorimeterObservable2",
    "CalorimeterObservable3",
    "CalorimeterSampler",
    "em_shower_profile",
    "hadronic_shower_profile",
    "build_measurements",
    "CalorimeterLinearModel",
    "CalorimeterElectromagneticModel",
    "CalorimeterHadronicModel",
    "VARIANTS",
]
