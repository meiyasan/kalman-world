"""
``kalman.apps.buildings`` — the air of a room.

- ``co2`` — :class:`RoomCO2Model`, a CO2 mass balance on the cells of a room
  (mixing, ventilation, occupants), and :func:`l_shaped_room`.
"""
from .co2 import RoomCO2Model, l_shaped_room

VARIANTS = {
    "co2": RoomCO2Model,
}

__all__ = ["RoomCO2Model", "l_shaped_room", "VARIANTS"]
