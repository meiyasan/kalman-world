"""
``kalman.apps.buildings`` — the air of a room.

- ``co2`` — :class:`RoomCO2Model`, a CO2 mass balance on the cells of a room
  (mixing, ventilation, occupants), :func:`l_shaped_room`, and
  :func:`empty_at_night`, what is known of the occupants as a sensor.
"""
from .co2 import RoomCO2Model, empty_at_night, l_shaped_room

VARIANTS = {
    "co2": RoomCO2Model,
}

__all__ = ["RoomCO2Model", "empty_at_night", "l_shaped_room", "VARIANTS"]
