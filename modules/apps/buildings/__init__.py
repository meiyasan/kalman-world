"""
``kalman.apps.buildings`` — the air of a room.

- ``co2`` — :class:`RoomCO2Model`, a CO2 mass balance on the cells of a room
  (mixing, ventilation, occupants), :func:`l_shaped_room`, and
  :func:`empty_at_night`, what is known of the occupants as a sensor, and
  :func:`ventilation_from_decay`, the room's air changes from an evening's fall.
"""
from .co2 import (Opening, RoomCO2Model, air_conditioner, door, empty_at_night, l_shaped_room,
                  ventilation_from_decay, window)

VARIANTS = {
    "co2": RoomCO2Model,
}

__all__ = ["Opening", "air_conditioner", "door", "window", "RoomCO2Model", "empty_at_night", "l_shaped_room", "ventilation_from_decay", "VARIANTS"]
