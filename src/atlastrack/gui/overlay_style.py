"""How every layer drawn *on top of* the slide must blend.

This has now been got wrong three times, in three different layer types, with the
same symptom each time: the layer exists, is visible, holds the right data, sits
above the image in the stack - and draws nothing.

napari's default blending is ``translucent``, which **depth-tests**. An overlay
placed at the same z as the slide image loses that test and is culled, wholly or
in part, depending on the driver and the zoom. ``translucent_no_depth`` composites
over whatever is below instead, which is what an annotation drawn on a picture
actually wants.

The failure is invisible to every check short of looking at rendered pixels:
``layer.visible`` is True, ``len(layer.data)`` is right, and the layer is at the
top of the stack. So it survives code review, unit tests and headless screenshots
(which render nothing at all), and only shows up when a person looks at the canvas
and says "the landmarks are not there".

Every ``add_points`` / ``add_shapes`` / ``add_labels`` that decorates the slide
passes ``blending=OVERLAY_BLENDING``. ``test_every_overlay_layer_composites_over_
the_slide`` fails if a new one forgets.
"""
from __future__ import annotations

#: Blending for any layer drawn over the slide image. See the module docstring -
#: napari's default (``translucent``) depth-tests and silently culls the layer.
OVERLAY_BLENDING = "translucent_no_depth"
