#  Copyright (C) 2026
#  Smithsonian Astrophysical Observatory, Cambridge, MA, USA
#  For conditions of distribution and use, see copyright notice in "copyright"
#
# No shared fixtures needed at the moment: daq.py and daq_gui.py both defer
# their nidaqmx/isw-instruments imports to inside
# vibetest_daq.acquisition.run_acquisition(), so importing either module (or
# vibetest_daq.acquisition itself) needs no hardware-driver faking.
