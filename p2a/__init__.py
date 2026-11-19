"""pulse2abp: sample arterial pressure waveforms from the pulse (PPG).

Fixed conventions used everywhere in this package:
  * signals are 10 s at 125 Hz -> 1250 samples
  * ABP is modelled in normalised units  z = (mmHg - ABP_MEAN) / ABP_STD
  * PPG is z-scored per segment before it reaches a model
"""

FS = 125          # Hz
SEG_LEN = 1250    # samples (10 s)
ABP_MEAN = 90.0   # mmHg, fixed (not fitted) so every dataset shares one scale
ABP_STD = 30.0    # mmHg
HYPO_MAP = 65.0   # mmHg, hypotension threshold on mean arterial pressure


def abp_to_z(abp_mmhg):
    return (abp_mmhg - ABP_MEAN) / ABP_STD


def z_to_abp(z):
    return z * ABP_STD + ABP_MEAN
