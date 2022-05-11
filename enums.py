
import enum

# All possible modes must be properly defined

class Modes(enum.Enum): # Face Detection mode
    NOTHING = 0 #DLR or control group
    DETECT_ALL_FACES = 1
    BRIGHTEN_FIRST_FACE = 2
    DETECT_FACES_WITH_EYES = 3
    DETECT_FACE_FEATURES = 4
    VJFR_ROI_M = 5 ###
    SFR_ROI_M = 6
    VJFR_ROI_C = 7 ###
    SFR_ROI_HE = 8
    SFR_ROI_M_TD = 9
    SFR_ROI_M_ER = 10
    VJFR_ROI_M_TD = 11 ###
    VJFR_ROI_M_ER = 12 ###
    VJFR_ROI_HE = 13 ###


class Simode(enum.Enum): # Simulation mode
    BCM = 0 # Blur color modulated
    BSM = 1 # Blur size modulated
    ACM = 2 # Array color modulated
    ASM = 3 # Array size modulated