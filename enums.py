
import enum

class Modes(enum.Enum): # Face Detection mode
    NOTHING = 0
    DETECT_ALL_FACES = 1
    SCALE_TO_FIRST_FACE = 2
    BRIGHTEN_FIRST_FACE = 3
    DETECT_FACES_WITH_EYES = 4
    DETECT_FACE_FEATURES = 5


class Simode(enum.Enum): # Simulation mode
    BCM = 0 # Blur color modulated
    BSM = 1 # Blur size modulated
    ACM = 2 # Array color modulated
    ASM = 3 # Array size modulated