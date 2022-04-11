import numpy as np

def modulate(value, max = 255, newMax = 15):
    valueN = value * (newMax/ max) # Linearly normzalized value
    valueN = np.rint(valueN) # round to nearest integer
    value = valueN * (max/newMax) # Modulated value
    return np.uint8(value)


def main():
    for i in range(20):
        print(str(i)+" : "+str(modulate(i, 20, 15)))

if __name__ == "__main__":
    main()