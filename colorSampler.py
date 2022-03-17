class colorSampler:
    def __init__(self, noColors):
        if(noColors < 2): raise Exception('Cannot have less than 2 colors')
        self.noColors = noColors
        self.colors = [0]
        stepSize = 255 / (noColors-1)
        nextColor = stepSize
        while(nextColor <= 255):
            self.colors.append(nextColor)
            nextColor += stepSize



    def sample(self, color):
        index = (color/ 255) * (self.noColors - 1)
        if(index.is_integer()): return int(self.colors[int(index)])
        index = int(index)
        diff1 = color - self.colors[index]
        diff2 = self.colors[index + 1] - color
        return int(self.colors[index]) if diff1 < diff2 else int(self.colors[index+1])